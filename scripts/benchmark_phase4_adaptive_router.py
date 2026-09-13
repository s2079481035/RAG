"""Benchmark the frozen operational query-only Router with batch size one."""

from __future__ import annotations

import argparse
import json
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np

from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
from phase2_model import configure_cublas_workspace
from phase4_adaptive_router import file_sha256, read_jsonl, write_jsonl_atomic
from train_phase4_adaptive_router import collate_batch, replace_output_head


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "adaptive_rag_router.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["training"]["operational_seed"])
    run_dir = ROOT / config["output"]["experiment_dir"] / f"seed{seed}"
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete" or run_manifest.get("seed") != seed:
        raise ValueError("Operational Router run is not complete")
    if resolved["resolved"]["model_input_fields"] != ["question"]:
        raise ValueError("Operational Router is not question-only")

    evaluation_dir = run_dir / "evaluation" / "dev_policy"
    frozen_path = evaluation_dir / "predictions.jsonl"
    frozen = {row["question_id"]: row for row in read_jsonl(frozen_path)}
    source_path = ROOT / config["data"]["router_dir"] / "dev_policy.jsonl"
    source = list(read_jsonl(source_path))
    if set(frozen) != {row["question_id"] for row in source}:
        raise ValueError("Frozen Router predictions are incomplete")

    result_dir = ROOT / config["output"]["result_dir"]
    output_path = result_dir / "router_latency.jsonl"
    manifest_path = result_dir / "router_latency_manifest.json"
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Router benchmark: {existing}")

    configure_cublas_workspace(config["training"]["cublas_workspace_config"])
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    backbone = args.model or resolved["resolved"]["backbone"]
    load_kwargs = {
        "local_files_only": resolved["resolved"]["local_files_only"]
        and not args.allow_download
    }
    tokenizer = AutoTokenizer.from_pretrained(backbone, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, **load_kwargs)
    replace_output_head(model, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(run_manifest["checkpoint_path"], map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    collate = partial(
        collate_batch,
        tokenizer=tokenizer,
        max_length=int(config["model"]["max_length"]),
    )

    def infer_one(row: dict, measured: bool):
        encoded, _ = collate([row])
        input_tokens = int(encoded["attention_mask"].sum().item())
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            probability = torch.softmax(model(**encoded).logits, dim=-1)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not measured:
            return None
        replay = probability.cpu().numpy()
        saved = frozen[row["question_id"]]["route_probabilities"]
        difference = max(
            abs(float(replay[index]) - float(saved[name]))
            for index, name in enumerate(config["route_names"])
        )
        return {
            "question_id": row["question_id"],
            "split": "dev_policy",
            "router_latency_ms": elapsed_ms,
            "input_tokens": input_tokens,
            "maximum_probability_difference": difference,
            "batch_size": 1,
            "cuda_synchronized": bool(torch.cuda.is_available()),
        }

    warmup = 5
    for row in source[:warmup]:
        infer_one(row, measured=False)
    measured = []
    for index, row in enumerate(source, start=1):
        measured.append(infer_one(row, measured=True))
        if index % 250 == 0:
            logger.info("benchmarked Router %s/%s", index, len(source))
    maximum_difference = max(row["maximum_probability_difference"] for row in measured)
    tolerance = 0.0001
    if maximum_difference > tolerance:
        raise ValueError(f"Router replay drift {maximum_difference} exceeds {tolerance}")
    write_jsonl_atomic(output_path, measured)
    latencies = np.asarray([row["router_latency_ms"] for row in measured])
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "split": "dev_policy",
            "heldout_consulted": False,
            "operational_seed": seed,
            "model_input_fields": ["question"],
            "questions": len(measured),
            "warmup_questions": warmup,
            "batch_size": 1,
            "cuda_synchronized": bool(torch.cuda.is_available()),
            "latency": {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.quantile(latencies, 0.95)),
            },
            "maximum_probability_replay_difference": maximum_difference,
            "probability_replay_tolerance": tolerance,
            "checkpoint_sha256": file_sha256(Path(run_manifest["checkpoint_path"])),
            "source_predictions": portable_path(frozen_path, ROOT),
            "source_predictions_sha256": file_sha256(frozen_path),
            "output": portable_path(output_path, ROOT),
            "output_sha256": file_sha256(output_path),
            "git_commit": git_commit(ROOT),
            "environment": collect_environment(ROOT),
        },
    )
    logger.info("saved Router benchmark -> %s", output_path)


if __name__ == "__main__":
    main()

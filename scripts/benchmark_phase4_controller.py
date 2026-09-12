"""Batch-size-one latency benchmark for the frozen 2Wiki Controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np

try:
    from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
    from phase2_model import configure_cublas_workspace
    from phase3a_model import load_phase3a_model
    from train_phase2_controller import read_jsonl, write_jsonl_atomic
    from train_phase3a_controller import collate_batch, model_outputs, prepare_records
except ModuleNotFoundError:  # Imported as scripts.* in tests.
    from scripts.experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
    from scripts.phase2_model import configure_cublas_workspace
    from scripts.phase3a_model import load_phase3a_model
    from scripts.train_phase2_controller import read_jsonl, write_jsonl_atomic
    from scripts.train_phase3a_controller import collate_batch, model_outputs, prepare_records


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"
DEFAULT_CONTROLLER_CONFIG = ROOT / "configs" / "phase4" / "controller.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER_CONFIG)
    parser.add_argument("--backbone", help="Local frozen Controller backbone path")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    controller_config = json.loads(args.controller_config.read_text(encoding="utf-8"))
    seed = int(protocol["in_domain_policy"]["operational_seed"])
    run_dir = ROOT / "experiments" / "phase4" / "2wiki" / "final_evidence_aware" / f"seed{seed}"
    manifest_path = run_dir / "run_manifest.json"
    resolved_path = run_dir / "resolved_config.json"
    checkpoint_path = run_dir / "best.pt"
    prediction_path = run_dir / "evaluation" / "dev_policy" / "original_predictions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("seed") != seed:
        raise ValueError("Operational 2Wiki Controller is not a complete frozen run")
    resolved = resolved_config["resolved"]
    if resolved["baseline"] != "query_evidence" or resolved["representation"] != "score_aware_packing":
        raise ValueError("Unexpected operational Controller representation")
    if not resolved["coverage_auxiliary"] or float(resolved["coverage_lambda"]) != 0.1:
        raise ValueError("Unexpected operational Controller auxiliary task")

    output_dir = ROOT / "results" / "phase4" / "2wiki" / "llm_judge" / "dev_policy"
    output_path = output_dir / "critic_latency.jsonl"
    output_manifest = output_dir / "critic_latency_manifest.json"
    existing = [path for path in (output_path, output_manifest) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Controller latency benchmark: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_cublas_workspace(controller_config["training"]["cublas_workspace_config"])
    import torch
    from transformers import AutoTokenizer

    torch.use_deterministic_algorithms(True)
    configured_backbone = args.backbone or resolved["backbone"]
    load_kwargs = {"local_files_only": bool(resolved["local_files_only"]) and not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(configured_backbone, **load_kwargs)
    model, _ = load_phase3a_model(configured_backbone, load_kwargs, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    source_path = ROOT / "data" / "2wiki" / "controller" / "sentence_256" / "dev_policy.jsonl"
    source = read_jsonl(source_path)
    actionable = set(protocol["in_domain_policy"]["decision_stages"])
    source = [row for row in source if row["stage"] in actionable]
    if any(row.get("split") != "dev_policy" for row in source):
        raise ValueError("Controller benchmark may only read dev_policy")
    chunks = read_jsonl(ROOT / "data" / "2wiki" / "chunks" / "sentence_256.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    namespace = argparse.Namespace(baseline="query_evidence", representation="score_aware_packing")
    prepared = prepare_records(
        source,
        args=namespace,
        config=resolved_config,
        tokenizer=tokenizer,
        chunk_by_id=chunk_by_id,
    )
    collate = partial(
        collate_batch,
        tokenizer=tokenizer,
        max_length=int(controller_config["backbone"]["max_length"]),
    )
    frozen = {
        (row["question_id"], row["stage"]): float(row["stop_probability"])
        for row in read_jsonl(prediction_path)
        if row["stage"] in actionable
    }
    if len(frozen) != len(prepared):
        raise ValueError("Frozen dev_policy predictions are incomplete")

    def infer_one(item: dict, measured: bool) -> dict | None:
        encoded, _, _ = collate([item])
        input_tokens = int(encoded["attention_mask"].sum().item())
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            logits, _ = model_outputs(model, encoded, True)
            probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not measured:
            return None
        key = (item["question_id"], item["stage"])
        return {
            "question_id": item["question_id"],
            "split": "dev_policy",
            "stage": item["stage"],
            "controller_latency_ms": (time.perf_counter() - started) * 1000.0,
            "input_tokens": input_tokens,
            "replayed_stop_probability": probability,
            "frozen_stop_probability": frozen[key],
            "absolute_probability_difference": abs(probability - frozen[key]),
            "batch_size": 1,
            "cuda_synchronized": bool(torch.cuda.is_available()),
        }

    warmup = int(protocol["llm_judge"]["latency_warmup_states"])
    for item in prepared[:warmup]:
        infer_one(item, measured=False)
    measured = []
    for index, item in enumerate(prepared, start=1):
        measured.append(infer_one(item, measured=True))
        if index % 250 == 0:
            logger.info("benchmarked Controller %s/%s", index, len(prepared))
    maximum_difference = max(row["absolute_probability_difference"] for row in measured)
    tolerance = float(protocol["llm_judge"]["controller_probability_replay_tolerance"])
    if maximum_difference > tolerance:
        raise ValueError(f"Controller replay drift {maximum_difference} exceeds {tolerance}")
    write_jsonl_atomic(output_path, measured)
    latencies = np.asarray([row["controller_latency_ms"] for row in measured])
    write_json_atomic(
        output_manifest,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "split": "dev_policy",
            "heldout_consulted": False,
            "operational_seed": seed,
            "actionable_stages": protocol["in_domain_policy"]["decision_stages"],
            "terminal_stage_excluded": protocol["in_domain_policy"]["forced_final_stage"],
            "actionable_stage_decisions": len(measured),
            "warmup_decisions": warmup,
            "batch_size": 1,
            "cuda_synchronized": bool(torch.cuda.is_available()),
            "latency": {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.quantile(latencies, 0.95)),
            },
            "maximum_probability_replay_difference": maximum_difference,
            "probability_replay_tolerance": tolerance,
            "checkpoint": portable_path(checkpoint_path, ROOT),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "source_predictions": portable_path(prediction_path, ROOT),
            "source_predictions_sha256": file_sha256(prediction_path),
            "output": portable_path(output_path, ROOT),
            "output_sha256": file_sha256(output_path),
            "git_commit": git_commit(ROOT),
            "environment": collect_environment(ROOT),
        },
    )
    logger.info("saved Controller benchmark -> %s", output_path)


if __name__ == "__main__":
    main()

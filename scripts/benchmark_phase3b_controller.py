"""Fresh batch-size-one latency benchmark for the frozen Phase 3B Controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np

from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
from generate_phase3b_stage_answers import validate_test_gate
from phase2_model import configure_cublas_workspace
from phase3a_model import load_phase3a_model
from train_phase2_controller import read_jsonl, write_jsonl_atomic
from train_phase3a_controller import collate_batch, model_outputs, prepare_records


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", required=True, choices=["dev_policy", "test"])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.split == "test":
        validate_test_gate(config, args.config)
    output_dir = ROOT / "results" / "phase3b" / "latency"
    output_path = output_dir / f"{args.split}_controller.jsonl"
    manifest_path = output_dir / f"{args.split}_controller_manifest.json"
    existing = [path for path in [output_path, manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Controller benchmark: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = ROOT / "results" / "phase3b" / "base_controller.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    resolved_path = repo_path(base["resolved_config"])
    run_manifest_path = repo_path(base["run_manifest"])
    if file_sha256(resolved_path) != base["resolved_config_sha256"]:
        raise ValueError("Frozen Controller configuration changed")
    if file_sha256(run_manifest_path) != base["run_manifest_sha256"]:
        raise ValueError("Frozen Controller run manifest changed")
    resolved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise ValueError("Frozen Base Controller run is incomplete")
    resolved = resolved_config["resolved"]
    configure_cublas_workspace(resolved_config["training"]["cublas_workspace_config"])

    import torch
    from transformers import AutoTokenizer

    torch.use_deterministic_algorithms(True)
    load_kwargs = {"local_files_only": bool(resolved["local_files_only"]) and not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(resolved["backbone"], **load_kwargs)
    model, _ = load_phase3a_model(resolved["backbone"], load_kwargs, bool(resolved["coverage_auxiliary"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = repo_path(base["checkpoint"])
    if file_sha256(checkpoint) != base["checkpoint_sha256"]:
        raise ValueError("Frozen Base Controller checkpoint changed")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    source_split = "dev" if args.split == "dev_policy" else "test"
    variant = config["base_controller"]["variant"]
    source_path = ROOT / "data" / "phase2" / "controller" / variant / f"{source_split}.jsonl"
    source = read_jsonl(source_path)
    if args.split == "dev_policy":
        ids = read_ids(repo_path(config["dev_split"]["policy_ids"]))
        source = [row for row in source if row["question_id"] in ids]
    actionable = set(config["policy"]["decision_stages"])
    source = [row for row in source if row["stage"] in actionable]
    chunks = read_jsonl(ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    namespace = argparse.Namespace(baseline=resolved["baseline"], representation=resolved["representation"])
    prepared = prepare_records(source, args=namespace, config=resolved_config, tokenizer=tokenizer, chunk_by_id=chunk_by_id)
    collate = partial(collate_batch, tokenizer=tokenizer, max_length=int(resolved_config["backbone"]["max_length"]))

    frozen_predictions_path = repo_path(
        base["dev_predictions"] if args.split == "dev_policy" else base["test_predictions"]
    )
    frozen_predictions = {
        (row["question_id"], row["stage"]): float(row["stop_probability"])
        for row in read_jsonl(frozen_predictions_path)
        if row["stage"] in actionable
    }
    warmup = int(config["cost_measurement"]["warmup_questions"])

    def infer_one(item: dict, measured: bool) -> dict | None:
        encoded, _, _ = collate([item])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            logits, _ = model_outputs(model, encoded, bool(resolved["coverage_auxiliary"]))
            probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not measured:
            return None
        key = (item["question_id"], item["stage"])
        return {
            "question_id": item["question_id"],
            "stage": item["stage"],
            "phase3b_split": args.split,
            "controller_latency_ms": elapsed_ms,
            "replayed_stop_probability": probability,
            "frozen_stop_probability": frozen_predictions[key],
            "absolute_probability_difference": abs(probability - frozen_predictions[key]),
            "batch_size": 1,
            "cuda_synchronized": bool(torch.cuda.is_available()),
        }

    for item in prepared[:warmup]:
        infer_one(item, measured=False)
    measured_rows = []
    for index_value, item in enumerate(prepared, start=1):
        measured_rows.append(infer_one(item, measured=True))
        if index_value % 100 == 0:
            logger.info("benchmarked Controller %s/%s", index_value, len(prepared))
    maximum_difference = max(row["absolute_probability_difference"] for row in measured_rows)
    tolerance = float(config["cost_measurement"]["controller_probability_replay_tolerance"])
    if maximum_difference > tolerance:
        raise ValueError(
            f"Batch-size-one Controller replay drift {maximum_difference} exceeds {tolerance}"
        )

    write_jsonl_atomic(output_path, measured_rows)
    latencies = np.asarray([row["controller_latency_ms"] for row in measured_rows], dtype=float)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3B",
            "git_commit": git_commit(ROOT),
            "split": args.split,
            "actionable_stage_decisions": len(measured_rows),
            "warmup_decisions": warmup,
            "batch_size": 1,
            "cuda_synchronized": bool(torch.cuda.is_available()),
            "maximum_probability_replay_difference": maximum_difference,
            "probability_replay_tolerance": tolerance,
            "latency": {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.quantile(latencies, 0.95)),
            },
            "output": portable_path(output_path, ROOT),
            "output_sha256": file_sha256(output_path),
            "environment": collect_environment(ROOT),
        },
    )
    logger.info("saved Controller benchmark -> %s", output_path)


if __name__ == "__main__":
    main()

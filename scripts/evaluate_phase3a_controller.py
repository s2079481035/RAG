"""Evaluate a completed Phase 3A Controller with its frozen dev threshold."""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

from critic_metrics import binary_stop_metrics
from experiment_utils import portable_path, utc_now, write_json_atomic
from phase2_model import configure_cublas_workspace
from phase3a_metrics import coverage_regression_metrics
from phase3a_model import load_phase3a_model
from train_phase2_controller import attach_diagnostics, packing_summary, read_jsonl, write_jsonl_atomic
from train_phase3a_controller import attach_hard_partial, collate_batch, infer, prepare_records


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Phase 3A training run is not complete")
    resolved = config["resolved"]
    if args.split == "test" and not config["training"]["test_is_evaluation_only"]:
        raise ValueError("Configuration does not lock test to evaluation-only")
    configure_cublas_workspace(config["training"]["cublas_workspace_config"])

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    torch.use_deterministic_algorithms(True)
    load_kwargs = {"local_files_only": resolved["local_files_only"] and not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(resolved["backbone"], **load_kwargs)
    auxiliary = bool(resolved["coverage_auxiliary"])
    model, _ = load_phase3a_model(resolved["backbone"], load_kwargs, auxiliary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(manifest["checkpoint_path"], map_location=device, weights_only=True))
    model.to(device)

    variant = resolved["variant"]
    chunks = read_jsonl(ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    source_path = ROOT / "data" / "phase2" / "controller" / variant / f"{args.split}.jsonl"
    source = read_jsonl(source_path)
    namespace = argparse.Namespace(
        baseline=resolved["baseline"],
        representation=(
            "score_aware_packing"
            if resolved["representation"] == "not_applicable"
            else resolved["representation"]
        ),
    )
    prepared = prepare_records(
        source, args=namespace, config=config, tokenizer=tokenizer, chunk_by_id=chunk_by_id
    )
    loader = DataLoader(
        prepared,
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=partial(
            collate_batch,
            tokenizer=tokenizer,
            max_length=int(config["backbone"]["max_length"]),
        ),
    )
    labels, probabilities, truth, predicted = infer(model, loader, device, auxiliary)
    threshold = float(manifest["selected_dev_threshold"])
    metrics, decisions = binary_stop_metrics(labels, probabilities, threshold)
    metrics = attach_diagnostics(metrics, prepared, decisions)
    metrics = attach_hard_partial(metrics, prepared, decisions)
    metrics["packing"] = packing_summary(prepared)
    if auxiliary:
        metrics["coverage_regression"] = coverage_regression_metrics(truth, predicted)
    metrics["split"] = args.split

    output_dir = run_dir / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "original_predictions.jsonl"
    metrics_path = output_dir / "original_metrics.json"
    evaluation_manifest_path = output_dir / "evaluation_manifest_original.json"
    existing = [path for path in [predictions_path, metrics_path, evaluation_manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3A evaluation: {existing}")
    rows = []
    for index, (item, probability, decision) in enumerate(zip(prepared, probabilities, decisions)):
        rows.append(
            {
                "question_id": item["question_id"],
                "stage": item["stage"],
                "split": args.split,
                "actual_stop_label": item["label"],
                "actual_three_class_label": item["diagnostic_label"],
                "true_coverage": item["coverage_target"],
                "predicted_coverage": float(predicted[index]) if auxiliary else None,
                "stop_probability": float(probability),
                "predicted_stop_label": int(decision),
                "packing_audit": item["packing_audit"],
            }
        )
    write_jsonl_atomic(predictions_path, rows)
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(
        evaluation_manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "run_dir": portable_path(run_dir, ROOT),
            "source": portable_path(source_path, ROOT),
            "split": args.split,
            "condition": "original",
            "threshold_source": "selected on dev during training; unchanged for evaluation",
            "threshold": threshold,
            "coverage_lambda_source": "fixed before test; selected from dev candidates when enabled",
            "sampling_source": "fixed before test; train only",
            "metrics": metrics,
        },
    )
    logger.info("evaluated %s -> %s", args.split, output_dir)


if __name__ == "__main__":
    main()

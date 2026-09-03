"""Evaluate a trained hierarchical Phase 2 feasibility baseline."""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

from critic_metrics import binary_stop_metrics
from experiment_utils import portable_path, utc_now, write_json_atomic
from train_phase2_controller import attach_diagnostics, write_jsonl_atomic
from train_phase2_hierarchical import (
    build_model,
    collate_hierarchical,
    infer,
    prepare_items,
    read_jsonl,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Controller training run is not complete")
    resolved = config["resolved"]
    representation = resolved["representation"]
    if not representation.startswith("hierarchical_"):
        raise ValueError("This evaluator only accepts hierarchical runs")
    aggregation = representation.removeprefix("hierarchical_")
    variant = resolved["variant"]
    if args.split == "test":
        retrieval_config = json.loads(
            (ROOT / "configs" / "phase2" / "chunk_retrieval.json").read_text(encoding="utf-8")
        )
        if retrieval_config["retrieval_evaluation"]["selected_variant_after_dev"] != variant:
            raise ValueError("Test evaluation is locked to the dev-selected chunk variant")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    backbone = resolved["backbone"]
    load_kwargs = {"local_files_only": not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(backbone, **load_kwargs)
    chunks = read_jsonl(
        ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl"
    )
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    source_path = ROOT / "data" / "phase2" / "controller" / variant / f"{args.split}.jsonl"
    records = read_jsonl(source_path)
    prepared = prepare_items(
        records,
        chunk_by_id,
        resolved["evidence_mode"],
        tokenizer,
        int(config["training"]["hierarchical_pair_max_length"]),
    )

    training = config["training"]
    model = build_model(
        backbone,
        aggregation,
        bool(training["hierarchical_freeze_encoder"]),
        load_kwargs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(manifest["checkpoint_path"], map_location=device, weights_only=True)
    )
    model.to(device)
    loader = DataLoader(
        prepared,
        batch_size=int(training["hierarchical_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        collate_fn=partial(
            collate_hierarchical,
            tokenizer=tokenizer,
            max_length=int(training["hierarchical_pair_max_length"]),
        ),
    )
    labels, probabilities = infer(
        model,
        loader,
        device,
        int(training["hierarchical_encoder_chunk_batch_size"]),
    )
    threshold = float(manifest["selected_dev_threshold"])
    metrics, predictions = binary_stop_metrics(labels, probabilities, threshold)
    metrics = attach_diagnostics(metrics, prepared, predictions)
    metrics.update(
        {
            "split": args.split,
            "condition": "original",
            "hierarchical_encoder_frozen": bool(training["hierarchical_freeze_encoder"]),
            "mean_raw_evidence_tokens": sum(
                item["raw_evidence_tokens"] for item in prepared
            )
            / len(prepared),
            "mean_visible_evidence_chunk_ratio": sum(
                item["hierarchical_packing_audit"]["fully_visible_evidence_chunk_ratio"]
                for item in prepared
            )
            / len(prepared),
        }
    )

    output_dir = run_dir / "evaluation" / args.split
    metrics_path = output_dir / "original_metrics.json"
    predictions_path = output_dir / "original_predictions.jsonl"
    evaluation_manifest_path = output_dir / "evaluation_manifest_original.json"
    existing = [path for path in [metrics_path, predictions_path, evaluation_manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite evaluation outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(metrics_path, metrics)
    prediction_records = []
    for item, probability, prediction in zip(prepared, probabilities, predictions):
        prediction_records.append(
            {
                "question_id": item["question_id"],
                "stage": item["stage"],
                "split": args.split,
                "condition": "original",
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_supporting_facts": item["gold_supporting_facts"],
                "gold_supporting_fact_count": item["gold_supporting_fact_count"],
                "input_evidence_chunk_ids": [chunk["chunk_id"] for chunk in item["chunks"]],
                "actual_stop_label": item["label"],
                "actual_three_class_label": item["diagnostic_label"],
                "stop_probability": float(probability),
                "predicted_stop_label": int(prediction),
                "raw_evidence_tokens": item["raw_evidence_tokens"],
                "evidence_chunk_count": len(item["chunks"]),
                "visible_supporting_fact_ratio_evaluation_only": item[
                    "hierarchical_packing_audit"
                ]["visible_supporting_fact_ratio_evaluation_only"],
                "hierarchical_packing_audit": item["hierarchical_packing_audit"],
                "packing_audit": None,
            }
        )
    write_jsonl_atomic(predictions_path, prediction_records)
    write_json_atomic(
        evaluation_manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "run_dir": portable_path(run_dir, ROOT),
            "source": portable_path(source_path, ROOT),
            "split": args.split,
            "threshold_source": "selected on dev during training; unchanged for this evaluation",
            "threshold": threshold,
            "metrics": metrics,
        },
    )
    logger.info("evaluated %s -> %s", args.split, metrics_path)


if __name__ == "__main__":
    main()

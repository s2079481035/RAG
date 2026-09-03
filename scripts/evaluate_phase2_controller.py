"""Evaluate a trained Phase 2 Controller without fitting anything on test."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

from critic_metrics import binary_stop_metrics
from evidence_utils import unique_supporting_facts
from experiment_utils import portable_path, utc_now, write_json_atomic
from phase2_controller_inputs import EVIDENCE_BASELINES, evidence_key, prepare_nonhierarchical_input
from train_phase2_controller import (
    attach_diagnostics,
    collate_batch,
    infer,
    packing_summary,
    read_jsonl,
    write_jsonl_atomic,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = [
    "original",
    "evidence_order_shuffle",
    "cross_question_evidence_swap",
    "stage_metadata_removal",
    "title_only_evidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--conditions",
        default="original",
        help="Comma-separated diagnostic conditions; counterfactuals are dev-only",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_shuffle(items: list[dict], key: str, seed: int) -> list[dict]:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    output = list(items)
    rng.shuffle(output)
    return output


def swapped_items_by_sample(records: list[dict], evidence_mode: str) -> dict[tuple[str, str], list[dict]]:
    grouped = defaultdict(list)
    view_key = evidence_key(evidence_mode)
    for record in records:
        grouped[record["stage"]].append(record)
    swapped = {}
    for stage, stage_records in grouped.items():
        if len(stage_records) < 2:
            raise ValueError(f"Cannot swap evidence in singleton stage group: {stage}")
        donors = stage_records[1:] + stage_records[:1]
        for record, donor in zip(stage_records, donors):
            if record["question_id"] == donor["question_id"]:
                raise AssertionError("Cross-question swap selected the same question")
            swapped[(record["question_id"], stage)] = donor[view_key]["items"]
    return swapped


def visible_gold_ratio(item: dict) -> float | None:
    audit = item["packing_audit"]
    if audit is None:
        return None
    gold = set(unique_supporting_facts(item["gold_supporting_facts"]))
    visible = {
        (fact["title"], fact["sentence_id"])
        for fact in audit["visible_sentence_facts"]
    }
    return len(gold & visible) / len(gold)


def main() -> None:
    args = parse_args()
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    if not conditions or set(conditions) - set(CONDITIONS):
        raise ValueError(f"Unknown evaluation conditions: {conditions}")
    if args.split == "test" and conditions != ["original"]:
        raise ValueError("Counterfactual diagnostics are dev-only and cannot be run on test")

    run_dir = args.run_dir.resolve()
    resolved_path = run_dir / "resolved_config.json"
    manifest_path = run_dir / "run_manifest.json"
    if not resolved_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run directory lacks resolved_config.json or run_manifest.json")
    config = json.loads(resolved_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Controller training run is not complete")
    resolved = config["resolved"]
    representation = resolved["representation"]
    if representation.startswith("hierarchical_"):
        raise ValueError("Use evaluate_phase2_hierarchical.py for hierarchical runs")
    baseline = resolved["baseline"]
    evidence_mode = resolved["evidence_mode"]
    if any(condition != "original" for condition in conditions) and baseline not in EVIDENCE_BASELINES:
        raise ValueError("Evidence counterfactuals require an evidence-input baseline")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    backbone = resolved["backbone"]
    load_kwargs = {"local_files_only": not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(backbone, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=2, ignore_mismatched_sizes=True, **load_kwargs
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(manifest["checkpoint_path"], map_location=device, weights_only=True)
    )
    model.to(device)

    variant = resolved["variant"]
    chunks = read_jsonl(ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl")
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    source_path = ROOT / "data" / "phase2" / "controller" / variant / f"{args.split}.jsonl"
    records = read_jsonl(source_path)
    if args.split == "test":
        retrieval_config = json.loads(
            (ROOT / "configs" / "phase2" / "chunk_retrieval.json").read_text(encoding="utf-8")
        )
        selected = retrieval_config["retrieval_evaluation"]["selected_variant_after_dev"]
        if selected != variant:
            raise ValueError("Test evaluation is locked to the dev-selected chunk variant")

    output_dir = run_dir / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        condition: output_dir / f"{condition}_predictions.jsonl" for condition in conditions
    }
    metrics_paths = {
        condition: output_dir / f"{condition}_metrics.json" for condition in conditions
    }
    targets = [*output_paths.values(), *metrics_paths.values()]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite evaluation outputs: {existing}")

    representation_config = config["representations"].get(representation, {})
    minimum_chunk_tokens = int(representation_config.get("minimum_chunk_tokens", 12))
    swapped = swapped_items_by_sample(records, evidence_mode)
    threshold = float(manifest["selected_dev_threshold"])
    all_metrics = {}
    for condition in conditions:
        prepared = []
        view_key = evidence_key(evidence_mode)
        for record in records:
            override = None
            if condition == "evidence_order_shuffle":
                override = stable_shuffle(
                    record[view_key]["items"],
                    f"{record['question_id']}:{record['stage']}",
                    int(config["seed"]),
                )
            elif condition == "cross_question_evidence_swap":
                override = swapped[(record["question_id"], record["stage"])]
            prepared.append(
                prepare_nonhierarchical_input(
                    record,
                    baseline=baseline,
                    representation=representation if representation != "not_applicable" else "concat_truncate",
                    evidence_mode=evidence_mode,
                    tokenizer=tokenizer,
                    chunk_by_id=chunk_by_id,
                    max_length=int(config["backbone"]["max_length"]),
                    minimum_chunk_tokens=minimum_chunk_tokens,
                    include_stage_in_query_evidence=(
                        False
                        if condition == "stage_metadata_removal"
                        else config["include_stage_metadata_in_query_evidence"]
                    ),
                    evidence_items_override=override,
                    title_only=condition == "title_only_evidence",
                )
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
        labels, probabilities = infer(model, loader, device)
        metrics, predictions = binary_stop_metrics(labels, probabilities, threshold)
        metrics = attach_diagnostics(metrics, prepared, predictions)
        metrics["packing"] = packing_summary(prepared)
        metrics["condition"] = condition
        metrics["split"] = args.split
        all_metrics[condition] = metrics
        write_json_atomic(metrics_paths[condition], metrics)

        prediction_records = []
        for item, probability, prediction in zip(prepared, probabilities, predictions):
            prediction_records.append(
                {
                    "question_id": item["question_id"],
                    "stage": item["stage"],
                    "split": args.split,
                    "condition": condition,
                    "question": item["question"],
                    "gold_answer": item["answer"],
                    "gold_supporting_facts": item["gold_supporting_facts"],
                    "gold_supporting_fact_count": item["gold_supporting_fact_count"],
                    "input_evidence_chunk_ids": item.get("input_evidence_chunk_ids", []),
                    "input_evidence_titles": item.get("input_evidence_titles", []),
                    "actual_stop_label": item["label"],
                    "actual_three_class_label": item["diagnostic_label"],
                    "stop_probability": float(probability),
                    "predicted_stop_label": int(prediction),
                    "visible_supporting_fact_ratio_evaluation_only": visible_gold_ratio(item),
                    "packing_audit": item["packing_audit"],
                }
            )
        write_jsonl_atomic(output_paths[condition], prediction_records)
        logger.info("evaluated %s %s", args.split, condition)

    write_json_atomic(
        output_dir / f"evaluation_manifest_{'_'.join(conditions)}.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "run_dir": portable_path(run_dir, ROOT),
            "source": portable_path(source_path, ROOT),
            "split": args.split,
            "conditions": conditions,
            "threshold_source": "selected on dev during training; unchanged for this evaluation",
            "threshold": threshold,
            "metrics": all_metrics,
        },
    )


if __name__ == "__main__":
    main()

"""Train matched non-hierarchical Phase 2 Controller baselines A-E."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

from critic_metrics import binary_stop_metrics
from evidence_utils import assert_disjoint_question_ids, unique_supporting_facts, validate_tuning_splits
from experiment_utils import collect_environment, seed_worker, set_global_seed, utc_now, write_json_atomic
from phase2_controller_inputs import (
    EVIDENCE_BASELINES,
    NON_EVIDENCE_BASELINES,
    prepare_nonhierarchical_input,
)
from phase2_model import configure_cublas_workspace, load_binary_sequence_classifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase2" / "controller.json"
PACKING_REPRESENTATIONS = {"concat_truncate", "uniform_packing", "score_aware_packing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--baseline", required=True, choices=sorted(NON_EVIDENCE_BASELINES | EVIDENCE_BASELINES)
    )
    parser.add_argument("--representation", default="concat_truncate")
    parser.add_argument("--evidence-mode", choices=["raw", "cumulative"], default="cumulative")
    parser.add_argument("--model", help="Local backbone path or Hugging Face model name")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def collate_batch(batch, *, tokenizer, max_length: int):
    import torch

    texts_a = [item["text_a"] for item in batch]
    texts_b = [item["text_b"] for item in batch]
    if all(text is None for text in texts_b):
        encoded = tokenizer(
            texts_a,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
    elif all(text is not None for text in texts_b):
        encoded = tokenizer(
            texts_a,
            texts_b,
            truncation="only_second",
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        raise ValueError("A batch cannot mix pair and single-sequence inputs")
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return encoded, labels


def infer(model, loader, device):
    import torch

    model.eval()
    labels = []
    probabilities = []
    with torch.no_grad():
        for encoded, batch_labels in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.numpy())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def select_threshold(labels, probabilities, grid: dict) -> tuple[float, dict]:
    start = float(grid["start"])
    stop = float(grid["stop"])
    step = float(grid["step"])
    candidates = np.arange(start, stop + step / 2.0, step)
    scored = []
    for threshold in candidates:
        metrics, _ = binary_stop_metrics(labels, probabilities, float(threshold))
        scored.append((metrics["stop_f1"], metrics["stop_precision"], -float(threshold), metrics))
    best = max(scored, key=lambda item: item[:3])
    return -best[2], best[3]


def packing_summary(items: list[dict]) -> dict:
    audited = [item for item in items if item["packing_audit"] is not None]
    if not audited:
        return {"applicable": False, "samples": len(items)}
    truncated = 0
    raw_lengths = []
    visible_chunk_ratios = []
    fully_visible_chunk_ratios = []
    visible_gold_ratios = []
    for item in audited:
        audit = item["packing_audit"]
        truncated += int(audit["truncated"])
        raw_lengths.append(audit["raw_evidence_tokens"])
        visible_chunk_ratios.append(audit["visible_evidence_chunk_ratio"])
        fully_visible_chunk_ratios.append(audit["fully_visible_evidence_chunk_ratio"])
        gold = set(unique_supporting_facts(item["gold_supporting_facts"]))
        visible = {
            (fact["title"], fact["sentence_id"])
            for fact in audit["visible_sentence_facts"]
        }
        visible_gold_ratios.append(len(gold & visible) / len(gold))
    values = np.asarray(raw_lengths)
    return {
        "applicable": True,
        "samples": len(audited),
        "truncated_samples": truncated,
        "truncation_ratio": truncated / len(audited),
        "raw_evidence_tokens": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": int(values.max()),
        },
        "mean_visible_evidence_chunk_ratio": float(np.mean(visible_chunk_ratios)),
        "mean_fully_visible_evidence_chunk_ratio": float(
            np.mean(fully_visible_chunk_ratios)
        ),
        "mean_visible_supporting_fact_ratio_evaluation_only": float(
            np.mean(visible_gold_ratios)
        ),
    }


def attach_diagnostics(metrics: dict, items: list[dict], predictions: np.ndarray) -> dict:
    false_stop_states = Counter()
    unnecessary_states = Counter()
    for item, prediction in zip(items, predictions):
        if item["label"] == 0 and prediction == 1:
            false_stop_states[item["diagnostic_label"]] += 1
        if item["label"] == 1 and prediction == 0:
            unnecessary_states[item["diagnostic_label"]] += 1
    return {
        **metrics,
        "false_stop_by_three_class_label": dict(false_stop_states),
        "unnecessary_escalation_by_three_class_label": dict(unnecessary_states),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.baseline in EVIDENCE_BASELINES and args.representation not in PACKING_REPRESENTATIONS:
        raise ValueError("Use train_phase2_hierarchical.py for hierarchical representations")
    if args.baseline in NON_EVIDENCE_BASELINES:
        resolved_representation = "not_applicable"
    else:
        resolved_representation = args.representation
    validate_tuning_splits(
        [
            config["training"]["selection_split"],
            config["training"]["threshold_selection_split"],
        ]
    )
    cublas_workspace_config = configure_cublas_workspace(
        config["training"]["cublas_workspace_config"]
    )

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    seed = int(config["seed"])
    set_global_seed(seed)
    torch.use_deterministic_algorithms(True)
    backbone = args.model or config["backbone"]["name"]
    load_kwargs = {"local_files_only": not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(backbone, **load_kwargs)
    model, classifier_initialization = load_binary_sequence_classifier(backbone, load_kwargs)
    max_length = int(config["backbone"]["max_length"])
    chunk_path = ROOT / "data" / "phase2" / "chunks" / f"{args.variant}.jsonl"
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in read_jsonl(chunk_path)}
    data_dir = ROOT / "data" / "phase2" / "controller" / args.variant
    source = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ["train", "dev"]}
    assert_disjoint_question_ids(
        {split: {item["question_id"] for item in records} for split, records in source.items()}
    )

    representation_config = config["representations"].get(args.representation, {})
    minimum_chunk_tokens = int(representation_config.get("minimum_chunk_tokens", 12))
    prepared = {}
    for split, records in source.items():
        prepared[split] = [
            prepare_nonhierarchical_input(
                record,
                baseline=args.baseline,
                representation=args.representation,
                evidence_mode=args.evidence_mode,
                tokenizer=tokenizer,
                chunk_by_id=chunk_by_id,
                max_length=max_length,
                minimum_chunk_tokens=minimum_chunk_tokens,
                include_stage_in_query_evidence=config[
                    "include_stage_metadata_in_query_evidence"
                ],
            )
            for record in records
        ]
        logger.info("prepared %s %s samples", len(prepared[split]), split)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{args.baseline}__{resolved_representation}__{args.evidence_mode}__{timestamp}"
    run_dir = args.run_dir or (
        ROOT / "experiments" / "phase2" / "controller" / args.variant / run_name
    )
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    training = config["training"]
    packing_stats = {
        split: packing_summary(items) for split, items in prepared.items()
    }
    write_json_atomic(run_dir / "packing_stats.json", packing_stats)
    resolved = {
        **config,
        "resolved": {
            "variant": args.variant,
            "baseline": args.baseline,
            "representation": resolved_representation,
            "evidence_mode": args.evidence_mode,
            "backbone": backbone,
            "local_files_only": not args.allow_download,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "classifier_initialization": classifier_initialization,
        },
    }
    write_json_atomic(run_dir / "resolved_config.json", resolved)
    manifest = {
        "status": "running",
        "start_time_utc": utc_now(),
        "end_time_utc": None,
        "phase": 2,
        "phase1_frozen_commit": config["phase1_frozen_commit"],
        "variant": args.variant,
        "baseline": args.baseline,
        "representation": resolved_representation,
        "evidence_mode": args.evidence_mode,
        "seed": seed,
        "cublas_workspace_config": cublas_workspace_config,
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)

    generator = torch.Generator()
    generator.manual_seed(seed)
    collate = partial(collate_batch, tokenizer=tokenizer, max_length=max_length)
    loaders = {
        "train": DataLoader(
            prepared["train"],
            batch_size=int(training["batch_size"]),
            shuffle=True,
            num_workers=int(training["num_workers"]),
            worker_init_fn=seed_worker,
            generator=generator,
            collate_fn=collate,
        ),
        "dev": DataLoader(
            prepared["dev"],
            batch_size=int(training["eval_batch_size"]),
            shuffle=False,
            num_workers=int(training["num_workers"]),
            worker_init_fn=seed_worker,
            collate_fn=collate,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    total_steps = len(loaders["train"]) * epochs
    warmup_steps = math.floor(total_steps * float(training["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    fixed_threshold = float(training["threshold"])
    history = []
    best_metric = -1.0
    best_epoch = None
    checkpoint_path = run_dir / "best.pt"
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (encoded, labels) in enumerate(loaders["train"], start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**encoded, labels=labels)
            output.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += output.loss.item()
            if step % 200 == 0:
                logger.info(
                    "%s epoch=%s step=%s/%s loss=%.4f",
                    args.baseline,
                    epoch + 1,
                    step,
                    len(loaders["train"]),
                    total_loss / step,
                )
        labels, probabilities = infer(model, loaders["dev"], device)
        metrics, predictions = binary_stop_metrics(labels, probabilities, fixed_threshold)
        metrics = attach_diagnostics(metrics, prepared["dev"], predictions)
        history.append(
            {"epoch": epoch + 1, "train_loss": total_loss / len(loaders["train"]), "dev": metrics}
        )
        write_json_atomic(run_dir / "history.json", history)
        selection_value = float(metrics[training["selection_metric"]])
        logger.info("%s epoch=%s dev=%s", args.baseline, epoch + 1, metrics)
        if selection_value > best_metric:
            best_metric = selection_value
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    labels, probabilities = infer(model, loaders["dev"], device)
    selected_threshold, selected_metrics = select_threshold(
        labels, probabilities, training["threshold_grid"]
    )
    selected_metrics, predictions = binary_stop_metrics(
        labels, probabilities, selected_threshold
    )
    selected_metrics = attach_diagnostics(selected_metrics, prepared["dev"], predictions)
    write_json_atomic(run_dir / "dev_metrics.json", selected_metrics)
    prediction_records = []
    for item, probability, prediction in zip(prepared["dev"], probabilities, predictions):
        prediction_records.append(
            {
                "question_id": item["question_id"],
                "stage": item["stage"],
                "actual_stop_label": item["label"],
                "actual_three_class_label": item["diagnostic_label"],
                "stop_probability": float(probability),
                "predicted_stop_label": int(prediction),
                "packing_audit": item["packing_audit"],
            }
        )
    write_jsonl_atomic(run_dir / "dev_predictions.jsonl", prediction_records)
    manifest.update(
        {
            "status": "complete",
            "end_time_utc": utc_now(),
            "best_epoch": best_epoch,
            "best_dev_metric_at_fixed_threshold": best_metric,
            "selection_metric": training["selection_metric"],
            "fixed_training_threshold": fixed_threshold,
            "selected_dev_threshold": selected_threshold,
            "selected_dev_metrics": selected_metrics,
            "checkpoint_path": str(checkpoint_path),
        }
    )
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    logger.info("complete: %s", run_dir)


if __name__ == "__main__":
    main()

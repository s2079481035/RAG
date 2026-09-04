"""Train Phase 3A Controllers with optional coverage supervision and sampling."""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

from critic_metrics import binary_stop_metrics
from evidence_utils import assert_disjoint_question_ids, validate_tuning_splits
from experiment_utils import collect_environment, seed_worker, set_global_seed, utc_now, write_json_atomic
from phase2_controller_inputs import evidence_key, prepare_nonhierarchical_input
from phase2_model import configure_cublas_workspace
from phase3a_metrics import coverage_bucket, coverage_regression_metrics, sampling_weights
from phase3a_model import load_phase3a_model
from train_phase2_controller import attach_diagnostics, packing_summary, read_jsonl, select_threshold, write_jsonl_atomic


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3a" / "controller.json"
BASELINES = {"query_stage", "query_evidence"}
REPRESENTATIONS = {"concat_truncate", "score_aware_packing"}
SAMPLING = {"natural", "balanced_stop_continue", "hard_partial_aware"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    parser.add_argument("--representation", default="score_aware_packing", choices=sorted(REPRESENTATIONS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--coverage-lambda", type=float, default=0.0)
    parser.add_argument("--sampling", choices=sorted(SAMPLING), default="natural")
    parser.add_argument("--model", help="Local backbone path or Hugging Face model name")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def validate_experiment(args: argparse.Namespace, config: dict) -> None:
    if args.seed not in {int(value) for value in config["seeds"]}:
        raise ValueError(f"Seed {args.seed} is outside the preregistered seed set")
    if args.coverage_lambda < 0:
        raise ValueError("Coverage lambda cannot be negative")
    if args.coverage_lambda > 0:
        allowed = {float(value) for value in config["coverage_auxiliary"]["candidate_lambdas"]}
        if args.coverage_lambda not in allowed:
            raise ValueError("Coverage lambda is outside the preregistered dev candidates")
        if args.baseline != "query_evidence" or args.representation != "score_aware_packing":
            raise ValueError("Coverage auxiliary supervision is scoped to Query+Evidence score-aware")
    if args.sampling != "natural" and (
        args.baseline != "query_evidence" or args.representation != "score_aware_packing"
    ):
        raise ValueError("Sampling baselines are scoped to Query+Evidence score-aware")
    if args.baseline == "query_stage" and args.representation != "score_aware_packing":
        raise ValueError("Use the canonical placeholder representation for Query+Stage")


def prepare_records(records: list[dict], *, args, config, tokenizer, chunk_by_id) -> list[dict]:
    prepared = []
    for record in records:
        item = prepare_nonhierarchical_input(
            record,
            baseline=args.baseline,
            representation=args.representation,
            evidence_mode=config["input"]["evidence_mode"],
            tokenizer=tokenizer,
            chunk_by_id=chunk_by_id,
            max_length=int(config["backbone"]["max_length"]),
            minimum_chunk_tokens=int(config["input"]["minimum_chunk_tokens"]),
            include_stage_in_query_evidence=config["input"][
                "include_stage_metadata_in_query_evidence"
            ],
        )
        view = record[evidence_key(config["input"]["evidence_mode"])]
        item["coverage_target"] = float(view["supporting_fact_recall"])
        prepared.append(item)
    return prepared


def collate_batch(batch, *, tokenizer, max_length: int):
    import torch

    texts_a = [item["text_a"] for item in batch]
    texts_b = [item["text_b"] for item in batch]
    if all(text is None for text in texts_b):
        encoded = tokenizer(
            texts_a, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
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
    coverage = torch.tensor([item["coverage_target"] for item in batch], dtype=torch.float)
    return encoded, labels, coverage


def model_outputs(model, encoded: dict, coverage_auxiliary: bool):
    if coverage_auxiliary:
        return model(**encoded)
    return model(**encoded).logits, None


def infer(model, loader, device, coverage_auxiliary: bool):
    import torch

    model.eval()
    labels = []
    probabilities = []
    true_coverage = []
    predicted_coverage = []
    with torch.no_grad():
        for encoded, batch_labels, batch_coverage in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits, coverage = model_outputs(model, encoded, coverage_auxiliary)
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.numpy())
            true_coverage.extend(batch_coverage.numpy())
            if coverage is not None:
                predicted_coverage.extend(coverage.cpu().numpy())
    return (
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
        np.asarray(true_coverage, dtype=float),
        np.asarray(predicted_coverage, dtype=float) if coverage_auxiliary else None,
    )


def attach_hard_partial(metrics: dict, items: list[dict], predictions) -> dict:
    hard = [
        index for index, item in enumerate(items)
        if int(item["label"]) == 0 and coverage_bucket(item["coverage_target"]) == "hard_partial"
    ]
    false_stops = sum(int(predictions[index]) == 1 for index in hard)
    return {
        **metrics,
        "hard_partial_continue_count": len(hard),
        "hard_partial_false_stop_count": false_stops,
        "hard_partial_false_stop_rate": false_stops / len(hard) if hard else None,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_experiment(args, config)
    validate_tuning_splits(
        [config["training"]["selection_split"], config["training"]["threshold_selection_split"]]
    )
    cublas = configure_cublas_workspace(config["training"]["cublas_workspace_config"])

    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    set_global_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    backbone = args.model or config["backbone"]["name"]
    load_kwargs = {"local_files_only": not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(backbone, **load_kwargs)
    auxiliary = args.coverage_lambda > 0
    model, classifier_initialization = load_phase3a_model(backbone, load_kwargs, auxiliary)

    variant = config["variant"]
    chunk_path = ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl"
    chunk_by_id = {row["chunk_id"]: row for row in read_jsonl(chunk_path)}
    data_dir = ROOT / "data" / "phase2" / "controller" / variant
    source = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ["train", "dev"]}
    assert_disjoint_question_ids(
        {split: {item["question_id"] for item in records} for split, records in source.items()}
    )
    prepared = {
        split: prepare_records(
            records,
            args=args,
            config=config,
            tokenizer=tokenizer,
            chunk_by_id=chunk_by_id,
        )
        for split, records in source.items()
    }
    for split, records in prepared.items():
        logger.info("prepared %s %s samples", len(records), split)

    resolved_representation = "not_applicable" if args.baseline == "query_stage" else args.representation
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"{args.baseline}__{resolved_representation}__seed{args.seed}__"
        f"aux{args.coverage_lambda:g}__{args.sampling}__{timestamp}"
    )
    run_dir = (args.run_dir or ROOT / "experiments" / "phase3a" / run_name).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    training = config["training"]
    sampling_config = config["sampling"][args.sampling]
    weights, sampling_audit = sampling_weights(
        prepared["train"], args.sampling, sampling_config.get("target_group_mass", {})
    )
    sampling_audit["configured_replacement"] = sampling_config["replacement"]
    write_json_atomic(run_dir / "sampling_audit.json", sampling_audit)
    write_json_atomic(
        run_dir / "packing_stats.json",
        {split: packing_summary(items) for split, items in prepared.items()},
    )
    resolved_config = {
        **config,
        "resolved": {
            "variant": variant,
            "baseline": args.baseline,
            "representation": resolved_representation,
            "evidence_mode": config["input"]["evidence_mode"],
            "backbone": backbone,
            "local_files_only": not args.allow_download,
            "classifier_initialization": classifier_initialization,
            "seed": args.seed,
            "coverage_auxiliary": auxiliary,
            "coverage_lambda": args.coverage_lambda,
            "sampling": args.sampling,
        },
    }
    write_json_atomic(run_dir / "resolved_config.json", resolved_config)
    manifest = {
        "status": "running",
        "start_time_utc": utc_now(),
        "end_time_utc": None,
        "phase": "3A",
        "phase2_frozen_branch": config["phase2_frozen_branch"],
        "variant": variant,
        "baseline": args.baseline,
        "representation": resolved_representation,
        "evidence_mode": config["input"]["evidence_mode"],
        "seed": args.seed,
        "coverage_auxiliary": auxiliary,
        "coverage_lambda": args.coverage_lambda,
        "sampling": args.sampling,
        "cublas_workspace_config": cublas,
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)

    collate = partial(collate_batch, tokenizer=tokenizer, max_length=int(config["backbone"]["max_length"]))
    train_generator = torch.Generator().manual_seed(args.seed)
    worker_generator = torch.Generator().manual_seed(args.seed + 1)
    sampler = None
    shuffle = args.sampling == "natural"
    if weights:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(prepared["train"]),
            replacement=True,
            generator=train_generator,
        )
    loaders = {
        "train": DataLoader(
            prepared["train"],
            batch_size=int(training["batch_size"]),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(training["num_workers"]),
            worker_init_fn=seed_worker,
            generator=worker_generator if sampler else train_generator,
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
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    epochs = int(training["epochs"])
    total_steps = len(loaders["train"]) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.floor(total_steps * float(training["warmup_ratio"])), total_steps
    )
    fixed_threshold = float(training["fixed_threshold"])
    checkpoint_path = run_dir / "best.pt"
    best_metric = -1.0
    best_epoch = None
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = total_stop_loss = total_coverage_loss = 0.0
        for step, (encoded, labels, coverage_targets) in enumerate(loaders["train"], start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = labels.to(device)
            coverage_targets = coverage_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_coverage = model_outputs(model, encoded, auxiliary)
            stop_loss = functional.cross_entropy(logits, labels)
            coverage_loss = (
                functional.smooth_l1_loss(predicted_coverage, coverage_targets)
                if auxiliary else torch.zeros((), device=device)
            )
            loss = stop_loss + args.coverage_lambda * coverage_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            total_stop_loss += float(stop_loss.item())
            total_coverage_loss += float(coverage_loss.item())
            if step % 200 == 0:
                logger.info("%s seed=%s epoch=%s step=%s/%s loss=%.4f", args.baseline, args.seed, epoch + 1, step, len(loaders["train"]), total_loss / step)
        labels, probabilities, truth, predicted = infer(model, loaders["dev"], device, auxiliary)
        metrics, decisions = binary_stop_metrics(labels, probabilities, fixed_threshold)
        metrics = attach_diagnostics(metrics, prepared["dev"], decisions)
        metrics = attach_hard_partial(metrics, prepared["dev"], decisions)
        if auxiliary:
            metrics["coverage_regression"] = coverage_regression_metrics(truth, predicted)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(loaders["train"]),
                "train_stop_loss": total_stop_loss / len(loaders["train"]),
                "train_coverage_loss": total_coverage_loss / len(loaders["train"]),
                "dev": metrics,
            }
        )
        write_json_atomic(run_dir / "history.json", history)
        selection_value = float(metrics[training["selection_metric"]])
        if selection_value > best_metric:
            best_metric = selection_value
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
        logger.info("%s seed=%s epoch=%s dev=%s", args.baseline, args.seed, epoch + 1, metrics)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    labels, probabilities, truth, predicted = infer(model, loaders["dev"], device, auxiliary)
    selected_threshold, _ = select_threshold(labels, probabilities, training["threshold_grid"])
    selected_metrics, decisions = binary_stop_metrics(labels, probabilities, selected_threshold)
    selected_metrics = attach_diagnostics(selected_metrics, prepared["dev"], decisions)
    selected_metrics = attach_hard_partial(selected_metrics, prepared["dev"], decisions)
    if auxiliary:
        selected_metrics["coverage_regression"] = coverage_regression_metrics(truth, predicted)
    write_json_atomic(run_dir / "dev_metrics.json", selected_metrics)
    prediction_rows = []
    for index, (item, probability, decision) in enumerate(zip(prepared["dev"], probabilities, decisions)):
        prediction_rows.append(
            {
                "question_id": item["question_id"],
                "stage": item["stage"],
                "split": "dev",
                "actual_stop_label": item["label"],
                "actual_three_class_label": item["diagnostic_label"],
                "true_coverage": item["coverage_target"],
                "predicted_coverage": float(predicted[index]) if auxiliary else None,
                "stop_probability": float(probability),
                "predicted_stop_label": int(decision),
                "packing_audit": item["packing_audit"],
            }
        )
    write_jsonl_atomic(run_dir / "dev_predictions.jsonl", prediction_rows)
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

"""Train the frozen question-only Adaptive-RAG-style Router on Train-derived data."""

from __future__ import annotations

import argparse
import json
import logging
import math
from functools import partial
from pathlib import Path

import numpy as np

from experiment_utils import (
    collect_environment,
    seed_worker,
    set_global_seed,
    utc_now,
    write_json_atomic,
)
from phase2_model import configure_cublas_workspace
from phase4_adaptive_router import (
    ROUTE_NAMES,
    file_sha256,
    multiclass_metrics,
    read_jsonl,
    write_jsonl_atomic,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "adaptive_rag_router.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def replace_output_head(model, torch_module) -> dict:
    classifier = getattr(model, "classifier", None)
    output = getattr(classifier, "out_proj", None)
    if output is None or not isinstance(output, torch_module.nn.Linear):
        raise TypeError("Expected classifier.out_proj Linear on Router backbone")
    checkpoint_labels = int(output.out_features)
    replacement = torch_module.nn.Linear(
        int(output.in_features), len(ROUTE_NAMES), bias=output.bias is not None
    )
    model._init_weights(replacement)
    classifier.out_proj = replacement
    model.num_labels = len(ROUTE_NAMES)
    model.config.num_labels = len(ROUTE_NAMES)
    model.config.id2label = dict(enumerate(ROUTE_NAMES))
    model.config.label2id = {name: index for index, name in enumerate(ROUTE_NAMES)}
    return {
        "strategy": "load_checkpoint_head_then_reinitialize_three_class_out_proj",
        "checkpoint_num_labels": checkpoint_labels,
        "resolved_num_labels": len(ROUTE_NAMES),
    }


def collate_batch(batch, *, tokenizer, max_length: int):
    import torch

    encoded = tokenizer(
        [row["question"] for row in batch],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor([int(row["label"]) for row in batch], dtype=torch.long)
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
            labels.extend(batch_labels.numpy())
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    if args.seed not in {int(value) for value in training["seeds"]}:
        raise ValueError("Seed is outside the frozen Router seed set")
    if config["inference_features"] != ["question"]:
        raise ValueError("Router inference features are not question-only")
    if config["data"]["selection_split"] != "dev_calibration":
        raise ValueError("Router model selection must use dev_calibration")
    configure_cublas_workspace(training["cublas_workspace_config"])

    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_global_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    data_dir = ROOT / config["data"]["router_dir"]
    train_path = data_dir / f"{config['data']['train_split']}.jsonl"
    dev_path = data_dir / f"{config['data']['selection_split']}.jsonl"
    train_rows = list(read_jsonl(train_path))
    dev_rows = list(read_jsonl(dev_path))
    train_ids = {row["question_id"] for row in train_rows}
    dev_ids = {row["question_id"] for row in dev_rows}
    if train_ids & dev_ids:
        raise ValueError("Router train and dev_calibration IDs overlap")
    backbone = args.model or config["model"]["backbone"]
    load_kwargs = {"local_files_only": not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(backbone, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, **load_kwargs)
    initialization = replace_output_head(model, torch)
    run_dir = (
        args.run_dir
        or ROOT / config["output"]["experiment_dir"] / f"seed{args.seed}"
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Router run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        **config,
        "resolved": {
            "seed": args.seed,
            "backbone": backbone,
            "local_files_only": not args.allow_download,
            "classifier_initialization": initialization,
            "model_input_fields": ["question"],
        },
    }
    write_json_atomic(run_dir / "resolved_config.json", resolved)
    manifest = {
        "status": "running",
        "start_time_utc": utc_now(),
        "end_time_utc": None,
        "phase": 4,
        "formal_name": config["formal_name"],
        "seed": args.seed,
        "training_split": "train_core",
        "selection_split": "dev_calibration",
        "heldout_consulted": False,
        "model_input_fields": ["question"],
        "router_config_sha256": file_sha256(args.config),
        "stage_mapping_sha256": file_sha256(ROOT / config["stage_mapping"]),
        "label_construction": config["label_construction"],
        "source_sha256": {
            "train_core": file_sha256(train_path),
            "dev_calibration": file_sha256(dev_path),
        },
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)

    collate = partial(
        collate_batch,
        tokenizer=tokenizer,
        max_length=int(config["model"]["max_length"]),
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            train_rows,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            num_workers=int(training["num_workers"]),
            worker_init_fn=seed_worker,
            generator=train_generator,
            collate_fn=collate,
        ),
        "dev": DataLoader(
            dev_rows,
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
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        math.floor(total_steps * float(training["warmup_ratio"])),
        total_steps,
    )
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
            logits = model(**encoded).logits
            loss = functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            if step % 200 == 0:
                logger.info(
                    "router seed=%s epoch=%s step=%s/%s loss=%.4f",
                    args.seed,
                    epoch + 1,
                    step,
                    len(loaders["train"]),
                    total_loss / step,
                )
        labels, probabilities = infer(model, loaders["dev"], device)
        metrics, _ = multiclass_metrics(labels, probabilities)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(loaders["train"]),
                "dev_calibration": metrics,
            }
        )
        write_json_atomic(run_dir / "history.json", history)
        selection_value = float(metrics[training["selection_metric"]])
        if selection_value > best_metric:
            best_metric = selection_value
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
        logger.info("router seed=%s epoch=%s dev=%s", args.seed, epoch + 1, metrics)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    labels, probabilities = infer(model, loaders["dev"], device)
    metrics, predictions = multiclass_metrics(labels, probabilities)
    prediction_rows = []
    for row, probability, prediction in zip(dev_rows, probabilities, predictions):
        prediction_rows.append(
            {
                "question_id": row["question_id"],
                "split": "dev_calibration",
                "actual_route_label": int(row["label"]),
                "actual_route": row["route"],
                "route_probabilities": {
                    name: float(probability[index])
                    for index, name in enumerate(ROUTE_NAMES)
                },
                "predicted_route_label": int(prediction),
                "predicted_route": ROUTE_NAMES[int(prediction)],
            }
        )
    write_json_atomic(run_dir / "dev_calibration_metrics.json", metrics)
    write_jsonl_atomic(run_dir / "dev_calibration_predictions.jsonl", prediction_rows)
    manifest.update(
        {
            "status": "complete",
            "end_time_utc": utc_now(),
            "best_epoch": best_epoch,
            "best_dev_metric": best_metric,
            "selection_metric": training["selection_metric"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "dev_calibration_metrics": metrics,
        }
    )
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    logger.info("complete: %s", run_dir)


if __name__ == "__main__":
    main()

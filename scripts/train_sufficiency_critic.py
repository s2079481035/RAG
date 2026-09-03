"""Train matched query-only and query-plus-evidence sufficiency critics."""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from critic_metrics import classification_metrics
from evidence_utils import assert_disjoint_question_ids, validate_tuning_splits
from experiment_utils import collect_environment, seed_worker, set_global_seed, utc_now, write_json_atomic


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "critic" / "sufficiency_cross_encoder.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-mode", required=True, choices=["query_only", "query_evidence"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class SufficiencyDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate_batch(batch, *, tokenizer, input_mode: str, max_length: int):
    questions = [item["question"] for item in batch]
    kwargs = {
        "padding": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    if input_mode == "query_evidence":
        encoded = tokenizer(
            questions,
            [item["evidence_text"] for item in batch],
            truncation="only_second",
            **kwargs,
        )
    else:
        encoded = tokenizer(questions, truncation=True, **kwargs)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return encoded, labels


def token_lengths(tokenizer, items: list[dict], input_mode: str, batch_size: int = 128) -> list[int]:
    lengths = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        questions = [item["question"] for item in batch]
        if input_mode == "query_evidence":
            encoded = tokenizer(
                questions,
                [item["evidence_text"] for item in batch],
                truncation=False,
                padding=False,
                return_length=True,
                verbose=False,
            )
        else:
            encoded = tokenizer(
                questions,
                truncation=False,
                padding=False,
                return_length=True,
                verbose=False,
            )
        lengths.extend(int(value) for value in encoded["length"])
    return lengths


def length_summary(lengths: list[int], max_length: int) -> dict:
    values = np.asarray(lengths, dtype=int)
    truncated = int(np.sum(values > max_length))
    return {
        "samples": int(len(values)),
        "min": int(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "max_length": max_length,
        "truncated_samples": truncated,
        "truncation_ratio": truncated / len(values),
    }


def evaluate(model, loader, device, label_names, threshold):
    model.eval()
    labels = []
    probabilities = []
    with torch.no_grad():
        for encoded, batch_labels in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().numpy())
            labels.extend(batch_labels.numpy())
    return classification_metrics(
        labels,
        np.asarray(probabilities),
        label_names=label_names,
        sufficient_label=2,
        threshold=threshold,
    )[0]


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    seed = int(config["seed"])
    set_global_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    validate_tuning_splits(
        [
            config["training"]["selection_split"],
            config["evaluation"]["threshold_selection_split"],
        ]
    )

    data_dir = ROOT / config["data"]["directory"]
    train_items = load_jsonl(data_dir / config["data"]["train"])
    dev_items = load_jsonl(data_dir / config["data"]["dev"])
    test_items = load_jsonl(data_dir / config["data"]["test"])
    split_ids = {
        "train": {item["question_id"] for item in train_items},
        "dev": {item["question_id"] for item in dev_items},
        "test": {item["question_id"] for item in test_items},
    }
    assert_disjoint_question_ids(split_ids)
    del test_items

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or (
        ROOT / "experiments" / "sufficiency_critic" / f"{args.input_mode}_{timestamp}"
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = config["model"]
    revision = model_config.get("revision")
    load_kwargs = {
        "revision": revision,
        "local_files_only": not args.allow_download,
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name"],
        num_labels=int(model_config["num_labels"]),
        ignore_mismatched_sizes=True,
        **load_kwargs,
    )
    model_revision = getattr(model.config, "_commit_hash", None)
    max_length = int(model_config["max_length"])
    token_stats = {
        "input_mode": args.input_mode,
        "truncation_strategy": (
            "only_second_preserve_question" if args.input_mode == "query_evidence" else "longest_first"
        ),
        "train": length_summary(token_lengths(tokenizer, train_items, args.input_mode), max_length),
        "dev": length_summary(token_lengths(tokenizer, dev_items, args.input_mode), max_length),
    }
    write_json_atomic(run_dir / "token_length_stats.json", token_stats)

    resolved_config = {
        **config,
        "input_mode": args.input_mode,
        "run_dir": str(run_dir),
        "resolved_model_revision": model_revision,
        "local_files_only": not args.allow_download,
    }
    write_json_atomic(run_dir / "resolved_config.json", resolved_config)
    manifest = {
        "status": "running",
        "start_time_utc": utc_now(),
        "end_time_utc": None,
        "input_mode": args.input_mode,
        "seed": seed,
        "environment": collect_environment(ROOT),
        "checkpoint_path": None,
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training = config["training"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    collate = partial(
        collate_batch,
        tokenizer=tokenizer,
        input_mode=args.input_mode,
        max_length=max_length,
    )
    train_loader = DataLoader(
        SufficiencyDataset(train_items),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        SufficiencyDataset(dev_items),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        worker_init_fn=seed_worker,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    total_steps = len(train_loader) * epochs
    warmup_steps = math.floor(total_steps * float(training["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    threshold = float(config["evaluation"]["decision_threshold"])
    label_names = config["data"]["label_names"]
    history = []
    best_metric = -1.0
    best_epoch = None
    checkpoint_path = run_dir / "best.pt"
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (encoded, labels) in enumerate(train_loader, start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = labels.to(device)
            output = model(**encoded, labels=labels)
            output.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += output.loss.item()
            if step % 200 == 0:
                logger.info(
                    "%s epoch=%d step=%d/%d loss=%.4f",
                    args.input_mode,
                    epoch + 1,
                    step,
                    len(train_loader),
                    total_loss / step,
                )
        dev_metrics = evaluate(model, dev_loader, device, label_names, threshold)
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "dev": dev_metrics,
        }
        history.append(epoch_result)
        write_json_atomic(run_dir / "history.json", history)
        selection_value = float(dev_metrics[training["selection_metric"]])
        logger.info("%s epoch=%d dev=%s", args.input_mode, epoch + 1, dev_metrics)
        if selection_value > best_metric:
            best_metric = selection_value
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

    manifest.update(
        {
            "status": "complete",
            "end_time_utc": utc_now(),
            "best_epoch": best_epoch,
            "best_dev_metric": best_metric,
            "selection_metric": training["selection_metric"],
            "checkpoint_path": str(checkpoint_path),
            "model_revision": model_revision,
        }
    )
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    logger.info("complete: %s", run_dir)


if __name__ == "__main__":
    main()

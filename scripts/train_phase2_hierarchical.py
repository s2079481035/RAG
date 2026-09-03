"""Train the simple frozen-encoder chunk-wise Phase 2 feasibility baselines."""

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
from evidence_utils import assert_disjoint_question_ids, unique_supporting_facts, validate_tuning_splits
from experiment_utils import collect_environment, seed_worker, set_global_seed, utc_now, write_json_atomic
from phase2_controller_inputs import evidence_key, stage_metadata
from phase2_model import configure_cublas_workspace
from phase2_packing import pack_evidence
from train_phase2_controller import attach_diagnostics, select_threshold, write_jsonl_atomic


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase2" / "controller.json"
AGGREGATIONS = {"mean", "max", "mean_max"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--aggregation", choices=sorted(AGGREGATIONS), required=True)
    parser.add_argument("--evidence-mode", choices=["raw", "cumulative"], default="cumulative")
    parser.add_argument("--model", help="Local backbone path or Hugging Face model name")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_items(
    records: list[dict],
    chunk_by_id: dict,
    evidence_mode: str,
    tokenizer,
    pair_max_length: int,
):
    items = []
    view_key = evidence_key(evidence_mode)
    for record in records:
        view = record[view_key]
        chunks = []
        raw_tokens = 0
        visible_facts = set()
        truncated_pairs = 0
        fully_visible_chunks = 0
        query_text = f"{record['question']} {stage_metadata(record)}"
        for evidence_item in view["items"]:
            chunk = chunk_by_id[evidence_item["chunk_id"]]
            text, audit = pack_evidence(
                tokenizer,
                [evidence_item],
                chunk_by_id,
                question=query_text,
                max_length=pair_max_length,
                strategy="concat_truncate",
                pair_input=True,
            )
            raw_tokens += audit["raw_evidence_tokens"]
            truncated_pairs += int(audit["truncated"])
            fully_visible_chunks += int(audit["fully_visible_evidence_chunk_ratio"] == 1.0)
            visible_facts.update(
                (fact["title"], fact["sentence_id"])
                for fact in audit["visible_sentence_facts"]
            )
            chunks.append({"chunk_id": chunk["chunk_id"], "text": text})
        if not chunks:
            raise ValueError(f"Hierarchical sample has no evidence chunks: {record['question_id']}")
        items.append(
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "answer": record["answer"],
                "stage": record["stage"],
                "text_a": query_text,
                "chunks": chunks,
                "label": int(view["stop_label"]),
                "diagnostic_label": view["evidence_state"],
                "gold_supporting_facts": record["gold_supporting_facts"],
                "gold_supporting_fact_count": record["gold_supporting_fact_count"],
                "raw_evidence_tokens": raw_tokens,
                "hierarchical_packing_audit": {
                    "pair_max_length": pair_max_length,
                    "evidence_chunks": len(chunks),
                    "truncated_chunk_pairs": truncated_pairs,
                    "truncated_chunk_pair_ratio": truncated_pairs / len(chunks),
                    "fully_visible_evidence_chunk_ratio": fully_visible_chunks / len(chunks),
                    "visible_supporting_fact_ratio_evaluation_only": len(
                        set(unique_supporting_facts(record["gold_supporting_facts"]))
                        & visible_facts
                    )
                    / len(set(unique_supporting_facts(record["gold_supporting_facts"]))),
                },
                "evidence_mode": evidence_mode,
                "packing_audit": None,
            }
        )
    return items


def collate_hierarchical(batch, *, tokenizer, max_length: int):
    import torch

    questions = []
    chunks = []
    sample_indices = []
    for sample_index, item in enumerate(batch):
        for chunk in item["chunks"]:
            questions.append(item["text_a"])
            chunks.append(chunk["text"])
            sample_indices.append(sample_index)
    encoded = tokenizer(
        questions,
        chunks,
        truncation="only_second",
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return (
        encoded,
        torch.tensor(sample_indices, dtype=torch.long),
        torch.tensor([item["label"] for item in batch], dtype=torch.long),
    )


def build_model(backbone_name: str, aggregation: str, freeze_encoder: bool, load_kwargs: dict):
    import torch
    from transformers import AutoModel

    class HierarchicalController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(backbone_name, **load_kwargs)
            self.aggregation = aggregation
            self.freeze_encoder = freeze_encoder
            hidden = int(self.encoder.config.hidden_size)
            classifier_input = hidden * 2 if aggregation == "mean_max" else hidden
            self.classifier = torch.nn.Linear(classifier_input, 2)
            if freeze_encoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad = False

        def forward(self, encoded, sample_indices, batch_size, encoder_chunk_batch_size):
            representations = []
            if self.freeze_encoder:
                self.encoder.eval()
            for start in range(0, len(sample_indices), encoder_chunk_batch_size):
                current = {
                    key: value[start : start + encoder_chunk_batch_size]
                    for key, value in encoded.items()
                }
                if self.freeze_encoder:
                    with torch.no_grad():
                        hidden = self.encoder(**current).last_hidden_state[:, 0]
                else:
                    hidden = self.encoder(**current).last_hidden_state[:, 0]
                representations.append(hidden)
            chunk_representations = torch.cat(representations, dim=0)
            pooled = []
            for sample_index in range(batch_size):
                current = chunk_representations[sample_indices == sample_index]
                mean = current.mean(dim=0)
                maximum = current.max(dim=0).values
                if self.aggregation == "mean":
                    pooled.append(mean)
                elif self.aggregation == "max":
                    pooled.append(maximum)
                else:
                    pooled.append(torch.cat([mean, maximum], dim=-1))
            return self.classifier(torch.stack(pooled))

    return HierarchicalController()


def infer(model, loader, device, encoder_chunk_batch_size: int):
    import torch

    model.eval()
    labels = []
    probabilities = []
    with torch.no_grad():
        for encoded, sample_indices, batch_labels in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            sample_indices = sample_indices.to(device)
            logits = model(
                encoded,
                sample_indices,
                len(batch_labels),
                encoder_chunk_batch_size,
            )
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.numpy())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
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
    chunks = read_jsonl(
        ROOT / "data" / "phase2" / "chunks" / f"{args.variant}.jsonl"
    )
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    data_dir = ROOT / "data" / "phase2" / "controller" / args.variant
    source = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ["train", "dev"]}
    assert_disjoint_question_ids(
        {split: {item["question_id"] for item in records} for split, records in source.items()}
    )
    prepared = {
        split: prepare_items(
            records,
            chunk_by_id,
            args.evidence_mode,
            tokenizer,
            int(config["training"]["hierarchical_pair_max_length"]),
        )
        for split, records in source.items()
    }
    training = config["training"]
    freeze_encoder = bool(training["hierarchical_freeze_encoder"])
    model = build_model(backbone, args.aggregation, freeze_encoder, load_kwargs)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    representation = f"hierarchical_{args.aggregation}"
    run_name = f"query_evidence__{representation}__{args.evidence_mode}__{timestamp}"
    run_dir = args.run_dir or (
        ROOT / "experiments" / "phase2" / "controller" / args.variant / run_name
    )
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        run_dir / "resolved_config.json",
        {
            **config,
            "resolved": {
                "variant": args.variant,
                "baseline": "query_evidence",
                "representation": representation,
                "evidence_mode": args.evidence_mode,
                "backbone": backbone,
                "local_files_only": not args.allow_download,
                "encoder_frozen": freeze_encoder,
            },
        },
    )
    manifest = {
        "status": "running",
        "start_time_utc": utc_now(),
        "end_time_utc": None,
        "phase": 2,
        "phase1_frozen_commit": config["phase1_frozen_commit"],
        "variant": args.variant,
        "baseline": "query_evidence",
        "representation": representation,
        "evidence_mode": args.evidence_mode,
        "encoder_frozen": freeze_encoder,
        "seed": seed,
        "cublas_workspace_config": cublas_workspace_config,
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)

    generator = torch.Generator()
    generator.manual_seed(seed)
    collate = partial(
        collate_hierarchical,
        tokenizer=tokenizer,
        max_length=int(training["hierarchical_pair_max_length"]),
    )
    loaders = {
        "train": DataLoader(
            prepared["train"],
            batch_size=int(training["hierarchical_batch_size"]),
            shuffle=True,
            num_workers=int(training["num_workers"]),
            worker_init_fn=seed_worker,
            generator=generator,
            collate_fn=collate,
        ),
        "dev": DataLoader(
            prepared["dev"],
            batch_size=int(training["hierarchical_batch_size"]),
            shuffle=False,
            num_workers=int(training["num_workers"]),
            worker_init_fn=seed_worker,
            collate_fn=collate,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["hierarchical_gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    updates_per_epoch = math.ceil(len(loaders["train"]) / accumulation)
    total_steps = updates_per_epoch * epochs
    warmup_steps = math.floor(total_steps * float(training["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_function = torch.nn.CrossEntropyLoss()
    encoder_chunk_batch_size = int(training["hierarchical_encoder_chunk_batch_size"])
    fixed_threshold = float(training["threshold"])
    best_metric = -1.0
    best_epoch = None
    history = []
    checkpoint_path = run_dir / "best.pt"
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, (encoded, sample_indices, labels) in enumerate(loaders["train"], start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            sample_indices = sample_indices.to(device)
            labels = labels.to(device)
            logits = model(encoded, sample_indices, len(labels), encoder_chunk_batch_size)
            loss = loss_function(logits, labels)
            (loss / accumulation).backward()
            total_loss += loss.item()
            if step % accumulation == 0 or step == len(loaders["train"]):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 500 == 0:
                logger.info(
                    "%s epoch=%s step=%s/%s loss=%.4f",
                    representation,
                    epoch + 1,
                    step,
                    len(loaders["train"]),
                    total_loss / step,
                )
        labels, probabilities = infer(
            model, loaders["dev"], device, encoder_chunk_batch_size
        )
        metrics, predictions = binary_stop_metrics(labels, probabilities, fixed_threshold)
        metrics = attach_diagnostics(metrics, prepared["dev"], predictions)
        history.append(
            {"epoch": epoch + 1, "train_loss": total_loss / len(loaders["train"]), "dev": metrics}
        )
        write_json_atomic(run_dir / "history.json", history)
        value = float(metrics[training["selection_metric"]])
        if value > best_metric:
            best_metric = value
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    labels, probabilities = infer(model, loaders["dev"], device, encoder_chunk_batch_size)
    selected_threshold, _ = select_threshold(labels, probabilities, training["threshold_grid"])
    selected_metrics, predictions = binary_stop_metrics(labels, probabilities, selected_threshold)
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
                "raw_evidence_tokens": item["raw_evidence_tokens"],
                "evidence_chunk_count": len(item["chunks"]),
                "hierarchical_packing_audit": item["hierarchical_packing_audit"],
            }
        )
    write_jsonl_atomic(run_dir / "dev_predictions.jsonl", prediction_records)
    manifest.update(
        {
            "status": "complete",
            "end_time_utc": utc_now(),
            "best_epoch": best_epoch,
            "best_dev_metric_at_fixed_threshold": best_metric,
            "fixed_training_threshold": fixed_threshold,
            "selected_dev_threshold": selected_threshold,
            "selected_dev_metrics": selected_metrics,
            "checkpoint_path": str(checkpoint_path),
            "effective_training_batch_size": int(training["hierarchical_batch_size"])
            * accumulation,
        }
    )
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    logger.info("complete: %s", run_dir)


if __name__ == "__main__":
    main()

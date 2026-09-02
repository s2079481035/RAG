"""Evaluate and compare matched query-only and query-plus-evidence critics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from critic_metrics import classification_metrics
from evidence_utils import assert_disjoint_question_ids, validate_tuning_splits
from experiment_utils import collect_environment, git_commit, utc_now, write_json_atomic
from train_sufficiency_critic import load_jsonl, token_lengths


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-only-run", type=Path, required=True)
    parser.add_argument("--query-evidence-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(run_dir: Path, allow_download: bool, device):
    config = load_json(run_dir / "resolved_config.json")
    model_config = config["model"]
    revision = config.get("resolved_model_revision") or model_config.get("revision")
    kwargs = {"revision": revision, "local_files_only": not allow_download}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name"],
        num_labels=int(model_config["num_labels"]),
        ignore_mismatched_sizes=True,
        **kwargs,
    )
    try:
        state_dict = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return config, tokenizer, model


def predict(model, tokenizer, items, input_mode, max_length, batch_size, device):
    probabilities = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            questions = [item["question"] for item in batch]
            common = {
                "padding": True,
                "max_length": max_length,
                "return_tensors": "pt",
            }
            if input_mode == "query_evidence":
                encoded = tokenizer(
                    questions,
                    [item["evidence_text"] for item in batch],
                    truncation="only_second",
                    **common,
                )
            else:
                encoded = tokenizer(questions, truncation=True, **common)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return np.asarray(probabilities), elapsed_ms / len(items)


def save_confusion_matrix(path: Path, matrix, label_names):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *label_names])
        for label, row in zip(label_names, matrix):
            writer.writerow([label, *row])


def save_confusion_plot(path: Path, matrix, label_names, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(np.asarray(matrix), cmap="Blues")
    axis.set_xticks(range(len(label_names)), label_names, rotation=25, ha="right")
    axis.set_yticks(range(len(label_names)), label_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def evaluate_run(run_dir, expected_mode, output_dir, allow_download, device):
    config, tokenizer, model = load_model(run_dir, allow_download, device)
    input_mode = config["input_mode"]
    if input_mode != expected_mode:
        raise ValueError(f"Expected {expected_mode} run, got {input_mode}: {run_dir}")
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
    assert_disjoint_question_ids(
        {
            "train": {item["question_id"] for item in train_items},
            "dev": {item["question_id"] for item in dev_items},
            "test": {item["question_id"] for item in test_items},
        }
    )
    del train_items, dev_items
    max_length = int(config["model"]["max_length"])
    lengths = token_lengths(tokenizer, test_items, input_mode)
    probabilities, latency_ms = predict(
        model,
        tokenizer,
        test_items,
        input_mode,
        max_length,
        int(config["training"]["eval_batch_size"]),
        device,
    )
    labels = [item["label"] for item in test_items]
    threshold = float(config["evaluation"]["decision_threshold"])
    label_names = config["data"]["label_names"]
    metrics, predictions = classification_metrics(
        labels,
        probabilities,
        label_names=label_names,
        sufficient_label=2,
        threshold=threshold,
    )
    metrics["input_mode"] = input_mode
    metrics["samples"] = len(test_items)
    metrics["mean_inference_latency_ms"] = latency_ms
    metrics["truncated_samples"] = int(sum(length > max_length for length in lengths))
    metrics["truncation_ratio"] = metrics["truncated_samples"] / len(lengths)
    write_json_atomic(output_dir / "metrics.json", metrics)
    save_confusion_matrix(output_dir / "confusion_matrix.csv", metrics["confusion_matrix"], label_names)
    save_confusion_plot(
        output_dir / "confusion_matrix.png",
        metrics["confusion_matrix"],
        label_names,
        input_mode.replace("_", " ").title(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item, length, probability, prediction in zip(
            test_items, lengths, probabilities, predictions
        ):
            record = {
                **item,
                "critic_scores": {
                    name: float(probability[index]) for index, name in enumerate(label_names)
                },
                "critic_score": float(probability[2]),
                "critic_prediction": label_names[int(prediction)],
                "critic_prediction_id": int(prediction),
                "retrieval_action": "stop" if prediction == 2 else "escalate",
                "token_count": int(length),
                "was_truncated": bool(length > max_length),
                "latency": {
                    "milliseconds": latency_ms,
                    "source": "batched_test_mean",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    bad_cases = []
    for item, probability, prediction in zip(test_items, probabilities, predictions):
        true_label = int(item["label"])
        prediction = int(prediction)
        if prediction == true_label:
            continue
        bad_cases.append(
            {
                **item,
                "true_label": label_names[true_label],
                "predicted_label": label_names[prediction],
                "bad_case_type": (
                    "false_stop"
                    if prediction == 2 and true_label != 2
                    else "false_escalation"
                    if prediction != 2 and true_label == 2
                    else "insufficient_partial_confusion"
                ),
                "critic_scores": {
                    name: float(probability[index]) for index, name in enumerate(label_names)
                },
            }
        )
    bad_cases.sort(
        key=lambda item: (
            item["bad_case_type"] != "false_stop",
            -item["critic_scores"][item["predicted_label"]],
        )
    )
    requested = int(config["evaluation"]["bad_case_count"])
    with (output_dir / "bad_cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in bad_cases[:requested]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    metrics["bad_cases_saved"] = min(requested, len(bad_cases))
    write_json_atomic(output_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    start_time = utc_now()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "results" / "critic" / f"comparison_{timestamp}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_by_mode = {}
    for mode, run_dir in [
        ("query_only", args.query_only_run),
        ("query_evidence", args.query_evidence_run),
    ]:
        mode_output = output_dir / mode
        mode_output.mkdir(parents=True, exist_ok=True)
        metrics_by_mode[mode] = evaluate_run(
            run_dir, mode, mode_output, args.allow_download, device
        )
        logger.info("evaluated %s", mode)

    rows = []
    for mode, metrics in metrics_by_mode.items():
        sufficient = metrics["per_class"]["sufficient"]
        rows.append(
            {
                "critic": mode,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "sufficient_precision": sufficient["precision"],
                "sufficient_recall": sufficient["recall"],
                "sufficient_f1": sufficient["f1"],
                "false_stop_rate": metrics["false_stop_rate"],
                "sufficient_auroc_ovr": metrics["sufficient_auroc_ovr"],
                "truncation_ratio": metrics["truncation_ratio"],
                "mean_inference_latency_ms": metrics["mean_inference_latency_ms"],
            }
        )
    comparison_csv = output_dir / "critic_comparison.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Query-only vs Query + Evidence Critic",
        "",
        "Test split is evaluation-only. Both critics use the same data split, backbone, seed, optimizer, training steps, and sufficient threshold.",
        "",
        "| Critic | Accuracy | Macro F1 | Sufficient P | Sufficient R | Sufficient F1 | False Stop Rate | Sufficient AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['critic']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['sufficient_precision']:.4f} | {row['sufficient_recall']:.4f} | "
            f"{row['sufficient_f1']:.4f} | {row['false_stop_rate']:.4f} | "
            f"{row['sufficient_auroc_ovr']:.4f} |"
        )
    (output_dir / "critic_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "start_time_utc": start_time,
        "end_time_utc": utc_now(),
        "git_commit": git_commit(ROOT),
        "query_only_run": str(args.query_only_run),
        "query_evidence_run": str(args.query_evidence_run),
        "output_dir": str(output_dir),
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(output_dir / "evaluation_manifest.json", manifest)
    logger.info("comparison saved: %s", output_dir)


if __name__ == "__main__":
    main()

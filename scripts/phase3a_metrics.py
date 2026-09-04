"""Pure analysis helpers for Phase 3A Controller experiments."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


METRIC_NAMES = [
    "macro_f1",
    "stop_f1",
    "auroc",
    "false_stop_rate",
    "unnecessary_escalation_rate",
]


def coverage_bucket(coverage: float) -> str:
    value = float(coverage)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Coverage must be in [0, 1], got {value}")
    if value == 0.0:
        return "easy_continue"
    if value < 0.5:
        return "medium_partial"
    if value < 1.0:
        return "hard_partial"
    return "sufficient"


def metrics_from_decisions(labels, predictions, probabilities) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or predictions.shape != labels.shape or probabilities.shape != labels.shape:
        raise ValueError("labels, predictions, and probabilities must be same-length vectors")
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    actual_continue = int(np.sum(labels == 0))
    actual_stop = int(np.sum(labels == 1))
    false_stops = int(np.sum((labels == 0) & (predictions == 1)))
    escalations = int(np.sum((labels == 1) & (predictions == 0)))
    auroc = None
    if len(np.unique(labels)) == 2:
        auroc = float(roc_auc_score(labels, probabilities))
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "stop_precision": float(precision[1]),
        "stop_recall": float(recall[1]),
        "stop_f1": float(f1[1]),
        "stop_support": int(support[1]),
        "auroc": auroc,
        "false_stop_count": false_stops,
        "actual_continue_count": actual_continue,
        "false_stop_rate": false_stops / actual_continue if actual_continue else None,
        "unnecessary_escalation_count": escalations,
        "actual_stop_count": actual_stop,
        "unnecessary_escalation_rate": escalations / actual_stop if actual_stop else None,
    }


def average_ranks(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_correlation(left, right) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Spearman inputs must be same-length vectors")
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])
    return value if math.isfinite(value) else None


def coverage_regression_metrics(true_coverage, predicted_coverage) -> dict:
    truth = np.asarray(true_coverage, dtype=float)
    predicted = np.asarray(predicted_coverage, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("Coverage values must be same-length vectors")
    return {
        "samples": int(len(truth)),
        "mae": float(np.mean(np.abs(truth - predicted))) if len(truth) else None,
        "spearman": spearman_correlation(truth, predicted),
    }


def sampling_group(item: dict, strategy: str) -> str:
    if strategy == "balanced_stop_continue":
        return "stop" if int(item["label"]) == 1 else "continue"
    if strategy == "hard_partial_aware":
        if int(item["label"]) == 1:
            return "stop"
        return (
            "continue_hard_partial"
            if coverage_bucket(float(item["coverage_target"])) == "hard_partial"
            else "continue_non_hard"
        )
    if strategy == "natural":
        return "natural"
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def sampling_weights(items: list[dict], strategy: str, target_group_mass: dict) -> tuple[list[float], dict]:
    if strategy == "natural":
        return [], {"strategy": strategy, "group_counts": {"natural": len(items)}}
    groups = [sampling_group(item, strategy) for item in items]
    counts = Counter(groups)
    expected = set(target_group_mass)
    if set(counts) != expected:
        raise ValueError(f"Sampling groups {sorted(counts)} do not match configured groups {sorted(expected)}")
    total_mass = sum(float(value) for value in target_group_mass.values())
    if not math.isclose(total_mass, 1.0):
        raise ValueError(f"Target sampling mass must sum to 1, got {total_mass}")
    weights = [float(target_group_mass[group]) / counts[group] for group in groups]
    return weights, {
        "strategy": strategy,
        "group_counts": dict(counts),
        "target_group_mass": {key: float(value) for key, value in target_group_mass.items()},
        "num_samples": len(items),
        "replacement": True,
    }

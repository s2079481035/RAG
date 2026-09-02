"""Metrics for three-class evidence sufficiency prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score


def predictions_with_sufficient_threshold(
    probabilities: np.ndarray,
    sufficient_label: int = 2,
    threshold: float = 0.5,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [samples, classes]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    predictions = probabilities.argmax(axis=1)
    below_threshold = probabilities[:, sufficient_label] < threshold
    non_sufficient = probabilities.copy()
    non_sufficient[:, sufficient_label] = -np.inf
    predictions[below_threshold] = non_sufficient[below_threshold].argmax(axis=1)
    predictions[~below_threshold] = sufficient_label
    return predictions


def classification_metrics(
    labels,
    probabilities,
    *,
    label_names: list[str],
    sufficient_label: int = 2,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    expected = (len(labels), len(label_names))
    if probabilities.shape != expected:
        raise ValueError(f"Expected probability shape {expected}, got {probabilities.shape}")
    predictions = predictions_with_sufficient_threshold(
        probabilities, sufficient_label=sufficient_label, threshold=threshold
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    non_sufficient = labels != sufficient_label
    false_stop_count = int(np.sum(non_sufficient & (predictions == sufficient_label)))
    non_sufficient_count = int(np.sum(non_sufficient))
    sufficient_targets = (labels == sufficient_label).astype(int)
    sufficient_auroc = None
    if len(np.unique(sufficient_targets)) == 2:
        sufficient_auroc = float(
            roc_auc_score(sufficient_targets, probabilities[:, sufficient_label])
        )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "sufficient_threshold": threshold,
        "sufficient_auroc_ovr": sufficient_auroc,
        "false_stop_count": false_stop_count,
        "actual_non_sufficient_count": non_sufficient_count,
        "false_stop_rate": (
            false_stop_count / non_sufficient_count if non_sufficient_count else 0.0
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(label_names)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(label_names)))
        ).tolist(),
    }
    return metrics, predictions


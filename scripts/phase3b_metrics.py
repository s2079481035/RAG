"""Pure calibration, risk, and trajectory helpers for Phase 3B."""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from phase3a_metrics import coverage_bucket, metrics_from_decisions
except ModuleNotFoundError:  # Imported as scripts.phase3b_metrics in tests.
    from scripts.phase3a_metrics import coverage_bucket, metrics_from_decisions


def deterministic_question_split(
    question_ids: list[str], calibration_count: int, seed: int
) -> tuple[list[str], list[str]]:
    unique = sorted(set(question_ids))
    if len(unique) != len(question_ids):
        raise ValueError("Question IDs must be unique before splitting")
    if not 0 < calibration_count < len(unique):
        raise ValueError("Calibration count must leave non-empty calibration and policy sets")
    random.Random(seed).shuffle(unique)
    calibration = sorted(unique[:calibration_count])
    policy = sorted(unique[calibration_count:])
    return calibration, policy


def probability_to_logit(probabilities, epsilon: float = 1e-7) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional vector")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def temperature_scale(probabilities, temperature: float, epsilon: float = 1e-7) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature must be positive and finite")
    logits = probability_to_logit(probabilities, epsilon)
    scaled = logits / temperature
    return 1.0 / (1.0 + np.exp(-scaled))


def binary_nll(labels, probabilities, epsilon: float = 1e-7) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("Labels and probabilities must be same-length vectors")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Binary labels must be 0 or 1")
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))


def expected_calibration_error(labels, probabilities, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("Labels and probabilities must be same-length vectors")
    if bins <= 0:
        raise ValueError("ECE bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        mask = (
            (probabilities >= left) & (probabilities <= right)
            if index == bins - 1
            else (probabilities >= left) & (probabilities < right)
        )
        count = int(np.sum(mask))
        if count:
            confidence = float(np.mean(probabilities[mask]))
            frequency = float(np.mean(labels[mask]))
            error += count / total * abs(confidence - frequency)
    return float(error)


def calibration_metrics(labels, probabilities, bins: int, epsilon: float) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    auroc = None
    if len(np.unique(labels)) == 2:
        auroc = float(roc_auc_score(labels, probabilities))
    return {
        "samples": int(len(labels)),
        "nll": binary_nll(labels, probabilities, epsilon),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": expected_calibration_error(labels, probabilities, bins),
        "auroc": auroc,
    }


def fit_temperature(
    labels,
    probabilities,
    *,
    log_bounds: tuple[float, float],
    iterations: int,
    epsilon: float,
) -> tuple[float, float]:
    """Minimize binary NLL over log-temperature with golden-section search."""
    if iterations <= 0:
        raise ValueError("Optimizer iterations must be positive")
    left, right = map(float, log_bounds)
    if not left < right:
        raise ValueError("Temperature log bounds must be increasing")
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(log_temperature: float) -> float:
        scaled = temperature_scale(probabilities, math.exp(log_temperature), epsilon)
        return binary_nll(labels, scaled, epsilon)

    middle_left = right - ratio * (right - left)
    middle_right = left + ratio * (right - left)
    value_left = objective(middle_left)
    value_right = objective(middle_right)
    for _ in range(iterations):
        if value_left <= value_right:
            right = middle_right
            middle_right, value_right = middle_left, value_left
            middle_left = right - ratio * (right - left)
            value_left = objective(middle_left)
        else:
            left = middle_left
            middle_left, value_left = middle_right, value_right
            middle_right = left + ratio * (right - left)
            value_right = objective(middle_right)
    log_temperature = (left + right) / 2.0
    temperature = math.exp(log_temperature)
    return temperature, objective(log_temperature)


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    if not 0.0 <= start <= stop <= 1.0 or step <= 0.0:
        raise ValueError("Invalid threshold grid")
    count = int(round((stop - start) / step))
    values = [round(start + index * step, 12) for index in range(count + 1)]
    if not math.isclose(values[-1], stop, abs_tol=1e-12):
        raise ValueError("Threshold grid must include its stop value exactly")
    return values


def rows_by_question(rows: list[dict], stage_order: list[str]) -> dict[str, list[dict]]:
    grouped = defaultdict(dict)
    for row in rows:
        qid, stage = row["question_id"], row["stage"]
        if stage in grouped[qid]:
            raise ValueError(f"Duplicate stage {stage} for {qid}")
        grouped[qid][stage] = row
    expected = set(stage_order)
    result = {}
    for qid, stages in grouped.items():
        if set(stages) != expected:
            raise ValueError(f"Question {qid} has stages {sorted(stages)}, expected {stage_order}")
        result[qid] = [stages[stage] for stage in stage_order]
    return result


def actionable_metrics(rows: list[dict], probability_key: str, threshold: float) -> dict:
    labels = np.asarray([int(row["actual_stop_label"]) for row in rows], dtype=int)
    probabilities = np.asarray([float(row[probability_key]) for row in rows], dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    metrics = metrics_from_decisions(labels, predictions, probabilities)
    metrics["stop_decision_count"] = int(np.sum(predictions == 1))
    metrics["stop_decision_rate"] = float(np.mean(predictions == 1))
    hard = np.asarray(
        [
            int(row["actual_stop_label"]) == 0
            and coverage_bucket(float(row["true_coverage"])) == "hard_partial"
            for row in rows
        ],
        dtype=bool,
    )
    hard_false = int(np.sum(hard & (predictions == 1)))
    metrics["hard_partial_continue_count"] = int(np.sum(hard))
    metrics["hard_partial_false_stop_count"] = hard_false
    metrics["hard_partial_false_stop_rate"] = (
        hard_false / int(np.sum(hard)) if np.any(hard) else None
    )
    return metrics


def operational_metrics(
    grouped: dict[str, list[dict]], probability_key: str, threshold: float
) -> dict:
    """Score only Controller decisions reached by the sequential policy."""
    reached = []
    for question_id in sorted(grouped):
        for row in grouped[question_id][:-1]:
            prediction = int(float(row[probability_key]) >= threshold)
            reached.append((row, prediction))
            if prediction == 1:
                break
    if not reached:
        raise ValueError("Sequential policy reached no actionable Controller decisions")
    labels = np.asarray([int(row["actual_stop_label"]) for row, _ in reached], dtype=int)
    predictions = np.asarray([prediction for _, prediction in reached], dtype=int)
    probabilities = np.asarray([float(row[probability_key]) for row, _ in reached], dtype=float)
    metrics = metrics_from_decisions(labels, predictions, probabilities)
    metrics["stop_decision_count"] = int(np.sum(predictions == 1))
    metrics["stop_decision_rate"] = float(np.mean(predictions == 1))
    hard = np.asarray(
        [
            int(row["actual_stop_label"]) == 0
            and coverage_bucket(float(row["true_coverage"])) == "hard_partial"
            for row, _ in reached
        ],
        dtype=bool,
    )
    hard_false = int(np.sum(hard & (predictions == 1)))
    metrics["reached_actionable_decisions"] = len(reached)
    metrics["hard_partial_continue_count"] = int(np.sum(hard))
    metrics["hard_partial_false_stop_count"] = hard_false
    metrics["hard_partial_false_stop_rate"] = (
        hard_false / int(np.sum(hard)) if np.any(hard) else None
    )
    return metrics


def simulate_trajectory(
    question_rows: list[dict],
    probability_key: str,
    threshold: float,
    *,
    include_evidence: bool = True,
) -> dict:
    if len(question_rows) < 2:
        raise ValueError("A trajectory needs at least one actionable and one final stage")
    decisions = []
    selected = question_rows[-1]
    for row in question_rows[:-1]:
        stop = float(row[probability_key]) >= threshold
        decision = {
            "stage": row["stage"],
            "stop_probability": float(row[probability_key]),
            "decision": "stop" if stop else "continue",
            "actual_stop_label": int(row["actual_stop_label"]),
            "actual_three_class_label": row["actual_three_class_label"],
            "true_coverage": float(row["true_coverage"]),
        }
        if include_evidence:
            decision["evidence_chunk_ids"] = list(row.get("evidence_chunk_ids", []))
            decision["evidence_document_titles"] = list(
                row.get("evidence_document_titles", [])
            )
        decisions.append(decision)
        if stop:
            selected = row
            break
    return {
        "question_id": selected["question_id"],
        "final_stage": selected["stage"],
        "final_stage_index": int(selected["stage_index"]),
        "decisions": decisions,
        "controller_calls": len(decisions),
        "final_true_coverage": float(selected["true_coverage"]),
        "final_evidence_state": selected["actual_three_class_label"],
    }

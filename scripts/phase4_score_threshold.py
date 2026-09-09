"""Pure score extraction and normalization for the Phase 4 simple baseline."""

from __future__ import annotations

import math
from collections import defaultdict

from phase2_retrieval import parse_stage


def top1_score(record: dict) -> float:
    method, _ = parse_stage(record["stage"])
    statistics = record["retrieval_statistics"]
    if statistics.get("current_method") != method:
        raise ValueError(f"Current retrieval method mismatch for {record['stage']}")
    value = statistics.get(method, {}).get("max")
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Missing finite top-1 score for {record['question_id']} {record['stage']}")
    return float(value)


def fit_stage_minmax(records: list[dict], stages: list[str]) -> dict[str, dict[str, float]]:
    requested = set(stages)
    values = defaultdict(list)
    for record in records:
        if record["stage"] in requested:
            values[record["stage"]].append(top1_score(record))
    missing = requested - set(values)
    if missing:
        raise ValueError(f"Cannot fit score normalization; missing stages: {sorted(missing)}")
    return {
        stage: {"min": min(stage_values), "max": max(stage_values)}
        for stage, stage_values in sorted(values.items())
    }


def minmax_score(value: float, parameters: dict[str, float]) -> float:
    low = float(parameters["min"])
    high = float(parameters["max"])
    if high < low:
        raise ValueError("Invalid min-max parameters")
    if high == low:
        return 0.5
    return min(1.0, max(0.0, (float(value) - low) / (high - low)))


def attach_normalized_scores(
    records: list[dict], parameters: dict[str, dict[str, float]]
) -> list[dict]:
    output = []
    for record in records:
        stage = record["stage"]
        score = top1_score(record)
        normalized = minmax_score(score, parameters[stage]) if stage in parameters else 0.0
        output.append(
            {
                **record,
                "top1_retrieval_score": score,
                "normalized_score": normalized,
            }
        )
    return output

"""Shared logic for the Phase 4 query-only Adaptive-RAG-style baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROUTE_NAMES = ["light", "medium", "heavy"]
STAGE_NAMES = ["dense@5", "hybrid@10", "rerank@20"]
ROUTE_TO_STAGE = dict(zip(ROUTE_NAMES, STAGE_NAMES))
STAGE_TO_ROUTE = dict(zip(STAGE_NAMES, ROUTE_NAMES))
ALLOWED_DEV_SPLITS = {"train_core", "dev_calibration", "dev_policy"}


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_trajectories(path: Path, split: str) -> Iterator[list[dict]]:
    current_id = None
    current: list[dict] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        if row.get("split") != split:
            raise ValueError(f"Unexpected split in {path}: {row.get('split')}")
        question_id = row["question_id"]
        if current_id is not None and question_id != current_id:
            if current_id in seen:
                raise ValueError(f"Non-contiguous duplicate question: {current_id}")
            seen.add(current_id)
            validate_trajectory(current)
            yield current
            current = []
        current_id = question_id
        current.append(row)
    if current:
        if current_id in seen:
            raise ValueError(f"Non-contiguous duplicate question: {current_id}")
        validate_trajectory(current)
        yield current


def validate_trajectory(rows: list[dict]) -> None:
    stages = [row["stage"] for row in rows]
    if stages != STAGE_NAMES:
        raise ValueError(f"Invalid trajectory for {rows[0]['question_id']}: {stages}")
    questions = {row["question"] for row in rows}
    if len(questions) != 1:
        raise ValueError(f"Question text changed within trajectory: {rows[0]['question_id']}")
    coverage = [
        float(row["cumulative_evidence_memory"]["supporting_fact_recall"])
        for row in rows
    ]
    if any(right < left for left, right in zip(coverage, coverage[1:])):
        raise ValueError(f"Cumulative coverage decreased: {rows[0]['question_id']}")


def route_target(rows: list[dict]) -> dict:
    validate_trajectory(rows)
    sufficient = [
        bool(row["cumulative_evidence_memory"]["stop_label"])
        for row in rows
    ]
    first = next((index for index, value in enumerate(sufficient) if value), None)
    label = first if first is not None else 2
    return {
        "label": label,
        "route": ROUTE_NAMES[label],
        "first_sufficient_stage": STAGE_NAMES[first] if first is not None else None,
        "final_sufficient": sufficient[-1],
        "recoverable": first is not None,
    }


def router_record(rows: list[dict], split: str) -> dict:
    target = route_target(rows)
    return {
        "question_id": rows[0]["question_id"],
        "split": split,
        "question": rows[0]["question"],
        "label": target["label"],
        "route": target["route"],
        "first_sufficient_stage": target["first_sufficient_stage"],
        "final_sufficient": target["final_sufficient"],
        "recoverable": target["recoverable"],
    }


def multiclass_metrics(labels, probabilities) -> tuple[dict, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), len(ROUTE_NAMES)):
        raise ValueError("Router probability matrix has the wrong shape")
    predictions = probabilities.argmax(axis=1)
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(ROUTE_NAMES)))
        ).tolist(),
    }, predictions


def stage_latency_ms(row: dict) -> float:
    values = row["latency_ms"]
    dense = float(values["query_embedding_run_mean"]) + float(values["dense_search"])
    if row["stage"] == "dense@5":
        return dense
    hybrid = dense + float(values["bm25_scoring"]) + float(values["rrf_fusion"])
    if row["stage"] == "hybrid@10":
        return hybrid
    if row["stage"] == "rerank@20":
        return hybrid + float(values["reranker_allocated"])
    raise ValueError(f"Unknown stage: {row['stage']}")


def selected_stage_metrics(rows: list[dict], selected_index: int) -> dict:
    row = rows[selected_index]
    view = row["cumulative_evidence_memory"]
    return {
        "stage_index": selected_index + 1,
        "chunks": len(view["items"]),
        "unique_titles": len(set(view["document_titles"])),
        "reranker_calls": int(selected_index == 2),
        "latency_ms": stage_latency_ms(row),
        "supporting_fact_recall": float(view["supporting_fact_recall"]),
        "complete": int(view["stop_label"]),
    }


def summarize_routes(
    trajectories: dict[str, list[dict]], predictions: dict[str, int]
) -> dict:
    if set(trajectories) != set(predictions):
        missing = sorted(set(trajectories) - set(predictions))[:5]
        extra = sorted(set(predictions) - set(trajectories))[:5]
        raise ValueError(f"Router/source mismatch; missing={missing}, extra={extra}")
    totals = Counter()
    route_counts = Counter()
    recoverable = 0
    correct_recoverable = 0
    under_retrieval = 0
    unrecoverable_costs = defaultdict(float)
    unrecoverable = 0
    for question_id in sorted(trajectories):
        rows = trajectories[question_id]
        predicted_index = int(predictions[question_id])
        if predicted_index not in range(3):
            raise ValueError(f"Invalid route prediction for {question_id}: {predicted_index}")
        target = route_target(rows)
        selected = selected_stage_metrics(rows, predicted_index)
        route_counts[ROUTE_NAMES[predicted_index]] += 1
        for key, value in selected.items():
            if key != "complete":
                totals[key] += value
        totals["complete"] += selected["complete"]
        if target["recoverable"]:
            recoverable += 1
            oracle_index = int(target["label"])
            correct_recoverable += int(predicted_index == oracle_index)
            under_retrieval += int(predicted_index < oracle_index)
        else:
            unrecoverable += 1
            for key in ("chunks", "reranker_calls", "latency_ms"):
                unrecoverable_costs[key] += selected[key]
    questions = len(trajectories)
    result = {
        "questions": questions,
        "light_route_rate": route_counts["light"] / questions,
        "medium_route_rate": route_counts["medium"] / questions,
        "heavy_route_rate": route_counts["heavy"] / questions,
        "average_final_stage": totals["stage_index"] / questions,
        "average_retrieved_chunks": totals["chunks"] / questions,
        "average_unique_titles": totals["unique_titles"] / questions,
        "average_reranker_calls": totals["reranker_calls"] / questions,
        "average_retrieval_latency_ms": totals["latency_ms"] / questions,
        "supporting_fact_recall": totals["supporting_fact_recall"] / questions,
        "complete_evidence_coverage": totals["complete"] / questions,
        "terminal_retrieval_failure_rate": sum(
            not route_target(rows)["final_sufficient"] for rows in trajectories.values()
        ) / questions,
        "recoverable_questions": recoverable,
        "recoverable_route_accuracy": (
            correct_recoverable / recoverable if recoverable else None
        ),
        "recoverable_under_retrieval_rate": (
            under_retrieval / recoverable if recoverable else None
        ),
        "unrecoverable_questions": unrecoverable,
        "unrecoverable_average_chunks": (
            unrecoverable_costs["chunks"] / unrecoverable if unrecoverable else None
        ),
        "unrecoverable_average_reranker_calls": (
            unrecoverable_costs["reranker_calls"] / unrecoverable
            if unrecoverable else None
        ),
        "unrecoverable_average_retrieval_latency_ms": (
            unrecoverable_costs["latency_ms"] / unrecoverable
            if unrecoverable else None
        ),
    }
    return result


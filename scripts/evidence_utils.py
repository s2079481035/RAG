"""Shared evidence-coverage and split-safety utilities for research experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


LABEL_BY_STATE = {
    "insufficient": 0,
    "partial": 1,
    "sufficient": 2,
}


def normalize_supporting_fact(fact: Any) -> tuple[str, int]:
    """Return a canonical ``(title, sentence_id)`` pair."""
    if isinstance(fact, Mapping):
        title = fact.get("title", fact.get("document_title"))
        sentence_id = fact.get("sentence_id", fact.get("sent_id"))
    elif isinstance(fact, Sequence) and not isinstance(fact, (str, bytes)) and len(fact) == 2:
        title, sentence_id = fact
    else:
        raise ValueError(f"Unsupported supporting-fact value: {fact!r}")
    if not isinstance(title, str) or not title:
        raise ValueError(f"Supporting-fact title must be a non-empty string: {fact!r}")
    if isinstance(sentence_id, bool) or not isinstance(sentence_id, int) or sentence_id < 0:
        raise ValueError(f"Supporting-fact sentence_id must be a non-negative integer: {fact!r}")
    return title, sentence_id


def serialize_supporting_fact(fact: tuple[str, int]) -> dict[str, Any]:
    return {"title": fact[0], "sentence_id": fact[1]}


def unique_supporting_facts(facts: Iterable[Any]) -> list[tuple[str, int]]:
    """Deduplicate facts while retaining their source order."""
    seen: set[tuple[str, int]] = set()
    unique: list[tuple[str, int]] = []
    for raw_fact in facts:
        fact = normalize_supporting_fact(raw_fact)
        if fact not in seen:
            seen.add(fact)
            unique.append(fact)
    return unique


def covered_supporting_facts(
    retrieved_chunk_ids: Iterable[str],
    chunk_metadata: Mapping[str, Mapping[str, Any]],
    gold_supporting_facts: Iterable[Any],
) -> list[dict[str, Any]]:
    """Map retrieved chunks to exact gold facts using title and sentence ID."""
    available: set[tuple[str, int]] = set()
    for chunk_id in dict.fromkeys(retrieved_chunk_ids):
        if chunk_id not in chunk_metadata:
            raise KeyError(f"Missing metadata for retrieved chunk: {chunk_id}")
        chunk = chunk_metadata[chunk_id]
        title = chunk.get("document_title", chunk.get("title"))
        sentence_ids = chunk.get("sentence_ids")
        if not isinstance(title, str) or not isinstance(sentence_ids, list):
            raise ValueError(f"Invalid chunk metadata for {chunk_id}: {chunk!r}")
        for sentence_id in sentence_ids:
            available.add(normalize_supporting_fact((title, sentence_id)))

    covered = [fact for fact in unique_supporting_facts(gold_supporting_facts) if fact in available]
    return [serialize_supporting_fact(fact) for fact in covered]


def supporting_fact_metrics(
    retrieved_chunk_ids: Iterable[str],
    chunk_metadata: Mapping[str, Mapping[str, Any]],
    gold_supporting_facts: Iterable[Any],
) -> dict[str, Any]:
    gold = unique_supporting_facts(gold_supporting_facts)
    if not gold:
        raise ValueError("At least one gold supporting fact is required")
    covered = covered_supporting_facts(retrieved_chunk_ids, chunk_metadata, gold)
    recall = len(covered) / len(gold)
    if not 0.0 <= recall <= 1.0:
        raise AssertionError(f"Supporting-fact recall out of range: {recall}")
    if not covered:
        state = "insufficient"
    elif len(covered) == len(gold):
        state = "sufficient"
    else:
        state = "partial"
    return {
        "covered_supporting_facts": covered,
        "supporting_fact_recall": recall,
        "complete_evidence_coverage": int(state == "sufficient"),
        "evidence_state": state,
        "label": LABEL_BY_STATE[state],
    }


def unique_document_titles(
    retrieved_chunk_ids: Iterable[str],
    chunk_metadata: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for chunk_id in retrieved_chunk_ids:
        if chunk_id not in chunk_metadata:
            raise KeyError(f"Missing metadata for retrieved chunk: {chunk_id}")
        title = chunk_metadata[chunk_id].get("document_title", chunk_metadata[chunk_id].get("title"))
        if not isinstance(title, str):
            raise ValueError(f"Missing document title for chunk: {chunk_id}")
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def assert_disjoint_question_ids(split_to_ids: Mapping[str, Iterable[str]]) -> None:
    normalized = {name: set(ids) for name, ids in split_to_ids.items()}
    names = list(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"Question-ID leakage between {left_name} and {right_name}: "
                    f"{len(overlap)} overlaps, examples={examples}"
                )


def validate_tuning_splits(tuning_splits: Iterable[str], test_split: str = "test") -> None:
    used = set(tuning_splits)
    if test_split in used:
        raise ValueError(f"The test split cannot be used for tuning: {test_split}")


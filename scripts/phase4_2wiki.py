"""Pure validation and audit helpers for the official 2WikiMultiHopQA JSON."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Iterable
from statistics import mean, median
from typing import Any

from evidence_utils import normalize_supporting_fact


YEAR_RE = re.compile(r"^(?:1[0-9]{3}|20[0-9]{2})$")
NUMBER_RE = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)%?$")
MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def answer_type(answer: str) -> str:
    value = answer.strip()
    lowered = value.lower()
    if lowered in {"yes", "no"}:
        return "yes_no"
    if YEAR_RE.fullmatch(value):
        return "year"
    if NUMBER_RE.fullmatch(value.replace(",", "")):
        return "number"
    if any(month in lowered.split() for month in MONTHS) and any(
        character.isdigit() for character in value
    ):
        return "date"
    return "span"


def require_official_record(raw: dict[str, Any], official_split: str) -> dict[str, Any]:
    required = {"_id", "question", "answer", "supporting_facts", "context"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"{official_split} record is missing official fields: {missing}")
    qid = raw["_id"]
    if not isinstance(qid, str) or not qid:
        raise ValueError(f"Invalid 2Wiki _id: {qid!r}")
    if not isinstance(raw["question"], str) or not raw["question"].strip():
        raise ValueError(f"Question {qid} has invalid question text")
    if not isinstance(raw["answer"], str):
        raise ValueError(f"Question {qid} has invalid answer")

    supporting = raw["supporting_facts"]
    if not isinstance(supporting, list) or not supporting:
        raise ValueError(f"Question {qid} has no supporting facts")
    normalized_facts = []
    for fact in supporting:
        if not isinstance(fact, list) or len(fact) != 2:
            raise ValueError(
                f"Question {qid} does not use official [title, sentence_id] facts: {fact!r}"
            )
        title, sentence_id = normalize_supporting_fact(fact)
        normalized_facts.append({"title": title, "sentence_id": sentence_id})

    contexts = raw["context"]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"Question {qid} has no context documents")
    normalized_contexts = []
    for context_index, context in enumerate(contexts):
        if not isinstance(context, list) or len(context) != 2:
            raise ValueError(
                f"Question {qid} context {context_index} is not official [title, sentences]"
            )
        title, sentences = context
        if not isinstance(title, str) or not title:
            raise ValueError(f"Question {qid} context {context_index} has invalid title")
        if not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise ValueError(f"Question {qid} context {context_index} has invalid sentences")
        normalized_contexts.append(
            {"document_title": title, "sentence_texts": sentences}
        )

    return {
        "question_id": qid,
        "question": raw["question"],
        "answer": raw["answer"],
        "answer_type": answer_type(raw["answer"]),
        "question_type": raw.get("type"),
        "gold_supporting_facts": normalized_facts,
        "contexts": normalized_contexts,
        "evidences": raw.get("evidences"),
        "entity_ids": raw.get("entity_ids"),
        "official_split": official_split,
        "source_keys": sorted(raw),
    }


def deterministic_train_split(
    question_ids: Iterable[str],
    *,
    seed: int,
    dev_calibration_questions: int,
    dev_policy_questions: int,
) -> dict[str, str]:
    ids = sorted(question_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs in official Train")
    held_out = dev_calibration_questions + dev_policy_questions
    if held_out >= len(ids):
        raise ValueError("Internal calibration/policy splits consume all training questions")
    random.Random(seed).shuffle(ids)
    calibration = set(ids[:dev_calibration_questions])
    policy = set(ids[dev_calibration_questions:held_out])
    return {
        qid: (
            "dev_calibration"
            if qid in calibration
            else "dev_policy"
            if qid in policy
            else "train_core"
        )
        for qid in ids
    }


def numeric_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    ordered = sorted(values)
    p95_index = round(0.95 * (len(ordered) - 1))
    return {
        "min": ordered[0],
        "mean": mean(ordered),
        "median": median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def source_statistics(questions: list[dict[str, Any]]) -> dict[str, Any]:
    context_counts = [len(question["contexts"]) for question in questions]
    support_counts = [len(question["gold_supporting_facts"]) for question in questions]
    titles = [
        context["document_title"]
        for question in questions
        for context in question["contexts"]
    ]
    texts = [
        " ".join(context["sentence_texts"])
        for question in questions
        for context in question["contexts"]
    ]
    title_counts = Counter(titles)
    text_counts = Counter(texts)
    return {
        "questions": len(questions),
        "answer_types": dict(sorted(Counter(q["answer_type"] for q in questions).items())),
        "question_types": dict(
            sorted(Counter(str(q["question_type"]) for q in questions).items())
        ),
        "supporting_fact_count_distribution": dict(
            sorted(Counter(support_counts).items())
        ),
        "supporting_fact_count_summary": numeric_summary(support_counts),
        "raw_context_documents": len(titles),
        "unique_titles": len(title_counts),
        "repeated_title_groups": sum(count > 1 for count in title_counts.values()),
        "repeated_title_instances_beyond_first": sum(
            count - 1 for count in title_counts.values() if count > 1
        ),
        "unique_texts": len(text_counts),
        "duplicate_text_groups": sum(count > 1 for count in text_counts.values()),
        "duplicate_text_instances_beyond_first": sum(
            count - 1 for count in text_counts.values() if count > 1
        ),
        "contexts_per_question": numeric_summary(context_counts),
        "observed_source_key_sets": sorted(
            {tuple(question["source_keys"]) for question in questions}
        ),
    }


def split_digest(split_to_ids: dict[str, list[str]]) -> str:
    payload = "\n".join(
        f"{split}\t{qid}"
        for split in sorted(split_to_ids)
        for qid in sorted(split_to_ids[split])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

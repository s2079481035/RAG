"""Frozen prompt, extraction, and HotpotQA scoring helpers for Phase 3A."""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path


SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_rows_sha256(rows: list[dict], fields: list[str]) -> str:
    projected = [
        {field: str(row.get(field, "")) for field in fields}
        for row in rows
    ]
    payload = json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_short_answer(raw_output: str) -> str:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return ""
    answer = re.sub(
        r"^(?:final\s+answer|answer)\s*:\s*",
        "",
        lines[0],
        flags=re.IGNORECASE,
    ).strip()
    return answer.strip(" \t\"'")


def normalize_answer(answer: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punctuation(text: str) -> str:
        return "".join(character for character in text if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(answer.lower())).split())


def exact_match_score(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def token_f1_score(prediction: str, gold: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if (
        normalized_prediction in SPECIAL_ANSWERS
        or normalized_gold in SPECIAL_ANSWERS
    ) and normalized_prediction != normalized_gold:
        return 0.0
    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["selection_split"] != "dev" or not config["test_is_evaluation_only"]:
        raise ValueError("Generation config must keep dev selection and evaluation-only test")
    decoding = config["decoding"]
    if decoding["do_sample"] or decoding["temperature"] is not None or decoding["num_beams"] != 1:
        raise ValueError("Phase 3A generation is preregistered as greedy decoding")
    return config

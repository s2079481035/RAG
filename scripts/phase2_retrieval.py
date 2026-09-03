"""Pure helpers for deterministic Phase 2 retrieval and trajectory construction."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


STOPWORDS = set(
    "a an the and or but if because as of at by for with about to in on is are was were "
    "be been it its this that these those".split()
)
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")


def parse_stage(stage_name: str) -> tuple[str, int]:
    try:
        method, value = stage_name.rsplit("@", 1)
        k = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid controller stage name: {stage_name!r}") from error
    if not method or k < 1:
        raise ValueError(f"Invalid controller stage name: {stage_name!r}")
    return method, k


def bm25_tokenize(text: str) -> list[str]:
    """Match the tokenization used by the frozen Phase 1 BM25 implementation."""
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def retrieval_document_text(chunk: dict[str, Any], mode: str) -> str:
    if mode == "chunk_text":
        return str(chunk["chunk_text"])
    if mode == "title_and_chunk_text":
        return f"[TITLE] {chunk['document_title']} [TEXT] {chunk['chunk_text']}"
    raise ValueError(f"Unknown retrieval document text mode: {mode}")


def deterministic_top_indices(scores: Sequence[float], k: int) -> np.ndarray:
    """Rank descending by score and use corpus position as a deterministic tie-break."""
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if k < 1:
        raise ValueError("k must be positive")
    positions = np.arange(len(values))
    return np.lexsort((positions, -values))[: min(k, len(values))]


def rrf_fuse_with_components(
    dense_indices: Sequence[int],
    dense_scores: Sequence[float],
    bm25_scores: Sequence[float],
    *,
    dense_depth: int,
    bm25_depth: int,
    rrf_k: int,
    top: int,
) -> list[dict[str, Any]]:
    """Fuse two rankings while retaining all component ranks and scores."""
    if len(dense_indices) != len(dense_scores):
        raise ValueError("dense indices and scores must have equal lengths")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    dense_rank = {
        int(index): rank
        for rank, index in enumerate(dense_indices[:dense_depth], start=1)
    }
    dense_score = {
        int(index): float(score)
        for index, score in zip(dense_indices[:dense_depth], dense_scores[:dense_depth])
    }
    bm25_top = deterministic_top_indices(bm25_scores, bm25_depth)
    bm25_rank = {int(index): rank for rank, index in enumerate(bm25_top, start=1)}

    candidates = set(dense_rank) | set(bm25_rank)
    rows = []
    for index in candidates:
        score = 0.0
        if index in dense_rank:
            score += 1.0 / (rrf_k + dense_rank[index])
        if index in bm25_rank:
            score += 1.0 / (rrf_k + bm25_rank[index])
        rows.append(
            {
                "index": index,
                "rrf_score": score,
                "dense_rank": dense_rank.get(index),
                "bm25_rank": bm25_rank.get(index),
                "dense_score": dense_score.get(index),
                "bm25_score": float(bm25_scores[index]),
            }
        )
    rows.sort(key=lambda row: (-row["rrf_score"], row["index"]))
    return rows[: min(top, len(rows))]


def ranked_entries(
    indices: Sequence[int], scores: Sequence[float], doc_ids: Sequence[str]
) -> list[dict[str, Any]]:
    if len(indices) != len(scores):
        raise ValueError("indices and scores must have equal lengths")
    return [
        {"chunk_id": doc_ids[int(index)], "score": float(score), "rank": rank}
        for rank, (index, score) in enumerate(zip(indices, scores), start=1)
    ]


def cumulative_unique(stage_rankings: Iterable[Sequence[str]]) -> list[list[str]]:
    """Build stage-wise cumulative evidence without duplicating chunks."""
    seen: set[str] = set()
    cumulative: list[str] = []
    output = []
    for ranking in stage_rankings:
        for chunk_id in ranking:
            if chunk_id not in seen:
                seen.add(chunk_id)
                cumulative.append(chunk_id)
        output.append(list(cumulative))
    return output


def score_statistics(scores: Sequence[float]) -> dict[str, float | int | None]:
    """Return fixed-dimensional ranking features for the metadata baseline."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "max": None,
            "min": None,
            "mean": None,
            "std": None,
            "top1_top2_margin": None,
            "softmax_entropy": None,
        }
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return {
        "count": int(values.size),
        "max": float(values.max()),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "top1_top2_margin": float(values[0] - values[1]) if values.size > 1 else None,
        "softmax_entropy": entropy / math.log(values.size) if values.size > 1 else 0.0,
    }

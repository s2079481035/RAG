"""Retrieval helpers kept independent from model imports for unit testing."""

from __future__ import annotations

import numpy as np


def rrf_fuse_indices(
    dense_indices,
    bm25_scores,
    *,
    n_docs: int,
    rrf_k: int = 60,
    top: int = 20,
    dense_depth: int = 200,
    bm25_depth: int = 200,
):
    """Fuse dense ranks and BM25 scores with reciprocal rank fusion."""
    if n_docs <= 0:
        raise ValueError("n_docs must be positive")
    if len(bm25_scores) != n_docs:
        raise ValueError("BM25 scores must contain one score per document")
    rrf_scores = np.zeros(n_docs, dtype=float)
    for rank, doc_index in enumerate(dense_indices[:dense_depth]):
        rrf_scores[int(doc_index)] += 1.0 / (rrf_k + rank + 1)
    for rank, doc_index in enumerate(np.argsort(-np.asarray(bm25_scores))[:bm25_depth]):
        rrf_scores[int(doc_index)] += 1.0 / (rrf_k + rank + 1)
    top_indices = np.argsort(-rrf_scores)[: min(top, n_docs)]
    return top_indices, rrf_scores


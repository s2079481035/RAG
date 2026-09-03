"""Auditable evidence packing strategies for the Phase 2 Controller."""

from __future__ import annotations

import math
from typing import Any


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _chunk_tokens(tokenizer, item: dict, chunk: dict) -> dict[str, Any]:
    title = item.get("document_title", chunk["document_title"])
    header_ids = _encode(tokenizer, f"[DOC] [TITLE] {title} [TEXT] ")
    token_ids = list(header_ids)
    sentence_ends = []
    for position, (sentence_id, sentence_text) in enumerate(
        zip(chunk["sentence_ids"], chunk["sentence_texts"])
    ):
        if position:
            token_ids.extend(_encode(tokenizer, " "))
        token_ids.extend(_encode(tokenizer, sentence_text))
        sentence_ends.append((int(sentence_id), len(token_ids)))
    return {
        "chunk_id": chunk["chunk_id"],
        "document_title": title,
        "token_ids": token_ids,
        "header_tokens": len(header_ids),
        "sentence_ends": sentence_ends,
        "retrieval_score": item.get("retrieval_score"),
        "retrieval_rank": item.get("retrieval_rank"),
        "score_source": item.get("score_source"),
    }


def _distribute(
    lengths: list[int], bases: list[int], remaining: int, weights: list[float]
) -> list[int]:
    allocations = list(bases)
    while remaining > 0:
        active = [index for index, value in enumerate(allocations) if value < lengths[index]]
        if not active:
            break
        total_weight = sum(max(weights[index], 0.0) for index in active)
        if total_weight <= 0:
            normalized = {index: 1.0 / len(active) for index in active}
        else:
            normalized = {
                index: max(weights[index], 0.0) / total_weight for index in active
            }
        additions = {index: int(remaining * normalized[index]) for index in active}
        if not any(additions.values()):
            chosen = max(
                active,
                key=lambda index: (normalized[index], -allocations[index], -index),
            )
            additions[chosen] = 1
        consumed = 0
        for index in active:
            addition = min(additions[index], lengths[index] - allocations[index])
            allocations[index] += addition
            consumed += addition
        if consumed == 0:
            break
        remaining -= consumed
    return allocations


def allocate_tokens(
    chunks: list[dict], budget: int, strategy: str, minimum_chunk_tokens: int
) -> list[int]:
    """Allocate an exact token budget across already-tokenized chunks."""
    if budget < 0:
        raise ValueError("budget cannot be negative")
    if not chunks:
        return []
    lengths = [len(chunk["token_ids"]) for chunk in chunks]
    if strategy == "concat_truncate":
        allocations = []
        remaining = budget
        for length in lengths:
            allocation = min(length, remaining)
            allocations.append(allocation)
            remaining -= allocation
        return allocations

    mandatory = [
        min(length, max(minimum_chunk_tokens, chunk["header_tokens"]))
        for length, chunk in zip(lengths, chunks)
    ]
    if sum(mandatory) > budget:
        bases = [0] * len(chunks)
        return _distribute(lengths, bases, budget, [1.0] * len(chunks))
    remaining = budget - sum(mandatory)
    if strategy == "uniform_packing":
        weights = [1.0] * len(chunks)
    elif strategy == "score_aware_packing":
        ranks = [chunk.get("retrieval_rank") for chunk in chunks]
        weights = [1.0 / max(int(rank or len(chunks) + 1), 1) for rank in ranks]
        scores = [chunk.get("retrieval_score") for chunk in chunks]
        finite_scores = [float(value) for value in scores if value is not None]
        if finite_scores:
            minimum = min(finite_scores)
            maximum = max(finite_scores)
            if maximum > minimum:
                for index, score in enumerate(scores):
                    if score is not None:
                        weights[index] *= 1.0 + (float(score) - minimum) / (maximum - minimum)
    else:
        raise ValueError(f"Unknown packing strategy: {strategy}")
    return _distribute(lengths, mandatory, remaining, weights)


def evidence_budget(tokenizer, question: str, max_length: int, pair_input: bool) -> int:
    question_tokens = len(_encode(tokenizer, question)) if pair_input else 0
    special_tokens = tokenizer.num_special_tokens_to_add(pair=pair_input)
    return max(max_length - question_tokens - special_tokens, 0)


def pack_evidence(
    tokenizer,
    evidence_items: list[dict],
    chunk_by_id: dict,
    *,
    question: str,
    max_length: int,
    strategy: str,
    minimum_chunk_tokens: int = 12,
    pair_input: bool = True,
) -> tuple[str, dict]:
    tokenized = [
        _chunk_tokens(tokenizer, item, chunk_by_id[item["chunk_id"]])
        for item in evidence_items
    ]
    budget = evidence_budget(tokenizer, question, max_length, pair_input)
    allocations = allocate_tokens(tokenized, budget, strategy, minimum_chunk_tokens)
    text_parts = []
    visible_facts = []
    allocation_records = []
    for chunk, allocation in zip(tokenized, allocations):
        visible_ids = chunk["token_ids"][:allocation]
        text_parts.append(tokenizer.decode(visible_ids, skip_special_tokens=True))
        fully_visible_sentence_ids = [
            sentence_id
            for sentence_id, end in chunk["sentence_ends"]
            if end <= allocation
        ]
        visible_facts.extend(
            {"title": chunk["document_title"], "sentence_id": sentence_id}
            for sentence_id in fully_visible_sentence_ids
        )
        allocation_records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_title": chunk["document_title"],
                "raw_tokens": len(chunk["token_ids"]),
                "allocated_tokens": allocation,
                "title_fully_visible": allocation >= chunk["header_tokens"],
                "fully_visible_sentence_ids": fully_visible_sentence_ids,
                "retrieval_score": chunk["retrieval_score"],
                "retrieval_rank": chunk["retrieval_rank"],
                "score_source": chunk["score_source"],
            }
        )
    raw_tokens = sum(len(chunk["token_ids"]) for chunk in tokenized)
    nonempty = sum(allocation > 0 for allocation in allocations)
    fully_visible = sum(
        allocation >= len(chunk["token_ids"])
        for chunk, allocation in zip(tokenized, allocations)
    )
    audit = {
        "strategy": strategy,
        "max_length": max_length,
        "evidence_token_budget": budget,
        "raw_evidence_tokens": raw_tokens,
        "allocated_evidence_tokens": sum(allocations),
        "truncated": raw_tokens > budget,
        "evidence_chunks": len(tokenized),
        "visible_evidence_chunks": nonempty,
        "visible_evidence_chunk_ratio": nonempty / len(tokenized) if tokenized else 0.0,
        "fully_visible_evidence_chunk_ratio": (
            fully_visible / len(tokenized) if tokenized else 0.0
        ),
        "visible_sentence_facts": visible_facts,
        "allocations": allocation_records,
    }
    return " ".join(part for part in text_parts if part), audit


def length_bucket(raw_tokens: int) -> str:
    if raw_tokens <= 512:
        return "<=512"
    if raw_tokens <= 1024:
        return "513-1024"
    if raw_tokens <= 2048:
        return "1025-2048"
    return ">2048"


def gold_fact_bucket(count: int) -> str:
    if count <= 2:
        return "2"
    if count == 3:
        return "3"
    return "4+"

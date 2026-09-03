"""Construct fair, auditable inputs for Phase 2 Controller baselines."""

from __future__ import annotations

from typing import Any

try:
    from phase2_packing import pack_evidence
except ModuleNotFoundError:  # Imported as scripts.phase2_controller_inputs in tests.
    from scripts.phase2_packing import pack_evidence


NON_EVIDENCE_BASELINES = {"query_only", "query_stage", "query_stage_stats"}
EVIDENCE_BASELINES = {"evidence_only", "query_evidence"}


def stage_metadata(record: dict) -> str:
    return (
        f"[STAGE] {record['stage']} [K] {record['k']} "
        f"[METHOD] {record['retrieval_method']}"
    )


def statistics_text(statistics: dict) -> str:
    fields = []
    for method in ["dense", "bm25", "hybrid", "rerank"]:
        if method not in statistics:
            continue
        values = statistics[method]
        for key in ["max", "mean", "std", "top1_top2_margin", "softmax_entropy"]:
            value = values.get(key)
            rendered = "NA" if value is None else f"{float(value):.6g}"
            fields.append(f"{method}_{key}={rendered}")
    return "[STATS] " + " ".join(fields)


def evidence_key(evidence_mode: str) -> str:
    if evidence_mode == "raw":
        return "raw_stage_evidence"
    if evidence_mode == "cumulative":
        return "cumulative_evidence_memory"
    raise ValueError(f"Unknown evidence mode: {evidence_mode}")


def prepare_nonhierarchical_input(
    record: dict,
    *,
    baseline: str,
    representation: str,
    evidence_mode: str,
    tokenizer,
    chunk_by_id: dict,
    max_length: int,
    minimum_chunk_tokens: int = 12,
    include_stage_in_query_evidence: bool = True,
    evidence_items_override: list[dict] | None = None,
    title_only: bool = False,
) -> dict[str, Any]:
    if baseline not in NON_EVIDENCE_BASELINES | EVIDENCE_BASELINES:
        raise ValueError(f"Unknown Controller baseline: {baseline}")
    view = record[evidence_key(evidence_mode)]
    label = int(view["stop_label"])
    common = {
        "question_id": record["question_id"],
        "question": record["question"],
        "answer": record["answer"],
        "stage": record["stage"],
        "label": label,
        "diagnostic_label": view["evidence_state"],
        "gold_supporting_facts": record["gold_supporting_facts"],
        "gold_supporting_fact_count": record["gold_supporting_fact_count"],
        "evidence_mode": evidence_mode,
    }
    if baseline == "query_only":
        return {**common, "text_a": record["question"], "text_b": None, "packing_audit": None}
    metadata = stage_metadata(record)
    if baseline == "query_stage":
        return {
            **common,
            "text_a": f"{record['question']} {metadata}",
            "text_b": None,
            "packing_audit": None,
        }
    if baseline == "query_stage_stats":
        return {
            **common,
            "text_a": (
                f"{record['question']} {metadata} "
                f"{statistics_text(record['retrieval_statistics'])}"
            ),
            "text_b": None,
            "packing_audit": None,
        }

    items = list(evidence_items_override if evidence_items_override is not None else view["items"])
    packing_chunks = chunk_by_id
    if title_only:
        packing_chunks = {
            item["chunk_id"]: {
                **chunk_by_id[item["chunk_id"]],
                "sentence_ids": [],
                "sentence_texts": [],
            }
            for item in items
        }
    pair_input = baseline == "query_evidence"
    query = record["question"] if pair_input else ""
    if pair_input and include_stage_in_query_evidence:
        query = f"{query} {metadata}"
    evidence_text, packing_audit = pack_evidence(
        tokenizer,
        items,
        packing_chunks,
        question=query,
        max_length=max_length,
        strategy=representation,
        minimum_chunk_tokens=minimum_chunk_tokens,
        pair_input=pair_input,
    )
    return {
        **common,
        "text_a": query if pair_input else evidence_text,
        "text_b": evidence_text if pair_input else None,
        "packing_audit": packing_audit,
        "input_evidence_chunk_ids": [item["chunk_id"] for item in items],
        "input_evidence_titles": [item["document_title"] for item in items],
    }

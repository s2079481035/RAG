"""Run the released SIM-RAG Critic on the frozen HotpotQA retrieval ladder.

This is deliberately an adapted controller, not an end-to-end SIM-RAG
reproduction.  Reject advances to the next frozen retrieval stage instead of
generating a new query.  The final answer is generated with the frozen Phase
3A Qwen prompt, independently of the candidate answer/rationale used by the
Critic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from evidence_utils import supporting_fact_metrics
from generate_phase3a import chat_prompt, fit_prompt, render_chunk
from phase3a_generation_utils import (
    exact_match_score,
    extract_short_answer,
    load_config,
    token_f1_score,
)


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("dense@5", "hybrid@10", "rerank@20")
DECISION_STAGES = STAGES[:2]
CRITIC_INSTRUCTION = (
    "Instruction: Predict if the following answer to the question and context "
    "should be accepted, 1, or rejected, 0, based on the rationale."
)
CANDIDATE_TEMPLATE = """You are a knowledgeable question-answering assistant. Use only the provided context to answer the multihop question. Provide a short candidate answer and a brief multihop rationale. If the context is insufficient, use "cannot determine" as the answer and explain what is missing. Respond only in this format:
Answer: <short answer>
Rationale: <brief rationale>

Context:
{context}

Question: {question}
"""

# These are the frozen server payload hashes recorded by the Phase 3B manifests.
EXPECTED_HASHES = {
    "data/phase2/chunks/sentence_256.jsonl": "324e77ec0d97130516c9ae0ffad4d29053f98f25c28acbc911ae15824a130f87",
    "data/phase2/controller/sentence_256/dev.jsonl": "f50ec6360bf569d538bc576ef7c2712227ecf2420ec6c7e2c5cd2b532dc0cb51",
    "results/phase3b/generation/dev_policy/stage_generations.jsonl": "13a8dbd0c22e35e47913cf3e8f365f5a494545f832a42ceb99e7dcb8001c67ce",
    "experiments/phase3a/lambda_dev/lambda_0.1_seed42/dev_predictions.jsonl": "0088ffc8a088eb8dfc9aa10e83d2c82e2d5601ea2591ba8a6d07e2e3158d8db0",
}
EXPECTED_CRITIC_SHARDS = {
    "model-00001-of-00003.safetensors": "47e51c675b4bfa4a838cf4a706c053e2b0fbbbcb100a706ad6a567a8fd304fd8",
    "model-00002-of-00003.safetensors": "72628625a8400cb33a81e9d6283d197aa945a84d778b80ddf9c7a00e064b2604",
    "model-00003-of-00003.safetensors": "81a1d0a434072d1e7ea2804c67c6df51ddfd42a69be41bed7baab6c4ae1c66a4",
    "model.safetensors.index.json": "bc663873ca2bacd34ec94fbedc6e709255241547fc6707f58f9865a6aca87227",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoner", required=True)
    parser.add_argument("--critic", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, choices=[32], default=32)
    parser.add_argument("--reasoner-gpu-memory-gib", type=int, default=10)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-critic-shards", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-new-questions",
        type=int,
        help="Validation/resume aid. Results remain a prefix of the frozen 32 IDs.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def read_jsonl(path: Path, selected_ids: set[str] | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if selected_ids is None or row.get("question_id") in selected_ids:
                rows.append(row)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl_durable(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def group_stage_rows(rows: list[dict], ids: list[str], label: str) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {qid: {} for qid in ids}
    for row in rows:
        qid, stage = row["question_id"], row["stage"]
        if qid not in grouped:
            continue
        if stage in grouped[qid]:
            raise ValueError(f"Duplicate {label} stage {stage} for {qid}")
        grouped[qid][stage] = row
    for qid in ids:
        if tuple(grouped[qid]) != STAGES and set(grouped[qid]) != set(STAGES):
            raise ValueError(
                f"{label} stages for {qid} are {sorted(grouped[qid])}, expected {STAGES}"
            )
    return grouped


def primary_ours_policy(selected: dict) -> dict:
    primary = selected["primary_report_policy"]
    for row in selected["policies"]:
        if (
            row.get("status") == "selected"
            and row["calibration_method"] == primary["calibration"]
            and row["selection"] == primary["selection"]
            and float(row["risk_target"]) == float(primary["risk_target"])
        ):
            return row
    raise ValueError("Frozen primary Ours policy is missing")


def validate_inputs(args: argparse.Namespace) -> dict:
    paths = {
        "questions": ROOT / "data/phase2/questions.json",
        "chunks": ROOT / "data/phase2/chunks/sentence_256.jsonl",
        "controller": ROOT / "data/phase2/controller/sentence_256/dev.jsonl",
        "generations": ROOT
        / "results/phase3b/generation/dev_policy/stage_generations.jsonl",
        "predictions": ROOT
        / "experiments/phase3a/lambda_dev/lambda_0.1_seed42/dev_predictions.jsonl",
        "calibrated_predictions": ROOT
        / "results/phase3b/calibrated_dev_predictions.jsonl",
        "ids": ROOT / "data/splits/phase3b_dev_policy_ids.txt",
        "protocol": ROOT / "configs/phase3b/protocol.json",
        "generation_config": ROOT / "configs/phase3a/generation.json",
        "selected_thresholds": ROOT / "results/phase3b/selected_thresholds.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    reasoner = Path(args.reasoner)
    critic = Path(args.critic)
    for label, path in [("reasoner", reasoner), ("critic", critic)]:
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"Missing local {label} model: {path}")

    actual_hashes = {}
    for relative, expected in EXPECTED_HASHES.items():
        actual = file_sha256(ROOT / relative)
        actual_hashes[relative] = actual
        if actual != expected:
            raise ValueError(f"Frozen payload hash mismatch: {relative}: {actual} != {expected}")

    critic_hashes = {}
    if args.verify_critic_shards:
        for filename, expected in EXPECTED_CRITIC_SHARDS.items():
            path = critic / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing Critic artifact: {path}")
            actual = file_sha256(path)
            critic_hashes[filename] = actual
            if actual != expected:
                raise ValueError(f"Critic artifact hash mismatch: {filename}")

    ids = [
        line.strip()
        for line in paths["ids"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.n]
    if len(ids) != args.n or len(set(ids)) != args.n:
        raise ValueError("Frozen Smoke32 IDs are incomplete or duplicated")
    selected_ids = set(ids)
    questions = json.loads(paths["questions"].read_text(encoding="utf-8"))
    chunks = {
        row["chunk_id"]: row for row in read_jsonl(paths["chunks"])
    }
    if len(chunks) != 13891:
        raise ValueError(f"Unexpected sentence_256 corpus size: {len(chunks)}")
    source = group_stage_rows(
        read_jsonl(paths["controller"], selected_ids), ids, "controller"
    )
    generations = group_stage_rows(
        read_jsonl(paths["generations"], selected_ids), ids, "generation"
    )
    predictions = group_stage_rows(
        read_jsonl(paths["predictions"], selected_ids), ids, "prediction"
    )
    calibrated = group_stage_rows(
        read_jsonl(paths["calibrated_predictions"], selected_ids), ids, "calibrated prediction"
    )

    for qid in ids:
        question = questions.get(qid)
        if question is None or question.get("split") != "dev":
            raise ValueError(f"Smoke ID is not an internal Dev question: {qid}")
        previous_ids: list[str] = []
        for index, stage in enumerate(STAGES):
            row = source[qid][stage]
            view = row["cumulative_evidence_memory"]
            chunk_ids = list(view["chunk_ids"])
            item_ids = [item["chunk_id"] for item in view["items"]]
            if row["question"] != question["question"] or row["answer"] != question["answer"]:
                raise ValueError(f"Question/gold mismatch for {qid} at {stage}")
            if int(row["stage_index"]) != index or chunk_ids != item_ids:
                raise ValueError(f"Stage index/evidence item mismatch for {qid} at {stage}")
            if len(chunk_ids) != len(set(chunk_ids)) or any(cid not in chunks for cid in chunk_ids):
                raise ValueError(f"Missing/duplicate Chunk ID for {qid} at {stage}")
            if chunk_ids[: len(previous_ids)] != previous_ids:
                raise ValueError(f"Cumulative evidence is not prefix-monotone for {qid} at {stage}")
            previous_ids = chunk_ids

            generated = generations[qid][stage]
            if (
                generated["question"] != question["question"]
                or generated["gold_answer"] != question["answer"]
                or generated["available_evidence_chunk_ids"] != chunk_ids
            ):
                raise ValueError(f"Frozen generation alignment mismatch for {qid} at {stage}")
            used = list(generated["used_evidence_chunk_ids"])
            if used != chunk_ids[: len(used)]:
                raise ValueError(f"Frozen model-visible evidence is not a prefix for {qid} at {stage}")
            for pred_group in (predictions, calibrated):
                pred = pred_group[qid][stage]
                if int(pred["actual_stop_label"]) != int(view["stop_label"]):
                    raise ValueError(f"Critic label mismatch for {qid} at {stage}")

    selected_thresholds = json.loads(paths["selected_thresholds"].read_text(encoding="utf-8"))
    ours_policy = primary_ours_policy(selected_thresholds)
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    generation_config = load_config(paths["generation_config"])
    return {
        "paths": paths,
        "ids": ids,
        "questions": questions,
        "chunks": chunks,
        "source": source,
        "generations": generations,
        "predictions": predictions,
        "calibrated": calibrated,
        "ours_policy": ours_policy,
        "protocol": protocol,
        "generation_config": generation_config,
        "actual_hashes": actual_hashes,
        "critic_hashes": critic_hashes,
    }


def prepare_output(args: argparse.Namespace, data: dict) -> tuple[Path, dict, list[dict]]:
    out = args.output.resolve()
    allowed = (ROOT / "results/paper_v1").resolve()
    if not out.is_relative_to(allowed):
        raise ValueError("Output must be under results/paper_v1")
    manifest_path = out / "manifest.json"
    predictions_path = out / "predictions.jsonl"
    if out.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {out}")
    if args.resume and not manifest_path.is_file():
        raise FileNotFoundError("Resume requires an existing manifest.json")
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "method": "SIM-RAG adapted controller",
        "source_type": "adapted_controller_not_official_end_to_end_sim_rag",
        "status": "PREFLIGHT_VALIDATED" if args.preflight else "RUNNING",
        "split": "internal_hotpotqa_dev_policy",
        "frozen_test_used_for_tuning_or_debugging": False,
        "n": args.n,
        "ids": data["ids"],
        "git_commit": git_commit(),
        "reasoner": args.reasoner,
        "critic": args.critic,
        "critic_architecture": "T5ForConditionalGeneration (released general checkpoint)",
        "critic_checkpoint_revision": "0e1cc45ab595543557837b237020ebcd16aeb357",
        "critic_decision": "decoded output stripped exactly equals '1' => Accept/STOP; every other output => Reject/CONTINUE",
        "blank_critic_output": "preserved as Reject and counted as blank_reject; never coerced to Accept",
        "ladder": list(STAGES),
        "decision_stages": list(DECISION_STAGES),
        "forced_final_stage": "rerank@20",
        "evidence_mode": "cumulative_evidence_memory",
        "candidate_generation": "Qwen answer+rationale generated separately at every reached decision stage",
        "final_answer_generation": "frozen configs/phase3a/generation.json prompt, extraction, model, and deterministic decoding",
        "reject_action_adaptation": "advance to the next fixed retrieval stage; no SIM-RAG query generation",
        "cost_accounting": {
            "candidate_reasoner": "fresh wall-clock and tokens; extra SIM-RAG control cost",
            "critic": "fresh CPU wall-clock and tokens",
            "final_reader": "fresh wall-clock and tokens, separate from candidate generation",
            "retrieval": "frozen per-stage component estimate from source cache; retrieval is not replayed online",
            "online_wall_clock": "candidate reasoner + Critic + final reader only; model loading excluded",
        },
        "ours_matched_policy": {
            "name": "temperature_conservative_risk_10",
            "threshold": float(data["ours_policy"]["threshold"]),
            "source": "frozen calibrated Dev predictions and frozen stage answer cache",
        },
        "training_sample_provenance_risk": "UNKNOWN",
        "source_sha256": data["actual_hashes"],
        "critic_artifact_sha256": data["critic_hashes"] or "not_rehashed_this_invocation",
        "adaptations": [
            "Qwen2.5-7B-Instruct replaces the paper reasoner",
            "frozen sentence_256 Dense@5 -> Hybrid@10 -> Rerank@20 evidence ladder",
            "Reject upgrades the fixed ladder instead of generating a new retrieval query",
            "final answer generation is isolated from the candidate answer/rationale used for control",
        ],
    }
    if args.resume:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ["method", "ids", "git_commit", "reasoner", "critic", "ladder"]:
            if existing_manifest.get(key) != manifest.get(key):
                raise ValueError(f"Resume manifest mismatch: {key}")
        manifest = existing_manifest
        manifest["status"] = "RUNNING"
        manifest["last_resumed_at_unix"] = time.time()
    write_json_atomic(manifest_path, manifest)

    existing_rows = read_jsonl(predictions_path) if predictions_path.is_file() else []
    existing_ids = [row["question_id"] for row in existing_rows]
    if existing_ids != data["ids"][: len(existing_ids)] or len(set(existing_ids)) != len(existing_ids):
        raise ValueError("Existing predictions are not the exact frozen Smoke32 prefix")
    return out, manifest, existing_rows


def parse_candidate(text: str) -> tuple[str, str, bool]:
    answer_match = re.search(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", text)
    rationale_match = re.search(r"(?ims)^\s*Rationale\s*:\s*(.+?)\s*$", text)
    answer = answer_match.group(1).strip() if answer_match else ""
    rationale = rationale_match.group(1).strip() if rationale_match else ""
    return answer, rationale, bool(answer and rationale)


def critic_verdict(decoded: str) -> tuple[bool, str, bool]:
    normalized = decoded.strip()
    if normalized == "1":
        return True, "accept_1", False
    if normalized == "0":
        return False, "reject_0", False
    if normalized == "":
        return False, "blank_reject", False
    return False, "other_reject", True


def retrieval_component_ms(source_row: dict) -> float:
    values = source_row["latency_ms"]
    dense = float(values["query_embedding_run_mean"]) + float(values["dense_search"])
    if source_row["stage"] == "dense@5":
        return dense
    hybrid = dense + float(values["bm25_scoring"]) + float(values["rrf_fusion"])
    if source_row["stage"] == "hybrid@10":
        return hybrid
    return hybrid + float(values["reranker_allocated"])


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return ordered[left] * (1 - fraction) + ordered[right] * fraction


def ours_matched_rows(data: dict) -> list[dict]:
    threshold = float(data["ours_policy"]["threshold"])
    probability_key = f"{data['ours_policy']['calibration_method']}_stop_probability"
    rows = []
    for qid in data["ids"]:
        decisions = []
        final_stage = STAGES[-1]
        for stage in DECISION_STAGES:
            pred = data["calibrated"][qid][stage]
            stop = float(pred[probability_key]) >= threshold
            decisions.append(
                {
                    "stage": stage,
                    "stop_probability": float(pred[probability_key]),
                    "decision": "stop" if stop else "continue",
                }
            )
            if stop:
                final_stage = stage
                break
        source = data["source"][qid][final_stage]
        generated = data["generations"][qid][final_stage]
        view = source["cumulative_evidence_memory"]
        rows.append(
            {
                "question_id": qid,
                "final_stage": final_stage,
                "retrieved_chunks": len(view["items"]),
                "retrieval_calls": STAGES.index(final_stage) + 1,
                "critic_calls": len(decisions),
                "reader_calls": 1,
                "answer_em": int(generated["exact_match"]),
                "answer_f1": float(generated["token_f1"]),
                "supporting_fact_recall": float(generated["supporting_fact_recall"]),
                "complete_evidence_coverage": int(generated["complete_evidence_coverage"]),
                "final_answer": generated["extracted_answer"],
                "reader_input_tokens": int(generated["prompt_tokens"]),
                "reader_output_tokens": int(generated["output_tokens"]),
                "decisions": decisions,
            }
        )
    return rows


def summarize_and_write(out: Path, records: list[dict], data: dict) -> dict:
    ours = ours_matched_rows(data)
    critic_categories = Counter(
        decision["critic_output_category"]
        for row in records
        for decision in row["trajectory"]
    )
    sim = {
        "questions": len(records),
        "answer_em": mean(records, "answer_em"),
        "answer_f1": mean(records, "answer_f1"),
        "supporting_fact_recall": mean(records, "supporting_fact_recall"),
        "complete_evidence_coverage": mean(records, "complete_evidence_coverage"),
        "average_retrieved_chunks": mean(records, "retrieved_chunks"),
        "average_retrieval_calls": mean(records, "retrieval_calls"),
        "average_critic_calls": mean(records, "critic_calls"),
        "average_reader_calls": mean(records, "reader_calls"),
        "average_extra_candidate_reasoner_calls": mean(
            records, "candidate_reasoner_calls"
        ),
        "average_input_tokens": mean(records, "total_input_tokens"),
        "average_output_tokens": mean(records, "total_output_tokens"),
        "average_total_tokens": mean(records, "total_tokens"),
        "average_extra_candidate_tokens": mean(records, "candidate_total_tokens"),
        "average_critic_tokens": mean(records, "critic_total_tokens"),
        "average_final_reader_tokens": mean(records, "reader_total_tokens"),
        "retrieval_component_estimate_mean_ms": mean(
            records, "retrieval_component_estimate_ms"
        ),
        "candidate_reasoner_latency_mean_ms": mean(
            records, "candidate_reasoner_latency_ms"
        ),
        "critic_latency_mean_ms": mean(records, "critic_latency_ms"),
        "final_reader_latency_mean_ms": mean(records, "reader_latency_ms"),
        "online_controller_reader_wall_clock_mean_ms": mean(
            records, "online_controller_reader_wall_clock_ms"
        ),
        "online_controller_reader_wall_clock_median_ms": percentile(
            [float(row["online_controller_reader_wall_clock_ms"]) for row in records], 0.5
        ),
        "online_controller_reader_wall_clock_p95_ms": percentile(
            [float(row["online_controller_reader_wall_clock_ms"]) for row in records], 0.95
        ),
        "stop_stage_counts": dict(Counter(row["final_stage"] for row in records)),
        "critic_output_counts": dict(critic_categories),
        "blank_reject_count": critic_categories.get("blank_reject", 0),
        "invalid_critic_output_count": sum(
            decision["invalid_critic_output"]
            for row in records
            for decision in row["trajectory"]
        ),
        "candidate_parse_failure_count": sum(
            not decision["candidate_parse_valid"]
            for row in records
            for decision in row["trajectory"]
        ),
        "incorrect_stop_count": sum(row["incorrect_stop"] for row in records),
        "final_answer_cache_mismatch_count": sum(
            not row["final_answer_matches_frozen_cache"] for row in records
        ),
        "latency_limitation": (
            "retrieval is a frozen component estimate; fresh wall-clock excludes model loading "
            "and cached retrieval, and the GPU/CPU were shared"
        ),
    }
    ours_summary = {
        "questions": len(ours),
        "answer_em": mean(ours, "answer_em"),
        "answer_f1": mean(ours, "answer_f1"),
        "supporting_fact_recall": mean(ours, "supporting_fact_recall"),
        "complete_evidence_coverage": mean(ours, "complete_evidence_coverage"),
        "average_retrieved_chunks": mean(ours, "retrieved_chunks"),
        "average_retrieval_calls": mean(ours, "retrieval_calls"),
        "average_critic_calls": mean(ours, "critic_calls"),
        "average_reader_calls": mean(ours, "reader_calls"),
        "average_reader_input_tokens": mean(ours, "reader_input_tokens"),
        "average_reader_output_tokens": mean(ours, "reader_output_tokens"),
        "stop_stage_counts": dict(Counter(row["final_stage"] for row in ours)),
        "policy": "temperature_conservative_risk_10",
        "threshold": float(data["ours_policy"]["threshold"]),
    }
    quality_gate = {
        "completed_32_of_32": len(records) == 32,
        "question_and_chunk_alignment_validated": True,
        "official_critic_ran": all(row["critic_calls"] >= 1 for row in records),
        "accept_reject_mapping_validated": all(
            decision["accepted"] == (decision["critic_raw_stripped"] == "1")
            for row in records
            for decision in row["trajectory"]
        ),
        "all_trajectories_traceable": all(row["trajectory"] for row in records),
        "candidate_cost_accounted": all(row["candidate_reasoner_calls"] >= 1 for row in records),
        "invalid_outputs_zero": sim["invalid_critic_output_count"] == 0
        and sim["candidate_parse_failure_count"] == 0,
        "frozen_test_not_used": True,
        "training_sample_provenance_risk": "UNKNOWN",
    }
    summary = {
        "status": "SMOKE_COMPLETE_EXPLORATORY_NOT_NATIVE_SIM_RAG",
        "sim_rag_adapted_controller": sim,
        "ours_matched_risk10": ours_summary,
        "paired_delta_sim_minus_ours": {
            "answer_f1": sim["answer_f1"] - ours_summary["answer_f1"],
            "average_retrieved_chunks": sim["average_retrieved_chunks"]
            - ours_summary["average_retrieved_chunks"],
        },
        "quality_gate": quality_gate,
        "recommend_dev100": False,
        "recommend_dev100_note": (
            "Smoke completion never auto-authorizes Dev100; review quality gates, errors, "
            "cost, and UNKNOWN checkpoint training provenance first."
        ),
    }
    write_json_atomic(out / "summary.json", summary)

    def fmt(value: float) -> str:
        return f"{value:.6f}"

    comparison = f"""# HotpotQA Dev Smoke32: matched-controller comparison

This is an exploratory **SIM-RAG adapted controller** result, not an official end-to-end SIM-RAG reproduction. Both rows use the same 32 internal Dev questions, frozen cumulative stage evidence, Qwen2.5-7B-Instruct final-answer prompt, answer extraction, and EM/F1 evaluator. The SIM-RAG row adds answer+rationale generation before each reached official Critic call.

| Controller | Questions | Answer EM | Answer F1 | SF recall | Complete evidence coverage | Avg chunks | Avg retrieval calls | Avg Critic calls | Avg final Reader calls | Avg extra candidate calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours original Critic, matched Risk10 | {len(ours)} | {fmt(ours_summary['answer_em'])} | {fmt(ours_summary['answer_f1'])} | {fmt(ours_summary['supporting_fact_recall'])} | {fmt(ours_summary['complete_evidence_coverage'])} | {fmt(ours_summary['average_retrieved_chunks'])} | {fmt(ours_summary['average_retrieval_calls'])} | {fmt(ours_summary['average_critic_calls'])} | {fmt(ours_summary['average_reader_calls'])} | 0.000000 |
| SIM-RAG adapted controller | {len(records)} | {fmt(sim['answer_em'])} | {fmt(sim['answer_f1'])} | {fmt(sim['supporting_fact_recall'])} | {fmt(sim['complete_evidence_coverage'])} | {fmt(sim['average_retrieved_chunks'])} | {fmt(sim['average_retrieval_calls'])} | {fmt(sim['average_critic_calls'])} | {fmt(sim['average_reader_calls'])} | {fmt(sim['average_extra_candidate_reasoner_calls'])} |

## SIM-RAG control cost and diagnostics

- Average total tokens: {fmt(sim['average_total_tokens'])}; extra candidate answer+rationale tokens: {fmt(sim['average_extra_candidate_tokens'])}; Critic tokens: {fmt(sim['average_critic_tokens'])}; final Reader tokens: {fmt(sim['average_final_reader_tokens'])}.
- Mean fresh candidate / Critic / final Reader latency: {fmt(sim['candidate_reasoner_latency_mean_ms'])} / {fmt(sim['critic_latency_mean_ms'])} / {fmt(sim['final_reader_latency_mean_ms'])} ms.
- Mean fresh controller+Reader wall-clock (cached retrieval excluded): {fmt(sim['online_controller_reader_wall_clock_mean_ms'])} ms. Frozen retrieval component estimate: {fmt(sim['retrieval_component_estimate_mean_ms'])} ms.
- Stop stages: `{json.dumps(sim['stop_stage_counts'], ensure_ascii=False)}`. Critic outputs: `{json.dumps(sim['critic_output_counts'], ensure_ascii=False)}`.
- Incorrect early stops: {sim['incorrect_stop_count']}; invalid Critic outputs: {sim['invalid_critic_output_count']}; candidate parse failures: {sim['candidate_parse_failure_count']}; final-answer cache mismatches: {sim['final_answer_cache_mismatch_count']}.

## Interpretation limits

- Reject advances the fixed Dense -> Hybrid -> Rerank ladder; it does not invoke native SIM-RAG query generation.
- Empty decoded Critic output remains Reject under the official exact-`1` acceptance rule and is reported separately.
- Retrieval latency is a frozen component estimate. The fresh latency was measured on a shared GPU with a CPU Critic and is diagnostic, not a dedicated full online wall-clock comparison.
- The released general Critic's per-sample training provenance remains **UNKNOWN**. This Smoke32 must not be used to claim superiority over native SIM-RAG or to tune the frozen Test split.
"""
    (out / "comparison_dev32.md").write_text(comparison, encoding="utf-8")
    return summary


def load_models(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(42)
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen reasoner")
    free_bytes, _ = torch.cuda.mem_get_info()
    required = (args.reasoner_gpu_memory_gib + 2) * 2**30
    if free_bytes < required:
        raise RuntimeError(
            f"Insufficient visible-GPU headroom: {free_bytes / 2**30:.2f} GiB free, "
            f"need at least {required / 2**30:.2f} GiB; other jobs are left untouched"
        )
    reasoner_tokenizer = AutoTokenizer.from_pretrained(
        args.reasoner, local_files_only=True
    )
    reasoner_tokenizer.padding_side = "left"
    reasoner = AutoModelForCausalLM.from_pretrained(
        args.reasoner,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: f"{args.reasoner_gpu_memory_gib}GiB", "cpu": "64GiB"},
    ).eval()
    critic_tokenizer = AutoTokenizer.from_pretrained(
        args.critic, local_files_only=True
    )
    critic = AutoModelForSeq2SeqLM.from_pretrained(
        args.critic,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).eval()
    return torch, reasoner_tokenizer, reasoner, critic_tokenizer, critic


def qwen_generate(torch, tokenizer, model, prompt: str, max_new_tokens: int) -> dict:
    rendered = chat_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered, return_tensors="pt", add_special_tokens=False
    ).to(model.get_input_embeddings().weight.device)
    if int(encoded.input_ids.shape[1]) > 4096:
        raise ValueError(f"Qwen prompt exceeds 4096 tokens: {encoded.input_ids.shape[1]}")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    input_tokens = int(encoded.input_ids.shape[1])
    output_ids = output[0, input_tokens:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": int(output_ids.shape[0]),
        "latency_ms": latency_ms,
    }


def critic_generate(torch, tokenizer, model, gate: str) -> dict:
    raw_token_count = len(tokenizer(gate, add_special_tokens=True)["input_ids"])
    encoded = tokenizer(gate, return_tensors="pt", truncation=True)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=1)
    latency_ms = (time.perf_counter() - started) * 1000.0
    raw_decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    accepted, category, invalid = critic_verdict(raw_decoded)
    return {
        "raw_decoded": raw_decoded,
        "stripped": raw_decoded.strip(),
        "accepted": accepted,
        "category": category,
        "invalid": invalid,
        "input_tokens": int(encoded.input_ids.shape[1]),
        "input_tokens_before_truncation": raw_token_count,
        "input_truncated": raw_token_count > int(encoded.input_ids.shape[1]),
        "output_tokens": max(0, int(generated.shape[1]) - 1),
        "latency_ms": latency_ms,
    }


def run_one(
    qid: str,
    data: dict,
    torch,
    reasoner_tokenizer,
    reasoner,
    critic_tokenizer,
    critic,
) -> dict:
    question = data["questions"][qid]
    trajectory = []
    started = time.perf_counter()
    final_stage = STAGES[-1]
    candidate_input = candidate_output = critic_input = critic_output = 0
    candidate_latency = critic_latency = 0.0

    for stage in DECISION_STAGES:
        source = data["source"][qid][stage]
        frozen_generation = data["generations"][qid][stage]
        used_ids = frozen_generation["used_evidence_chunk_ids"]
        context = "\n\n".join(render_chunk(data["chunks"][cid]) for cid in used_ids)
        prompt = CANDIDATE_TEMPLATE.format(
            context=context, question=question["question"]
        )
        candidate = qwen_generate(
            torch, reasoner_tokenizer, reasoner, prompt, max_new_tokens=128
        )
        answer, rationale, parse_valid = parse_candidate(candidate["text"])
        if not parse_valid:
            raise RuntimeError(
                f"Candidate answer/rationale parse failure for {qid} at {stage}: "
                f"{candidate['text'][:300]!r}"
            )
        task = f"Question: {question['question']}\nContext:\n{context}\n"
        gate = (
            f"{CRITIC_INSTRUCTION}\n{task} \nAnswer: {answer}\n"
            f"Rationale: {rationale}"
        )
        verdict = critic_generate(torch, critic_tokenizer, critic, gate)
        decision = {
            "stage": stage,
            "stage_index": STAGES.index(stage) + 1,
            "evidence_chunk_ids": list(
                source["cumulative_evidence_memory"]["chunk_ids"]
            ),
            "model_visible_evidence_chunk_ids": list(used_ids),
            "candidate_raw": candidate["text"],
            "candidate_answer": answer,
            "candidate_rationale": rationale,
            "candidate_parse_valid": parse_valid,
            "candidate_input_tokens": candidate["input_tokens"],
            "candidate_output_tokens": candidate["output_tokens"],
            "candidate_latency_ms": candidate["latency_ms"],
            "critic_raw_decoded": verdict["raw_decoded"],
            "critic_raw_stripped": verdict["stripped"],
            "critic_output_category": verdict["category"],
            "invalid_critic_output": verdict["invalid"],
            "accepted": verdict["accepted"],
            "decision": "stop" if verdict["accepted"] else "continue",
            "critic_input_tokens": verdict["input_tokens"],
            "critic_input_tokens_before_truncation": verdict[
                "input_tokens_before_truncation"
            ],
            "critic_input_truncated": verdict["input_truncated"],
            "critic_output_tokens": verdict["output_tokens"],
            "critic_latency_ms": verdict["latency_ms"],
            "supporting_fact_recall": float(
                source["cumulative_evidence_memory"]["supporting_fact_recall"]
            ),
            "complete_evidence_coverage": int(
                source["cumulative_evidence_memory"]["complete_evidence_coverage"]
            ),
        }
        trajectory.append(decision)
        candidate_input += candidate["input_tokens"]
        candidate_output += candidate["output_tokens"]
        critic_input += verdict["input_tokens"]
        critic_output += verdict["output_tokens"]
        candidate_latency += candidate["latency_ms"]
        critic_latency += verdict["latency_ms"]
        if verdict["invalid"]:
            raise RuntimeError(
                f"Unexpected Critic output for {qid} at {stage}: {verdict['raw_decoded']!r}"
            )
        if verdict["accepted"]:
            final_stage = stage
            break

    source = data["source"][qid][final_stage]
    view = source["cumulative_evidence_memory"]
    evidence_chunks = [data["chunks"][item["chunk_id"]] for item in view["items"]]
    final_prompt, used_final_ids, context_truncated = fit_prompt(
        reasoner_tokenizer,
        data["generation_config"],
        question["question"],
        evidence_chunks,
    )
    frozen_final = data["generations"][qid][final_stage]
    if used_final_ids != frozen_final["used_evidence_chunk_ids"]:
        raise ValueError(f"Final Reader prompt evidence drift for {qid} at {final_stage}")
    reader = qwen_generate(
        torch,
        reasoner_tokenizer,
        reasoner,
        final_prompt,
        max_new_tokens=int(data["generation_config"]["max_new_tokens"]),
    )
    final_answer = extract_short_answer(reader["text"])
    visible_metrics = supporting_fact_metrics(
        used_final_ids, data["chunks"], question["gold_supporting_facts"]
    )
    online_wall_ms = (time.perf_counter() - started) * 1000.0
    retrieved_chunks = len(view["items"])
    candidate_total = candidate_input + candidate_output
    critic_total = critic_input + critic_output
    reader_total = reader["input_tokens"] + reader["output_tokens"]
    incorrect_stop = int(
        final_stage in DECISION_STAGES and not int(view["complete_evidence_coverage"])
    )
    return {
        "question_id": qid,
        "question": question["question"],
        "gold_answer": question["answer"],
        "final_stage": final_stage,
        "final_stage_index": STAGES.index(final_stage) + 1,
        "stop_reason": "critic_accept" if final_stage in DECISION_STAGES else "forced_final_stage",
        "final_evidence_chunk_ids": list(view["chunk_ids"]),
        "final_model_visible_evidence_chunk_ids": used_final_ids,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_calls": STAGES.index(final_stage) + 1,
        "critic_calls": len(trajectory),
        "reader_calls": 1,
        "candidate_reasoner_calls": len(trajectory),
        "final_raw_output": reader["text"],
        "final_answer": final_answer,
        "frozen_cached_final_answer": frozen_final["extracted_answer"],
        "final_answer_matches_frozen_cache": final_answer
        == frozen_final["extracted_answer"],
        "answer_em": exact_match_score(final_answer, question["answer"]),
        "answer_f1": token_f1_score(final_answer, question["answer"]),
        "supporting_fact_recall": float(view["supporting_fact_recall"]),
        "complete_evidence_coverage": int(view["complete_evidence_coverage"]),
        "model_visible_supporting_fact_recall": float(
            visible_metrics["supporting_fact_recall"]
        ),
        "model_visible_complete_evidence_coverage": int(
            visible_metrics["complete_evidence_coverage"]
        ),
        "incorrect_stop": incorrect_stop,
        "candidate_input_tokens": candidate_input,
        "candidate_output_tokens": candidate_output,
        "candidate_total_tokens": candidate_total,
        "critic_input_tokens": critic_input,
        "critic_output_tokens": critic_output,
        "critic_total_tokens": critic_total,
        "reader_input_tokens": reader["input_tokens"],
        "reader_output_tokens": reader["output_tokens"],
        "reader_total_tokens": reader_total,
        "total_input_tokens": candidate_input + critic_input + reader["input_tokens"],
        "total_output_tokens": candidate_output
        + critic_output
        + reader["output_tokens"],
        "total_tokens": candidate_total + critic_total + reader_total,
        "candidate_reasoner_latency_ms": candidate_latency,
        "critic_latency_ms": critic_latency,
        "reader_latency_ms": reader["latency_ms"],
        "retrieval_component_estimate_ms": retrieval_component_ms(source),
        "online_controller_reader_wall_clock_ms": online_wall_ms,
        "context_truncated": context_truncated,
        "trajectory": trajectory,
    }


def run() -> None:
    args = parse_args()
    if args.max_new_questions is not None and args.max_new_questions <= 0:
        raise ValueError("--max-new-questions must be positive")
    data = validate_inputs(args)
    out, manifest, records = prepare_output(args, data)
    print(
        f"INPUT_VALIDATION_OK questions={len(data['ids'])} chunks={len(data['chunks'])} "
        f"git={manifest['git_commit']}",
        flush=True,
    )
    if args.preflight:
        print("PREFLIGHT_OK; no model inference", flush=True)
        return
    remaining = data["ids"][len(records) :]
    if args.max_new_questions is not None:
        remaining = remaining[: args.max_new_questions]
    if remaining:
        torch, rtok, reasoner, ctok, critic = load_models(args)
        print(
            f"MODELS_LOADED visible_cuda_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"resume_from={len(records)}",
            flush=True,
        )
        for qid in remaining:
            try:
                row = run_one(qid, data, torch, rtok, reasoner, ctok, critic)
            except Exception as error:
                append_jsonl_durable(
                    out / "errors.jsonl",
                    {
                        "question_id": qid,
                        "completed_before_failure": len(records),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "failed_at_unix": time.time(),
                    },
                )
                write_json_atomic(
                    out / "progress.json",
                    {
                        "status": "FAILED",
                        "completed": len(records),
                        "expected": args.n,
                        "failed_question_id": qid,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "last_updated_unix": time.time(),
                    },
                )
                manifest["status"] = "FAILED"
                manifest["failure"] = {
                    "question_id": qid,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                write_json_atomic(out / "manifest.json", manifest)
                raise
            append_jsonl_durable(out / "predictions.jsonl", row)
            records.append(row)
            write_json_atomic(
                out / "progress.json",
                {
                    "status": "RUNNING" if len(records) < args.n else "INFERENCE_COMPLETE",
                    "completed": len(records),
                    "expected": args.n,
                    "last_question_id": qid,
                    "last_final_stage": row["final_stage"],
                    "last_updated_unix": time.time(),
                },
            )
            print(
                f"{len(records)}/{args.n} completed qid={qid} "
                f"stage={row['final_stage']} f1={row['answer_f1']:.4f} "
                f"critic_calls={row['critic_calls']}",
                flush=True,
            )
    if len(records) == args.n:
        summary = summarize_and_write(out, records, data)
        manifest["status"] = "SMOKE_COMPLETE"
        manifest["predictions_sha256"] = file_sha256(out / "predictions.jsonl")
        manifest["summary_sha256"] = file_sha256(out / "summary.json")
        manifest["comparison_sha256"] = file_sha256(out / "comparison_dev32.md")
        manifest["completed_at_unix"] = time.time()
        write_json_atomic(out / "manifest.json", manifest)
        write_json_atomic(
            out / "progress.json",
            {
                "status": "SMOKE_COMPLETE",
                "completed": len(records),
                "expected": args.n,
                "last_question_id": records[-1]["question_id"],
                "last_updated_unix": time.time(),
            },
        )
        print(
            "SMOKE_COMPLETE "
            + json.dumps(summary["sim_rag_adapted_controller"], ensure_ascii=False),
            flush=True,
        )
    else:
        print(
            f"VALIDATED_PREFIX_COMPLETE completed={len(records)}/{args.n}; resume required",
            flush=True,
        )


if __name__ == "__main__":
    run()

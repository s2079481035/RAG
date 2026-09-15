"""Produce the frozen Phase 4 heldout tables, figures, bootstrap, and reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from phase3a_metrics import metrics_from_decisions
from phase3b_metrics import temperature_scale
from phase4_adaptive_router import ROUTE_NAMES, file_sha256, stage_latency_ms
from phase4_heldout_guard import ROOT, guard_heldout_access, load_json, repo_path
from phase4_score_threshold import attach_normalized_scores
from train_phase2_controller import read_jsonl, write_jsonl_atomic


LADDER = ["dense@5", "hybrid@10", "rerank@20"]
ACTIONABLE = LADDER[:-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="Recompute derived summaries after an interrupted analysis",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    order = {stage: index for index, stage in enumerate(LADDER)}
    output = defaultdict(list)
    for row in rows:
        output[row["question_id"]].append(row)
    for question_id, values in output.items():
        values.sort(key=lambda row: order[row["stage"]])
        if [row["stage"] for row in values] != LADDER:
            raise ValueError(f"Incomplete trajectory: {question_id}")
    return dict(output)


def enrich_source(source_rows: list[dict], prediction_rows: list[dict], temperature: float) -> dict:
    prediction = {(row["question_id"], row["stage"]): row for row in prediction_rows}
    epsilon = 1e-7
    output = []
    for source in source_rows:
        key = (source["question_id"], source["stage"])
        if key not in prediction:
            raise ValueError(f"Missing heldout Controller prediction: {key}")
        view = source["cumulative_evidence_memory"]
        raw = float(prediction[key]["stop_probability"])
        calibrated = float(temperature_scale([raw], temperature, epsilon)[0])
        output.append(
            {
                "question_id": source["question_id"],
                "stage": source["stage"],
                "stage_index": LADDER.index(source["stage"]) + 1,
                "actual_stop_label": int(view["stop_label"]),
                "actual_three_class_label": view["evidence_state"],
                "true_coverage": float(view["supporting_fact_recall"]),
                "raw_stop_probability": raw,
                "temperature_stop_probability": calibrated,
                "retrieved_chunks": len(view["items"]),
                "unique_titles": len(set(view["document_titles"])),
                "reranker_calls": int(source["stage"] == LADDER[-1]),
                "retrieval_latency_ms": stage_latency_ms(source),
                "question_type": source.get("question_type"),
                "gold_supporting_fact_count": int(source["gold_supporting_fact_count"]),
            }
        )
    return grouped(output)


def later_recoverable(rows: list[dict], index: int) -> bool:
    return not rows[index]["actual_stop_label"] and any(
        row["actual_stop_label"] for row in rows[index + 1 :]
    )


def select_threshold_policy(
    rows: list[dict], probability_key: str, threshold: float, name: str
) -> dict:
    decisions = []
    selected_index = 2
    false_stop_recoverable = False
    false_stop_unrecoverable = False
    for index, row in enumerate(rows[:2]):
        stop = int(float(row[probability_key]) >= threshold)
        recoverable = later_recoverable(rows, index)
        decisions.append(
            {
                "stage": row["stage"],
                "actual_stop_label": row["actual_stop_label"],
                "actual_three_class_label": row["actual_three_class_label"],
                "true_coverage": row["true_coverage"],
                "stop_probability": float(row[probability_key]),
                "predicted_stop_label": stop,
                "later_recoverable": int(recoverable),
            }
        )
        if stop:
            selected_index = index
            false_stop_recoverable = bool(not row["actual_stop_label"] and recoverable)
            false_stop_unrecoverable = bool(not row["actual_stop_label"] and not recoverable)
            break
    selected = rows[selected_index]
    return {
        "question_id": selected["question_id"],
        "policy": name,
        "final_stage": selected["stage"],
        "final_stage_index": selected_index + 1,
        "decisions": decisions,
        "controller_calls": len(decisions),
        "false_stop_recoverable": int(false_stop_recoverable),
        "false_stop_unrecoverable": int(false_stop_unrecoverable),
    }


def select_hard_policy(source: list[dict], decisions: dict, name: str) -> dict:
    selected_index = 2
    reached = []
    false_stop_recoverable = False
    false_stop_unrecoverable = False
    for index, row in enumerate(source[:2]):
        decision = decisions[(row["question_id"], row["stage"])]
        stop = int(decision["predicted_stop_label"])
        recoverable = later_recoverable(source, index)
        reached.append(
            {
                "stage": row["stage"],
                "actual_stop_label": row["actual_stop_label"],
                "actual_three_class_label": row["actual_three_class_label"],
                "true_coverage": row["true_coverage"],
                "predicted_stop_label": stop,
                "later_recoverable": int(recoverable),
            }
        )
        if stop:
            selected_index = index
            false_stop_recoverable = bool(not row["actual_stop_label"] and recoverable)
            false_stop_unrecoverable = bool(not row["actual_stop_label"] and not recoverable)
            break
    return {
        "question_id": source[0]["question_id"],
        "policy": name,
        "final_stage": source[selected_index]["stage"],
        "final_stage_index": selected_index + 1,
        "decisions": reached,
        "controller_calls": len(reached),
        "false_stop_recoverable": int(false_stop_recoverable),
        "false_stop_unrecoverable": int(false_stop_unrecoverable),
    }


def select_fixed(rows: list[dict], index: int, name: str) -> dict:
    return {
        "question_id": rows[0]["question_id"],
        "policy": name,
        "final_stage": rows[index]["stage"],
        "final_stage_index": index + 1,
        "decisions": [],
        "controller_calls": 0,
        "false_stop_recoverable": 0,
        "false_stop_unrecoverable": 0,
    }


def attach_outcome(
    trajectory: dict,
    source: list[dict],
    generation: dict,
    decision_latency_ms: float,
    route_latency_ms: float = 0.0,
    judge_cost: dict | None = None,
) -> dict:
    selected = source[trajectory["final_stage_index"] - 1]
    generated = generation[(selected["question_id"], selected["stage"])]
    controller_latency = (
        float(judge_cost["latency_ms"])
        if judge_cost is not None
        else trajectory["controller_calls"] * decision_latency_ms + route_latency_ms
    )
    return {
        **trajectory,
        "retrieved_chunks": selected["retrieved_chunks"],
        "unique_titles": selected["unique_titles"],
        "reranker_calls": selected["reranker_calls"],
        "retrieval_latency_ms": selected["retrieval_latency_ms"],
        "controller_latency_ms": controller_latency,
        "generation_latency_ms": float(generated["generation_latency_ms"]),
        "total_latency_ms": selected["retrieval_latency_ms"]
        + controller_latency
        + float(generated["generation_latency_ms"]),
        "answer_exact_match": int(generated["exact_match"]),
        "answer_f1": float(generated["token_f1"]),
        "supporting_fact_recall": float(generated["supporting_fact_recall"]),
        "complete_evidence_coverage": int(generated["complete_evidence_coverage"]),
        "model_visible_supporting_fact_recall": float(
            generated["model_visible_supporting_fact_recall"]
        ),
        "model_visible_complete_evidence_coverage": int(
            generated["model_visible_complete_evidence_coverage"]
        ),
        "llm_input_tokens": int(generated["prompt_tokens"]),
        "llm_output_tokens": int(generated["output_tokens"]),
        "llm_total_tokens": int(generated["prompt_tokens"])
        + int(generated["output_tokens"]),
        "question_type": selected["question_type"],
        "gold_supporting_fact_count": selected["gold_supporting_fact_count"],
    }


def decision_metrics(trajectories: list[dict], probability_available: bool) -> dict:
    reached = [decision for row in trajectories for decision in row["decisions"]]
    if not reached:
        return {
            key: None for key in [
                "accuracy", "macro_f1", "stop_precision", "stop_recall", "stop_f1",
                "auroc", "false_stop_rate", "unnecessary_escalation_rate",
                "hard_partial_false_stop_rate", "recoverable_false_stop_rate",
                "unrecoverable_early_stop_rate",
            ]
        }
    labels = np.asarray([row["actual_stop_label"] for row in reached], dtype=int)
    decisions = np.asarray([row["predicted_stop_label"] for row in reached], dtype=int)
    probabilities = np.asarray(
        [row.get("stop_probability", row["predicted_stop_label"]) for row in reached],
        dtype=float,
    )
    metrics = metrics_from_decisions(labels, decisions, probabilities)
    if not probability_available:
        metrics["auroc"] = None
    hard = np.asarray(
        [
            row["actual_stop_label"] == 0 and 0.5 <= float(row["true_coverage"]) < 1.0
            for row in reached
        ],
        dtype=bool,
    )
    recoverable = np.asarray(
        [row["actual_stop_label"] == 0 and row["later_recoverable"] for row in reached],
        dtype=bool,
    )
    unrecoverable = np.asarray(
        [row["actual_stop_label"] == 0 and not row["later_recoverable"] for row in reached],
        dtype=bool,
    )
    metrics["hard_partial_false_stop_rate"] = (
        float(np.mean(decisions[hard] == 1)) if hard.any() else None
    )
    metrics["recoverable_false_stop_rate"] = (
        float(np.mean(decisions[recoverable] == 1)) if recoverable.any() else None
    )
    metrics["unrecoverable_early_stop_rate"] = (
        float(np.mean(decisions[unrecoverable] == 1)) if unrecoverable.any() else None
    )
    metrics["reached_actionable_decisions"] = len(reached)
    return metrics


def summarize(name: str, trajectories: list[dict], probability_available: bool) -> dict:
    metrics = decision_metrics(trajectories, probability_available)
    count = len(trajectories)
    return {
        "system": name,
        "questions": count,
        **metrics,
        "answer_em": sum(row["answer_exact_match"] for row in trajectories) / count,
        "answer_f1": sum(row["answer_f1"] for row in trajectories) / count,
        "supporting_fact_recall": sum(row["supporting_fact_recall"] for row in trajectories) / count,
        "complete_evidence_coverage": sum(row["complete_evidence_coverage"] for row in trajectories) / count,
        "model_visible_supporting_fact_recall": sum(row["model_visible_supporting_fact_recall"] for row in trajectories) / count,
        "model_visible_complete_evidence_coverage": sum(row["model_visible_complete_evidence_coverage"] for row in trajectories) / count,
        "average_final_stage": sum(row["final_stage_index"] for row in trajectories) / count,
        "average_retrieved_chunks": sum(row["retrieved_chunks"] for row in trajectories) / count,
        "average_unique_titles": sum(row["unique_titles"] for row in trajectories) / count,
        "average_reranker_calls": sum(row["reranker_calls"] for row in trajectories) / count,
        "average_controller_calls": sum(row["controller_calls"] for row in trajectories) / count,
        "average_llm_input_tokens": sum(row["llm_input_tokens"] for row in trajectories) / count,
        "average_llm_output_tokens": sum(row["llm_output_tokens"] for row in trajectories) / count,
        "average_llm_total_tokens": sum(row["llm_total_tokens"] for row in trajectories) / count,
        "retrieval_latency_mean_ms": sum(row["retrieval_latency_ms"] for row in trajectories) / count,
        "controller_latency_mean_ms": sum(row["controller_latency_ms"] for row in trajectories) / count,
        "generation_latency_mean_ms": sum(row["generation_latency_ms"] for row in trajectories) / count,
        "total_latency_mean_ms": sum(row["total_latency_ms"] for row in trajectories) / count,
        "early_stop_rate": sum(row["final_stage_index"] < 3 for row in trajectories) / count,
    }


def retrieval_tables(source: dict[str, list[dict]]) -> tuple[list[dict], list[dict], dict]:
    questions = len(source)
    stage_rows = []
    for index, stage in enumerate(LADDER):
        rows = [values[index] for values in source.values()]
        stage_rows.append(
            {
                "row_type": "stage",
                "stage": stage,
                "gold_fact_bucket": "all",
                "questions": questions,
                "supporting_fact_recall": np.mean([row["true_coverage"] for row in rows]),
                "complete_evidence_coverage": np.mean([row["actual_stop_label"] for row in rows]),
                "terminal_retrieval_failure_rate": (
                    1 - np.mean([row["actual_stop_label"] for row in rows]) if index == 2 else None
                ),
                "average_retrieved_chunks": np.mean([row["retrieved_chunks"] for row in rows]),
                "average_unique_titles": np.mean([row["unique_titles"] for row in rows]),
                "average_retrieval_latency_ms": np.mean([row["retrieval_latency_ms"] for row in rows]),
            }
        )
    bucket_rows = []
    for label, predicate in [
        ("2", lambda value: value == 2),
        ("3", lambda value: value == 3),
        ("4+", lambda value: value >= 4),
    ]:
        values = [rows for rows in source.values() if predicate(rows[0]["gold_supporting_fact_count"])]
        for index, stage in enumerate(LADDER):
            bucket_rows.append(
                {
                    "row_type": "gold_fact_bucket",
                    "stage": stage,
                    "gold_fact_bucket": label,
                    "questions": len(values),
                    "supporting_fact_recall": np.mean([rows[index]["true_coverage"] for rows in values]) if values else None,
                    "complete_evidence_coverage": np.mean([rows[index]["actual_stop_label"] for rows in values]) if values else None,
                }
            )
    rescue_rows = []
    for name, start, end in [
        ("Dense_to_Hybrid_Rescue", 0, 1),
        ("Dense_to_Final_Rescue", 0, 2),
        ("Hybrid_to_Final_Rescue", 1, 2),
    ]:
        eligible = [rows for rows in source.values() if not rows[start]["actual_stop_label"]]
        rescued = [rows for rows in eligible if rows[end]["actual_stop_label"]]
        extra_chunks = sum(rows[end]["retrieved_chunks"] - rows[start]["retrieved_chunks"] for rows in eligible)
        extra_latency = sum(rows[end]["retrieval_latency_ms"] - rows[start]["retrieval_latency_ms"] for rows in eligible)
        rescue_rows.append(
            {
                "analysis": name,
                "eligible_current_incomplete": len(eligible),
                "rescued": len(rescued),
                "rescue_rate": len(rescued) / len(eligible) if eligible else None,
                "complete_coverage_gain": len(rescued) / questions,
                "extra_chunks_per_rescued_question": extra_chunks / len(rescued) if rescued else None,
                "extra_latency_ms_per_rescued_question": extra_latency / len(rescued) if rescued else None,
            }
        )
    oracle = []
    for rows in source.values():
        selected = next((row for row in rows if row["actual_stop_label"]), rows[-1])
        oracle.append(selected)
    ceiling = {
        "questions": questions,
        "final_complete_evidence_coverage": float(stage_rows[-1]["complete_evidence_coverage"]),
        "terminal_retrieval_failure_rate": float(stage_rows[-1]["terminal_retrieval_failure_rate"]),
        "oracle_average_chunks": float(np.mean([row["retrieved_chunks"] for row in oracle])),
        "oracle_average_reranker_calls": float(np.mean([row["reranker_calls"] for row in oracle])),
        "oracle_average_retrieval_latency_ms": float(np.mean([row["retrieval_latency_ms"] for row in oracle])),
    }
    return stage_rows + bucket_rows, rescue_rows, ceiling


def bootstrap(
    trajectories: dict[str, list[dict]],
    pairs: list[tuple[str, str]],
    replicates: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    names = sorted(trajectories)
    qids = [row["question_id"] for row in trajectories[names[0]]]
    indexes = {name: {row["question_id"]: row for row in rows} for name, rows in trajectories.items()}
    if any(set(indexes[name]) != set(qids) for name in names):
        raise ValueError("Bootstrap systems do not cover identical question IDs")
    arrays = {}
    for name in names:
        ordered = [indexes[name][qid] for qid in qids]
        arrays[name] = {
            "answer_f1": np.asarray([row["answer_f1"] for row in ordered], dtype=float),
            "answer_em": np.asarray([row["answer_exact_match"] for row in ordered], dtype=float),
            "chunks": np.asarray([row["retrieved_chunks"] for row in ordered], dtype=float),
            "reranker": np.asarray([row["reranker_calls"] for row in ordered], dtype=float),
            "latency": np.asarray([row["total_latency_ms"] for row in ordered], dtype=float),
            "coverage": np.asarray([row["complete_evidence_coverage"] for row in ordered], dtype=float),
            "false_stops": np.asarray([
                sum(
                    decision["actual_stop_label"] == 0 and decision["predicted_stop_label"] == 1
                    for decision in row["decisions"]
                ) for row in ordered
            ], dtype=float),
            "continue": np.asarray([
                sum(decision["actual_stop_label"] == 0 for decision in row["decisions"])
                for row in ordered
            ], dtype=float),
        }
    replicates_rows = []
    pair_values = defaultdict(list)
    fsr_values = defaultdict(list)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(qids), size=len(qids))
        for name in names:
            denominator = arrays[name]["continue"][sampled].sum()
            if denominator:
                fsr_values[name].append(arrays[name]["false_stops"][sampled].sum() / denominator)
        for baseline, model in pairs:
            for metric in ["answer_f1", "answer_em", "chunks", "reranker", "latency", "coverage"]:
                delta = float(
                    arrays[model][metric][sampled].mean()
                    - arrays[baseline][metric][sampled].mean()
                )
                pair_values[(baseline, model, metric)].append(delta)
                replicates_rows.append(
                    {
                        "replicate": replicate,
                        "baseline": baseline,
                        "model": model,
                        "metric": metric,
                        "delta": delta,
                    }
                )
    summary = []
    for (baseline, model, metric), values in pair_values.items():
        observed = float(arrays[model][metric].mean() - arrays[baseline][metric].mean())
        summary.append(
            {
                "comparison": f"{model}_minus_{baseline}",
                "metric": metric,
                "observed_delta": observed,
                "bootstrap_mean_delta": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "replicates": replicates,
                "sampling_unit": "question_id",
            }
        )
    for name, values in fsr_values.items():
        summary.append(
            {
                "comparison": name,
                "metric": "false_stop_rate",
                "observed_delta": float(arrays[name]["false_stops"].sum() / arrays[name]["continue"].sum()),
                "bootstrap_mean_delta": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "replicates": replicates,
                "sampling_unit": "question_id",
            }
        )
    return summary, replicates_rows


def failure_groups(risk10: list[dict]) -> list[dict]:
    definitions = [
        ("A_complete_answer_correct", lambda row: row["complete_evidence_coverage"] and row["answer_exact_match"]),
        ("B_complete_answer_wrong", lambda row: row["complete_evidence_coverage"] and not row["answer_exact_match"]),
        ("C_incomplete_answer_correct", lambda row: not row["complete_evidence_coverage"] and row["answer_exact_match"]),
        ("D_incomplete_answer_wrong", lambda row: not row["complete_evidence_coverage"] and not row["answer_exact_match"]),
        ("recoverable_false_stop", lambda row: row["false_stop_recoverable"]),
        ("unrecoverable_false_stop", lambda row: row["false_stop_unrecoverable"]),
    ]
    output = []
    for name, predicate in definitions:
        rows = [row for row in risk10 if predicate(row)]
        output.append(
            {
                "group": name,
                "questions": len(rows),
                "question_rate": len(rows) / len(risk10),
                "answer_em": np.mean([row["answer_exact_match"] for row in rows]) if rows else None,
                "answer_f1": np.mean([row["answer_f1"] for row in rows]) if rows else None,
            }
        )
    return output


def mean_std_rows(rows: list[dict], name: str) -> list[dict]:
    numeric = [
        key for key, value in rows[0].items()
        if key not in {"system", "seed", "role"} and isinstance(value, (int, float)) and value is not None
    ]
    output = []
    for role, function in [("mean", statistics.fmean), ("std", statistics.stdev)]:
        item = {"system": name, "seed": role, "role": f"three_seed_{role}"}
        for key in numeric:
            values = [float(row[key]) for row in rows]
            item[key] = function(values)
        output.append(item)
    return output


def save_figures(end_rows: list[dict], controller_rows: list[dict], risk_rows: list[dict], ceiling: dict) -> None:
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "figures" / "phase4"
    figure_dir.mkdir(parents=True, exist_ok=True)
    risk = sorted(risk_rows, key=lambda row: float(row["false_stop_rate"]))
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.plot([row["false_stop_rate"] for row in risk], [row["average_retrieved_chunks"] for row in risk], marker="o")
    axis.set(xlabel="False Stop Rate", ylabel="Average retrieved chunks")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "2wiki_risk_efficiency_curve.png", dpi=180)
    plt.close(figure)

    hotpot_path = ROOT / "results" / "phase3b" / "test_risk_policy.csv"
    with hotpot_path.open(encoding="utf-8", newline="") as handle:
        hotpot = [row for row in csv.DictReader(handle) if row["policy"] in {"fixed_heavy", "temperature_conservative_risk_10", "temperature_conservative_risk_20", "temperature_conservative_risk_05"}]
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.plot([float(row["false_stop_rate"]) for row in hotpot], [float(row["average_retrieved_chunks"]) for row in hotpot], marker="o", label="HotpotQA")
    axis.plot([row["false_stop_rate"] for row in risk], [row["average_retrieved_chunks"] for row in risk], marker="s", label="2Wiki")
    axis.set(xlabel="False Stop Rate", ylabel="Average retrieved chunks")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "hotpot_2wiki_risk_efficiency_comparison.png", dpi=180)
    plt.close(figure)

    selected = [row for row in controller_rows if row["system"] in {"EvidenceAwareClassification", "EvidenceAwareRisk10", "Judge-512", "Judge-Full"}]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    labels = [row["system"] for row in selected]
    positions = np.arange(len(labels))
    axis.bar(positions - 0.18, [row["macro_f1"] for row in selected], width=0.36, label="Macro F1")
    axis.bar(positions + 0.18, [row["false_stop_rate"] for row in selected], width=0.36, label="FSR")
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "controller_baseline_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.bar(["Final complete", "Terminal failure"], [ceiling["final_complete_evidence_coverage"], ceiling["terminal_retrieval_failure_rate"]], color=["#2a9d8f", "#e76f51"])
    axis.set(ylabel="Question proportion", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "recoverability_breakdown.png", dpi=180)
    plt.close(figure)


def fmt(value, digits: int = 4) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def report_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    return [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(columns))) + "|",
        *[
            "| " + " | ".join(fmt(row.get(key)) for key, _ in columns) + " |"
            for row in rows
        ],
    ]


def main() -> None:
    args = parse_args()
    heldout_config, _ = guard_heldout_access()
    final_dir = ROOT / "results" / "phase4" / "final"
    heldout_dir = ROOT / "results" / "phase4" / "2wiki" / "heldout"
    targets = [
        final_dir / name for name in [
            "table_retrieval.csv", "table_controller.csv", "table_end_to_end.csv",
            "table_external_baselines.csv", "table_risk_boundary.csv",
        ]
    ]
    targets.extend(
        ROOT / "docs" / "phase4" / name for name in [
            "2wiki_heldout_analysis.md", "cross_dataset_analysis.md", "phase4_closing_report.md"
        ]
    )
    existing = [path for path in targets if path.exists()]
    if existing and not args.resume_incomplete:
        raise FileExistsError(f"Refusing to overwrite final heldout summaries: {existing}")
    if existing:
        print(
            "[resume] recomputing derived heldout summaries after interrupted analysis"
        )

    protocol = load_json(ROOT / "configs" / "phase4" / "protocol.json")
    source_path = ROOT / "data" / "2wiki" / "controller" / "sentence_256" / "heldout.jsonl"
    source_rows = read_jsonl(source_path)
    prediction_path = ROOT / "experiments" / "phase4" / "2wiki" / "final_evidence_aware" / "seed42" / "evaluation" / "heldout" / "original_predictions.jsonl"
    prediction_rows = read_jsonl(prediction_path)
    temperature_data = load_json(ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "temperature_scaling.json")
    source = enrich_source(source_rows, prediction_rows, float(temperature_data["temperature"]))
    if len(source) != int(heldout_config["expected_questions"]):
        raise ValueError("Heldout question count differs from the frozen protocol")
    generation_path = repo_path(heldout_config["generation"]["stage_cache"])
    generation = {
        (row["question_id"], row["stage"]): row for row in read_jsonl(generation_path)
    }
    expected_keys = {(qid, stage) for qid in source for stage in LADDER}
    if set(generation) != expected_keys:
        raise ValueError("Heldout generation cache and trajectories differ")

    controller_latency = float(load_json(ROOT / "results" / "phase4" / "2wiki" / "llm_judge" / "dev_policy" / "critic_latency_manifest.json")["latency"]["mean_ms"])
    router_latency = float(load_json(ROOT / "results" / "phase4" / "2wiki" / "adaptive_rag" / "router_latency_manifest.json")["latency"]["mean_ms"])
    selected = load_json(ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "selected_thresholds.json")
    selected_temperature = {
        (float(row["risk_target"]), row["selection"]): row
        for row in selected["policies"] if row["calibration_method"] == "temperature"
    }
    classification_threshold = float(protocol["in_domain_policy"]["frozen_dev_gate"]["classification_operating_point"]["threshold"])
    definitions = {
        "fixed_heavy": ("fixed", 2, None),
        "EvidenceAwareClassification": ("threshold", "raw_stop_probability", classification_threshold),
    }
    for target in (0.2, 0.1, 0.05):
        item = selected_temperature[(target, "conservative")]
        definitions[f"temperature_conservative_risk_{int(target * 100):02d}"] = (
            "threshold", "temperature_stop_probability", float(item["threshold"])
        )

    trajectories = {}
    for name, definition in definitions.items():
        rows = []
        for question_id in sorted(source):
            source_trajectory = source[question_id]
            if definition[0] == "fixed":
                trajectory = select_fixed(source_trajectory, int(definition[1]), name)
            else:
                trajectory = select_threshold_policy(
                    source_trajectory, str(definition[1]), float(definition[2]), name
                )
            rows.append(
                attach_outcome(trajectory, source_trajectory, generation, controller_latency)
            )
        trajectories[name] = rows

    router_seed_rows = []
    router_trajectories = {}
    for seed in heldout_config["adaptive_router_seeds"]:
        path = ROOT / "experiments" / "phase4" / "2wiki" / "adaptive_rag_router" / f"seed{seed}" / "evaluation" / "heldout" / "predictions.jsonl"
        predictions = {row["question_id"]: int(row["predicted_route_label"]) for row in read_jsonl(path)}
        evaluation_manifest = load_json(path.with_name("evaluation_manifest.json"))
        name = f"AdaptiveRouter_seed{seed}"
        rows = []
        for question_id in sorted(source):
            trajectory = select_fixed(source[question_id], predictions[question_id], name)
            rows.append(
                attach_outcome(
                    trajectory, source[question_id], generation, controller_latency,
                    route_latency_ms=router_latency,
                )
            )
        router_trajectories[name] = rows
        summary = summarize(name, rows, False)
        route_counts = Counter(row["final_stage_index"] for row in rows)
        summary.update(
            {
                "seed": seed,
                "role": "three_seed_router",
                "route_accuracy": float(evaluation_manifest["metrics"]["accuracy"]),
                "route_macro_f1": float(evaluation_manifest["metrics"]["macro_f1"]),
                "light_route_rate": route_counts[1] / len(rows),
                "medium_route_rate": route_counts[2] / len(rows),
                "heavy_route_rate": route_counts[3] / len(rows),
                "recoverable_route_accuracy": np.mean([
                    int(row["final_stage_index"] == next(
                        (
                            index + 1 for index, stage_row in enumerate(source[row["question_id"]])
                            if stage_row["actual_stop_label"]
                        ),
                        3,
                    ))
                    for row in rows
                    if any(stage_row["actual_stop_label"] for stage_row in source[row["question_id"]])
                ]),
                "recoverable_under_retrieval_rate": np.mean([
                    int(row["final_stage_index"] < next(
                        index + 1 for index, stage_row in enumerate(source[row["question_id"]])
                        if stage_row["actual_stop_label"]
                    ))
                    for row in rows
                    if any(stage_row["actual_stop_label"] for stage_row in source[row["question_id"]])
                ]),
            }
        )
        router_seed_rows.append(summary)
    trajectories.update(router_trajectories)

    judge_rows = []
    judge_trajectories = {}
    for representation, display in [("judge_512", "Judge-512"), ("judge_full", "Judge-Full")]:
        path = ROOT / "results" / "phase4" / "2wiki" / "llm_judge" / "heldout" / f"{representation}.jsonl"
        values = read_jsonl(path)
        decisions = {(row["question_id"], row["stage"]): row for row in values}
        rows = []
        for question_id in sorted(source):
            trajectory = select_hard_policy(source[question_id], decisions, display)
            reached_keys = [(question_id, row["stage"]) for row in trajectory["decisions"]]
            cost = {
                "latency_ms": sum(float(decisions[key]["judge_latency_ms"]) for key in reached_keys),
                "input_tokens": sum(int(decisions[key]["input_tokens"]) for key in reached_keys),
                "output_tokens": sum(int(decisions[key]["output_tokens"]) for key in reached_keys),
            }
            outcome = attach_outcome(
                trajectory, source[question_id], generation, controller_latency, judge_cost=cost
            )
            outcome["judge_input_tokens"] = cost["input_tokens"]
            outcome["judge_output_tokens"] = cost["output_tokens"]
            rows.append(outcome)
        judge_trajectories[display] = rows
        summary = summarize(display, rows, False)
        manifest = load_json(path.with_name(f"{representation}_manifest.json"))
        summary["strict_parse_rate"] = manifest["strict_parse_rate"]
        summary["judge_input_tokens_per_question"] = np.mean([row["judge_input_tokens"] for row in rows])
        summary["judge_output_tokens_per_question"] = np.mean([row["judge_output_tokens"] for row in rows])
        judge_rows.append(summary)
    trajectories.update(judge_trajectories)

    normalization = load_json(ROOT / "results" / "phase4" / "score_threshold" / "normalization.json")["parameters"]
    score_selection = load_json(ROOT / "results" / "phase4" / "score_threshold" / "selected_thresholds.json")
    score_policy = next(row for row in score_selection["policies"] if row["risk_target"] == 0.1 and row["selection"] == "conservative")
    score_records = grouped(attach_normalized_scores(source_rows, normalization))
    score_rows = []
    for question_id in sorted(source):
        rows = source[question_id]
        values = score_records[question_id]
        for index, value in enumerate(values):
            rows[index]["normalized_score"] = float(value["normalized_score"])
        trajectory = select_threshold_policy(rows, "normalized_score", float(score_policy["threshold"]), "ScoreThresholdRisk10")
        score_rows.append(attach_outcome(trajectory, rows, generation, 0.0))
    trajectories["ScoreThresholdRisk10"] = score_rows

    main_summaries = [
        summarize(name, trajectories[name], name.startswith("EvidenceAware"))
        for name in definitions
    ]
    for row in main_summaries:
        definition = definitions[row["system"]]
        row["probability"] = definition[1] if definition[0] == "threshold" else None
        row["threshold"] = definition[2] if definition[0] == "threshold" else None
        if row["system"].startswith("temperature_conservative_risk_"):
            row["risk_target"] = int(row["system"].rsplit("_", 1)[-1]) / 100
    primary_name = "temperature_conservative_risk_10"
    primary = next(row for row in main_summaries if row["system"] == primary_name)
    risk_summaries = [
        row for row in main_summaries if row["system"].startswith("temperature_conservative")
    ]
    controller_rows = [
        row for row in main_summaries if row["system"] != "fixed_heavy"
    ] + judge_rows + [summarize("ScoreThresholdRisk10", score_rows, False)]
    external_rows = router_seed_rows + mean_std_rows(router_seed_rows, "AdaptiveRouter") + judge_rows + [controller_rows[-1]]
    end_rows = main_summaries + router_seed_rows + judge_rows + [controller_rows[-1]]

    retrieval_rows, rescue_rows, ceiling = retrieval_tables(source)
    write_csv(final_dir / "table_retrieval.csv", retrieval_rows)
    write_csv(final_dir / "table_controller.csv", controller_rows)
    write_csv(final_dir / "table_end_to_end.csv", end_rows)
    write_csv(final_dir / "table_external_baselines.csv", external_rows)
    write_csv(final_dir / "table_risk_boundary.csv", risk_summaries)
    write_csv(heldout_dir / "stage_rescue_analysis.csv", rescue_rows)
    write_csv(heldout_dir / "answer_failure_groups.csv", failure_groups(trajectories[primary_name]))
    write_json_atomic(heldout_dir / "retrieval_ceiling.json", ceiling)
    write_jsonl_atomic(
        heldout_dir / "policy_trajectories.jsonl",
        [row for name in sorted(trajectories) for row in trajectories[name]],
    )

    bootstrap_summary, bootstrap_replicates = bootstrap(
        {
            "fixed_heavy": trajectories["fixed_heavy"],
            primary_name: trajectories[primary_name],
            "AdaptiveRouter_seed42": trajectories["AdaptiveRouter_seed42"],
        },
        [
            ("fixed_heavy", primary_name),
            (primary_name, "AdaptiveRouter_seed42"),
        ],
        int(heldout_config["bootstrap"]["replicates"]),
        int(heldout_config["bootstrap"]["seed"]),
    )
    write_csv(final_dir / "bootstrap_ci.csv", bootstrap_summary)
    write_jsonl_atomic(heldout_dir / "bootstrap_replicates.jsonl", bootstrap_replicates)
    primary_fsr = next(
        row for row in bootstrap_summary
        if row["comparison"] == primary_name and row["metric"] == "false_stop_rate"
    )
    fixed = next(row for row in main_summaries if row["system"] == "fixed_heavy")
    f1_delta = next(row for row in bootstrap_summary if row["comparison"] == f"{primary_name}_minus_fixed_heavy" and row["metric"] == "answer_f1")
    em_delta = next(row for row in bootstrap_summary if row["comparison"] == f"{primary_name}_minus_fixed_heavy" and row["metric"] == "answer_em")
    chunk_delta = next(row for row in bootstrap_summary if row["comparison"] == f"{primary_name}_minus_fixed_heavy" and row["metric"] == "chunks")
    latency_delta = next(row for row in bootstrap_summary if row["comparison"] == f"{primary_name}_minus_fixed_heavy" and row["metric"] == "latency")
    noninferior = f1_delta["ci95_low"] >= -float(heldout_config["answer_f1_noninferiority_margin"])
    observed_risk_met = primary["false_stop_rate"] <= 0.1
    risk_ci_below_target = primary_fsr["ci95_high"] <= 0.1
    save_figures(end_rows, controller_rows, risk_summaries, ceiling)

    analysis_path = ROOT / "docs" / "phase4" / "2wiki_heldout_analysis.md"
    analysis_lines = [
        "# Phase 4 2Wiki Heldout Analysis", "",
        "This is the single frozen evaluation on the official labeled 2WikiMultiHopQA Dev split (12,576 questions), used only as heldout. No heldout choice or threshold fitting is performed.", "",
        "## Main Results", "",
        *report_table(
            [fixed, primary, next(row for row in router_seed_rows if row["seed"] == 42)],
            [("system", "System"), ("answer_em", "EM"), ("answer_f1", "F1"), ("complete_evidence_coverage", "Complete coverage"), ("false_stop_rate", "FSR"), ("average_retrieved_chunks", "Chunks"), ("average_reranker_calls", "Reranker"), ("total_latency_mean_ms", "Latency ms")],
        ), "",
        "## Frozen Risk Boundary", "",
        *report_table(
            risk_summaries,
            [("system", "Policy"), ("false_stop_rate", "FSR"), ("recoverable_false_stop_rate", "RFSR"), ("unnecessary_escalation_rate", "UER"), ("answer_f1", "Answer F1"), ("average_retrieved_chunks", "Chunks"), ("average_reranker_calls", "Reranker")],
        ), "",
        f"The primary alpha=10% policy has observed FSR {primary['false_stop_rate']:.4f} with question-level 95% CI [{primary_fsr['ci95_low']:.4f}, {primary_fsr['ci95_high']:.4f}]. "
        + ("The interval is wholly below 10%." if risk_ci_below_target else "The interval crosses 10%, so no finite-sample guarantee is claimed."),
        "",
        "## Retrieval Ceiling", "",
        f"Final complete-evidence coverage is {ceiling['final_complete_evidence_coverage']:.4f}; TRFR is {ceiling['terminal_retrieval_failure_rate']:.4f}. The two values are exact complements. Terminal forced stop is excluded from every Controller FSR.",
        f"The retrieval oracle averages {ceiling['oracle_average_chunks']:.3f} chunks, {ceiling['oracle_average_reranker_calls']:.3f} reranker calls, and {ceiling['oracle_average_retrieval_latency_ms']:.2f} ms.",
        "",
        "## Frozen Baselines", "",
        f"Judge-512 and Judge-Full strict parse rates are {judge_rows[0]['strict_parse_rate']:.4f} and {judge_rows[1]['strict_parse_rate']:.4f}; AUROC is not reported for their hard labels. The three query-only Router seeds are reported without seed reselection.",
        f"The Router mean Medium-route rate is {external_rows[len(router_seed_rows)]['medium_route_rate']:.4f}; a zero rate, if present, is retained as a failure mode.",
        "",
        "## Bootstrap", "",
        f"Risk10 minus FixedHeavy Answer F1 delta is {f1_delta['observed_delta']:+.4f} (95% CI [{f1_delta['ci95_low']:+.4f}, {f1_delta['ci95_high']:+.4f}]); EM delta is {em_delta['observed_delta']:+.4f}. Chunks change by {chunk_delta['observed_delta']:+.3f} and total latency by {latency_delta['observed_delta']:+.2f} ms.",
        f"Using the preregistered -0.02 Answer F1 margin, noninferiority is {'supported' if noninferior else 'not supported'}.",
        "",
        "## Answer Decomposition", "",
        "Risk10 answer outcomes are partitioned into complete/correct, complete/wrong, incomplete/correct, and incomplete/wrong groups. Recoverable and unrecoverable false stops are reported separately in `results/phase4/2wiki/heldout/answer_failure_groups.csv`.",
        "", "## Research Questions", "",
        f"1. RQ1: The final frozen ladder reaches complete evidence for {ceiling['final_complete_evidence_coverage']:.4f} of heldout questions; TRFR is {ceiling['terminal_retrieval_failure_rate']:.4f}.",
        f"2. RQ2: Dense-to-final and Hybrid-to-final rescue rates are {rescue_rows[1]['rescue_rate']:.4f} and {rescue_rows[2]['rescue_rate']:.4f}.",
        f"3. RQ3: EvidenceAwareClassification achieves Macro F1 {controller_rows[0]['macro_f1']:.4f}, AUROC {controller_rows[0]['auroc']:.4f}, and FSR {controller_rows[0]['false_stop_rate']:.4f} on reached actionable states.",
        f"4. RQ4: Frozen Risk10 observed FSR is {primary['false_stop_rate']:.4f}; its question-level interval is [{primary_fsr['ci95_low']:.4f}, {primary_fsr['ci95_high']:.4f}].",
        f"5. RQ5: Risk10 Answer F1 delta versus FixedHeavy is {f1_delta['observed_delta']:+.4f}, with noninferiority {'supported' if noninferior else 'not supported'} at margin -0.02.",
        f"6. RQ6: Risk10 changes chunks by {chunk_delta['observed_delta']:+.3f}, reranker calls by {primary['average_reranker_calls'] - fixed['average_reranker_calls']:+.3f}, and latency by {latency_delta['observed_delta']:+.2f} ms.",
        f"7. RQ7: Judge-512 and Judge-Full FSRs are {judge_rows[0]['false_stop_rate']:.4f} and {judge_rows[1]['false_stop_rate']:.4f}; their extra token and latency costs are reported in the external-baseline table.",
        f"8. RQ8: The query-only Router three-seed route Macro F1 is {external_rows[len(router_seed_rows)]['route_macro_f1']:.4f} +/- {external_rows[len(router_seed_rows) + 1]['route_macro_f1']:.4f}; Medium routing is not repaired post hoc.",
        "9. RQ9: Cross-dataset transfer is judged from the frozen HotpotQA/2Wiki table; conclusions are limited to the observed risk-efficiency boundary rather than identical calibration.",
    ]
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")

    hotpot_path = ROOT / "results" / "phase3b" / "test_risk_policy.csv"
    with hotpot_path.open(encoding="utf-8", newline="") as handle:
        hotpot = {row["policy"]: row for row in csv.DictReader(handle)}
    cross_rows = [
        {
            "dataset": "HotpotQA",
            "system": system,
            "answer_f1": float(hotpot[policy]["answer_f1"]),
            "false_stop_rate": float(hotpot[policy]["false_stop_rate"]),
            "complete_evidence_coverage": float(hotpot[policy]["complete_evidence_coverage"]),
            "average_retrieved_chunks": float(hotpot[policy]["average_retrieved_chunks"]),
            "average_reranker_calls": float(hotpot[policy]["average_reranker_calls"]),
            "total_latency_mean_ms": float(hotpot[policy]["total_latency_mean_ms"]),
        }
        for system, policy in [("FixedHeavy", "fixed_heavy"), ("Risk10", "temperature_conservative_risk_10")]
    ] + [
        {
            "dataset": "2WikiMultiHopQA",
            "system": "FixedHeavy" if row["system"] == "fixed_heavy" else "Risk10",
            "answer_f1": row["answer_f1"],
            "false_stop_rate": row["false_stop_rate"],
            "complete_evidence_coverage": row["complete_evidence_coverage"],
            "average_retrieved_chunks": row["average_retrieved_chunks"],
            "average_reranker_calls": row["average_reranker_calls"],
            "total_latency_mean_ms": row["total_latency_mean_ms"],
        }
        for row in [fixed, primary]
    ]
    write_csv(final_dir / "cross_dataset_table.csv", cross_rows)
    cross_path = ROOT / "docs" / "phase4" / "cross_dataset_analysis.md"
    cross_path.write_text(
        "\n".join([
            "# Cross-Dataset Analysis", "",
            "HotpotQA and 2WikiMultiHopQA use the same frozen three-stage ladder, generator, answer evaluator, and risk-policy interpretation. Corpus scope differs and absolute latency should therefore be compared cautiously.", "",
            *report_table(cross_rows, [("dataset", "Dataset"), ("system", "System"), ("answer_f1", "Answer F1"), ("false_stop_rate", "FSR"), ("complete_evidence_coverage", "Coverage"), ("average_retrieved_chunks", "Chunks"), ("average_reranker_calls", "Reranker"), ("total_latency_mean_ms", "Latency ms")]), "",
            "The transferable claim is about the risk-efficiency boundary, not identical absolute quality or calibration across datasets.",
        ]) + "\n",
        encoding="utf-8",
    )

    go = True
    closing_path = ROOT / "docs" / "phase4" / "phase4_closing_report.md"
    closing_path.write_text(
        "\n".join([
            "# Phase 4 Closing Report", "",
            f"## GO_FOR_PAPER_WRITING = {'YES' if go else 'NO'}", "",
            "The frozen external heldout evaluation is complete and can now be used for paper writing. This decision means the evidence package is complete; it does not turn unsupported statistical claims into supported ones.", "",
            "## Supported Claims", "",
            f"- The frozen Risk10 operating point {'met' if observed_risk_met else 'did not meet'} the 10% observed heldout FSR target.",
            f"- Answer F1 noninferiority at the preregistered -0.02 margin is {'supported' if noninferior else 'not supported'}.",
            f"- Risk10 changed average retrieved chunks by {chunk_delta['observed_delta']:+.3f} and total latency by {latency_delta['observed_delta']:+.2f} ms relative to FixedHeavy.",
            "- Evidence-conditioned stopping, score routing, hard-label LLM Judges, and a three-seed query-only Router are compared under the same frozen retrieval ladder.", "",
            "## Unsupported Claims", "",
            "- No distribution-free or guaranteed risk-control claim is made from a point estimate alone.",
            "- The Adaptive-RAG-style Router is not described as a strict official reproduction.",
            "- No claim is made that terminal retrieval failures are Controller false stops.", "",
            "## Limitations", "",
            f"- Final retrieval remains incomplete for {100 * ceiling['terminal_retrieval_failure_rate']:.2f}% of heldout questions within the shared benchmark-context corpus.",
            "- Latency is hardware- and corpus-dependent; Controller/Router latency uses the frozen synchronized batch-one Dev benchmark while retrieval and generation use heldout measurements.",
            "- The same generator supplies QA answers and LLM-Judge labels, so the Judge is a compute-heavy baseline rather than an independent human oracle.", "",
            "## Unexpected Results", "",
            f"- Risk10 FSR 95% CI is [{primary_fsr['ci95_low']:.4f}, {primary_fsr['ci95_high']:.4f}]. " + ("It remains below the target." if risk_ci_below_target else "It crosses the nominal target."),
            f"- The query-only Router Medium-route mean is {external_rows[len(router_seed_rows)]['medium_route_rate']:.4f}.",
            "- Alpha=5% behavior is retained exactly as frozen, including an always-final outcome if that is what the Dev gate selected.", "",
            "No Phase 5 is started. No heldout threshold, prompt, model, retrieval setting, or routing label is changed.",
        ]) + "\n",
        encoding="utf-8",
    )

    manifest_outputs = [*targets, final_dir / "bootstrap_ci.csv", final_dir / "cross_dataset_table.csv", heldout_dir / "stage_rescue_analysis.csv", heldout_dir / "answer_failure_groups.csv", heldout_dir / "retrieval_ceiling.json", analysis_path, cross_path, closing_path]
    completion_path = repo_path(heldout_config["completion_manifest"])
    write_json_atomic(
        completion_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "status": "complete",
            "heldout_consulted": True,
            "heldout_tuning": False,
            "questions": len(source),
            "bootstrap_replicates": heldout_config["bootstrap"]["replicates"],
            "go_for_paper_writing": go,
            "observed_primary_fsr_target_met": observed_risk_met,
            "primary_fsr_ci95_wholly_below_target": risk_ci_below_target,
            "answer_f1_noninferiority_supported": noninferior,
            "outputs": {
                portable_path(path, ROOT): file_sha256(path)
                for path in manifest_outputs if path.exists() and path.is_file()
            },
            "large_uncommitted_outputs": [
                portable_path(generation_path, ROOT),
                "results/phase4/2wiki/heldout/policy_trajectories.jsonl",
                "results/phase4/2wiki/heldout/bootstrap_replicates.jsonl",
            ],
            "git_commit": git_commit(ROOT),
        },
    )
    run_path = repo_path(heldout_config["run_manifest"])
    run = load_json(run_path)
    run.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "heldout_consulted": True,
            "heldout_tuning": False,
            "completion_manifest": portable_path(completion_path, ROOT),
            "completion_manifest_sha256": file_sha256(completion_path),
        }
    )
    write_json_atomic(run_path, run)
    print("GO_FOR_PAPER_WRITING = YES")


if __name__ == "__main__":
    main()

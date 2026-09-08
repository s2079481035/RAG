"""Evaluate frozen Phase 3B stopping policies with cached stage generations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from generate_phase3b_stage_answers import validate_test_gate
from phase3a_metrics import metrics_from_decisions
from phase3b_metrics import rows_by_question, simulate_trajectory, temperature_scale
from select_phase3b_risk_policy import attach_source_fields
from train_phase2_controller import read_jsonl, write_jsonl_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", required=True, choices=["dev_policy", "test"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def fixed_trajectory(question_rows: list[dict], final_stage: str, policy_name: str) -> dict:
    selected = next(row for row in question_rows if row["stage"] == final_stage)
    return {
        "question_id": selected["question_id"],
        "policy": policy_name,
        "final_stage": final_stage,
        "final_stage_index": int(selected["stage_index"]),
        "decisions": [],
        "controller_calls": 0,
        "final_true_coverage": float(selected["true_coverage"]),
        "final_evidence_state": selected["actual_three_class_label"],
    }


def oracle_trajectory(question_rows: list[dict]) -> dict:
    selected = question_rows[-1]
    for row in question_rows[:-1]:
        if int(row["actual_stop_label"]) == 1:
            selected = row
            break
    return {
        "question_id": selected["question_id"],
        "policy": "oracle_stop_analysis_only",
        "final_stage": selected["stage"],
        "final_stage_index": int(selected["stage_index"]),
        "decisions": [],
        "controller_calls": 0,
        "final_true_coverage": float(selected["true_coverage"]),
        "final_evidence_state": selected["actual_three_class_label"],
    }


def reached_policy_decisions(
    question_rows: list[dict], definition: dict, trajectory: dict
) -> list[tuple[dict, int, float]]:
    if definition["kind"] == "adaptive":
        source = {row["stage"]: row for row in question_rows[:-1]}
        return [
            (
                source[decision["stage"]],
                int(decision["decision"] == "stop"),
                float(decision["stop_probability"]),
            )
            for decision in trajectory["decisions"]
        ]
    reached = []
    for row in question_rows[:-1]:
        decision = int(row["stage"] == trajectory["final_stage"])
        reached.append((row, decision, float(decision)))
        if decision:
            break
    return reached


def policy_metrics(grouped: dict, definition: dict, trajectories: list[dict]) -> dict:
    trajectory_by_question = {row["question_id"]: row for row in trajectories}
    reached = []
    for question_id in sorted(grouped):
        reached.extend(
            reached_policy_decisions(
                grouped[question_id], definition, trajectory_by_question[question_id]
            )
        )
    labels = np.asarray([int(row["actual_stop_label"]) for row, _, _ in reached], dtype=int)
    decisions = np.asarray([decision for _, decision, _ in reached], dtype=int)
    probabilities = np.asarray([probability for _, _, probability in reached], dtype=float)
    metrics = metrics_from_decisions(labels, decisions, probabilities)
    if definition["kind"] != "adaptive":
        metrics["auroc"] = None
    hard = np.asarray(
        [
            int(row["actual_stop_label"]) == 0
            and 0.5 <= float(row["true_coverage"]) < 1.0
            for row, _, _ in reached
        ],
        dtype=bool,
    )
    hard_false = int(np.sum(hard & (decisions == 1)))
    metrics["reached_actionable_decisions"] = len(reached)
    metrics["stop_decision_count"] = int(np.sum(decisions == 1))
    metrics["stop_decision_rate"] = float(np.mean(decisions == 1))
    metrics["hard_partial_continue_count"] = int(np.sum(hard))
    metrics["hard_partial_false_stop_count"] = hard_false
    metrics["hard_partial_false_stop_rate"] = hard_false / int(np.sum(hard)) if np.any(hard) else None
    return metrics


def question_risk_counts(question_rows: list[dict], definition: dict, trajectory: dict) -> dict:
    reached = reached_policy_decisions(question_rows, definition, trajectory)
    labels = np.asarray([int(row["actual_stop_label"]) for row, _, _ in reached], dtype=int)
    decisions = np.asarray([decision for _, decision, _ in reached], dtype=int)
    return {
        "reached_actionable_decisions": len(reached),
        "actionable_continue_count": int(np.sum(labels == 0)),
        "false_stop_count": int(np.sum((labels == 0) & (decisions == 1))),
        "actionable_stop_count": int(np.sum(labels == 1)),
        "unnecessary_escalation_count": int(np.sum((labels == 1) & (decisions == 0))),
    }


def policy_definitions(config: dict, base: dict, selected: dict) -> list[dict]:
    definitions = [
        {"name": "fixed_dense", "kind": "fixed", "final_stage": "dense@5", "deployable": True},
        {"name": "fixed_hybrid", "kind": "fixed", "final_stage": "hybrid@10", "deployable": True},
        {"name": "fixed_heavy", "kind": "fixed", "final_stage": "rerank@20", "deployable": True},
        {"name": "always_stop_earliest", "kind": "fixed", "final_stage": "dense@5", "deployable": False},
        {"name": "always_continue_final", "kind": "fixed", "final_stage": "rerank@20", "deployable": False},
        {"name": "oracle_stop_analysis_only", "kind": "oracle", "deployable": False},
        {
            "name": "uncalibrated_adaptive_phase3a_threshold",
            "kind": "adaptive",
            "calibration_method": "raw",
            "probability_key": "raw_stop_probability",
            "threshold": float(base["phase3a_selected_dev_threshold"]),
            "selection": "phase3a_dev",
            "risk_target": None,
            "deployable": True,
        },
        {
            "name": "temperature_calibrated_threshold_0.5",
            "kind": "adaptive",
            "calibration_method": "temperature",
            "probability_key": "temperature_stop_probability",
            "threshold": 0.5,
            "selection": "fixed_default",
            "risk_target": None,
            "deployable": True,
        },
    ]
    for row in selected["policies"]:
        if row["status"] != "selected":
            continue
        target = float(row["risk_target"])
        method = row["calibration_method"]
        selection = row["selection"]
        definitions.append(
            {
                "name": f"{method}_{selection}_risk_{int(round(target * 100)):02d}",
                "kind": "adaptive",
                "calibration_method": method,
                "probability_key": f"{method}_stop_probability",
                "threshold": float(row["threshold"]),
                "selection": selection,
                "risk_target": target,
                "deployable": True,
            }
        )
    return definitions


def attach_generation_and_cost(
    trajectory: dict,
    row_lookup: dict,
    generation_lookup: dict,
    retrieval_latency_lookup: dict,
    controller_latency_lookup: dict,
) -> dict:
    key = (trajectory["question_id"], trajectory["final_stage"])
    selected = row_lookup[key]
    generated = generation_lookup[key]
    retrieval_latency_ms = float(
        retrieval_latency_lookup[trajectory["question_id"]][f"{trajectory['final_stage']}_ms"]
    )
    controller_latency_ms = sum(
        float(controller_latency_lookup[(trajectory["question_id"], decision["stage"])])
        for decision in trajectory["decisions"]
    )
    generation_latency_ms = float(generated["generation_latency_ms"])
    return {
        **trajectory,
        "retrieved_chunks": int(selected["retrieved_chunks"]),
        "unique_titles": int(selected["unique_titles"]),
        "reranker_calls": int(selected["reranker_calls"]),
        "final_evidence_chunk_ids": list(selected["evidence_chunk_ids"]),
        "final_evidence_document_titles": list(selected["evidence_document_titles"]),
        "model_used_evidence_chunk_ids": list(generated["used_evidence_chunk_ids"]),
        "estimated_retrieval_latency_ms": float(selected["estimated_retrieval_latency_ms"]),
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
        "retrieval_latency_ms": retrieval_latency_ms,
        "controller_latency_ms": controller_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": retrieval_latency_ms + controller_latency_ms + generation_latency_ms,
        "llm_input_tokens": int(generated["prompt_tokens"]),
        "llm_output_tokens": int(generated["output_tokens"]),
        "llm_total_tokens": int(generated["prompt_tokens"]) + int(generated["output_tokens"]),
        "final_answer": generated["extracted_answer"],
        "gold_answer": generated["gold_answer"],
    }


def summarize_policy(definition: dict, metrics: dict, trajectories: list[dict]) -> dict:
    count = len(trajectories)
    generation_latency = [row["generation_latency_ms"] for row in trajectories]
    retrieval_latency = [row["retrieval_latency_ms"] for row in trajectories]
    controller_latency = [row["controller_latency_ms"] for row in trajectories]
    total_latency = [row["total_latency_ms"] for row in trajectories]
    estimated_total = [
        row["estimated_retrieval_latency_ms"] + row["generation_latency_ms"]
        for row in trajectories
    ]
    return {
        "policy": definition["name"],
        "kind": definition["kind"],
        "calibration_method": definition.get("calibration_method"),
        "selection": definition.get("selection"),
        "risk_target": definition.get("risk_target"),
        "threshold": definition.get("threshold"),
        "questions": count,
        "risk_evaluation_scope": "reached_actionable_controller_decisions",
        "false_stop_rate": metrics["false_stop_rate"],
        "hard_partial_false_stop_rate": metrics["hard_partial_false_stop_rate"],
        "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
        "macro_f1": metrics["macro_f1"],
        "stop_precision": metrics["stop_precision"],
        "stop_recall": metrics["stop_recall"],
        "auroc": metrics["auroc"],
        "stop_rate": metrics["stop_decision_rate"],
        "early_stop_rate": sum(row["final_stage_index"] < 3 for row in trajectories) / count,
        "average_final_stage": sum(row["final_stage_index"] for row in trajectories) / count,
        "final_insufficient_evidence_rate": sum(
            row["final_evidence_state"] != "sufficient" for row in trajectories
        ) / count,
        "answer_em": sum(row["answer_exact_match"] for row in trajectories) / count,
        "answer_f1": sum(row["answer_f1"] for row in trajectories) / count,
        "supporting_fact_recall": sum(row["supporting_fact_recall"] for row in trajectories) / count,
        "complete_evidence_coverage": sum(row["complete_evidence_coverage"] for row in trajectories) / count,
        "model_visible_supporting_fact_recall": sum(
            row["model_visible_supporting_fact_recall"] for row in trajectories
        ) / count,
        "model_visible_complete_evidence_coverage": sum(
            row["model_visible_complete_evidence_coverage"] for row in trajectories
        ) / count,
        "average_retrieved_chunks": sum(row["retrieved_chunks"] for row in trajectories) / count,
        "average_unique_titles": sum(row["unique_titles"] for row in trajectories) / count,
        "average_reranker_calls": sum(row["reranker_calls"] for row in trajectories) / count,
        "average_controller_calls": sum(row["controller_calls"] for row in trajectories) / count,
        "average_llm_input_tokens": sum(row["llm_input_tokens"] for row in trajectories) / count,
        "average_llm_output_tokens": sum(row["llm_output_tokens"] for row in trajectories) / count,
        "average_llm_total_tokens": sum(
            row["llm_input_tokens"] + row["llm_output_tokens"] for row in trajectories
        ) / count,
        "generation_latency_mean_ms": sum(generation_latency) / count,
        "generation_latency_median_ms": percentile(generation_latency, 0.5),
        "generation_latency_p95_ms": percentile(generation_latency, 0.95),
        "retrieval_latency_mean_ms": sum(retrieval_latency) / count,
        "retrieval_latency_median_ms": percentile(retrieval_latency, 0.5),
        "retrieval_latency_p95_ms": percentile(retrieval_latency, 0.95),
        "controller_latency_mean_ms": sum(controller_latency) / count,
        "controller_latency_median_ms": percentile(controller_latency, 0.5),
        "controller_latency_p95_ms": percentile(controller_latency, 0.95),
        "total_latency_mean_ms": sum(total_latency) / count,
        "total_latency_median_ms": percentile(total_latency, 0.5),
        "total_latency_p95_ms": percentile(total_latency, 0.95),
        "estimated_total_latency_mean_ms": sum(estimated_total) / count,
        "estimated_total_latency_median_ms": percentile(estimated_total, 0.5),
        "estimated_total_latency_p95_ms": percentile(estimated_total, 0.95),
        "retrieval_latency_source": "phase3b_fresh_component_benchmark_cuda_synchronized",
        "controller_latency_source": "phase3b_batch_size_one_cuda_synchronized",
        "generation_latency_source": "phase3b_cuda_synchronized_per_generation",
        "deployable": definition["deployable"],
        "pareto_optimal_chunks_answer_f1": False,
    }


def mark_pareto(rows: list[dict]) -> None:
    candidates = [row for row in rows if row["deployable"]]
    for row in candidates:
        dominated = any(
            other["answer_f1"] >= row["answer_f1"]
            and other["average_retrieved_chunks"] <= row["average_retrieved_chunks"]
            and (
                other["answer_f1"] > row["answer_f1"]
                or other["average_retrieved_chunks"] < row["average_retrieved_chunks"]
            )
            for other in candidates
            if other is not row
        )
        row["pareto_optimal_chunks_answer_f1"] = not dominated


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_predictions(config: dict, base: dict, split: str, temperature: float) -> list[dict]:
    if split == "dev_policy":
        rows = [
            row for row in read_jsonl(ROOT / "results" / "phase3b" / "calibrated_dev_predictions.jsonl")
            if row["phase3b_subset"] == "dev_policy"
        ]
        return rows
    path = repo_path(base["test_predictions"])
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Phase 3A Test predictions: {path}")
    rows = read_jsonl(path)
    epsilon = float(config["calibration"]["probability_epsilon"])
    for row in rows:
        row["raw_stop_probability"] = float(row["stop_probability"])
        row["temperature_stop_probability"] = float(
            temperature_scale([row["raw_stop_probability"]], temperature, epsilon)[0]
        )
    return rows


def save_curves(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "figures" / "phase3b"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for filename, x_key, y_key, xlabel, ylabel in [
        ("risk_cost_curve.png", "false_stop_rate", "average_retrieved_chunks", "False Stop Rate", "Average retrieved chunks"),
        ("quality_cost_curve.png", "average_retrieved_chunks", "answer_f1", "Average retrieved chunks", "Answer F1"),
    ]:
        figure, axis = plt.subplots(figsize=(6.4, 4.8))
        for method, color in [("raw", "#1f77b4"), ("temperature", "#d62728")]:
            selected = sorted(
                [row for row in rows if row["calibration_method"] == method],
                key=lambda row: float(row["threshold"]),
            )
            axis.plot(
                [float(row[x_key]) for row in selected],
                [float(row[y_key]) for row in selected],
                label=method,
                color=color,
            )
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(figure_dir / filename, dpi=180)
        plt.close(figure)


def enrich_dev_sweep(
    grouped: dict,
    row_lookup: dict,
    generation_lookup: dict,
    retrieval_latency_lookup: dict,
    controller_latency_lookup: dict,
) -> Path:
    source_path = ROOT / "results" / "phase3b" / "dev_threshold_selection_sweep.csv"
    with source_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        method = row["calibration_method"]
        probability_key = f"{method}_stop_probability"
        threshold = float(row["threshold"])
        trajectories = []
        for qid in sorted(grouped):
            trajectory = simulate_trajectory(grouped[qid], probability_key, threshold)
            trajectories.append(
                attach_generation_and_cost(
                    trajectory,
                    row_lookup,
                    generation_lookup,
                    retrieval_latency_lookup,
                    controller_latency_lookup,
                )
            )
        row["answer_em"] = sum(item["answer_exact_match"] for item in trajectories) / len(trajectories)
        row["answer_f1"] = sum(item["answer_f1"] for item in trajectories) / len(trajectories)
        row["average_llm_input_tokens"] = sum(item["llm_input_tokens"] for item in trajectories) / len(trajectories)
        row["average_reranker_calls"] = sum(item["reranker_calls"] for item in trajectories) / len(trajectories)
        row["retrieval_latency_mean_ms"] = sum(item["retrieval_latency_ms"] for item in trajectories) / len(trajectories)
        row["controller_latency_mean_ms"] = sum(item["controller_latency_ms"] for item in trajectories) / len(trajectories)
        row["generation_latency_mean_ms"] = sum(item["generation_latency_ms"] for item in trajectories) / len(trajectories)
        row["total_latency_mean_ms"] = sum(item["total_latency_ms"] for item in trajectories) / len(trajectories)
        row["latency_source"] = "fresh_phase3b_component_benchmarks"
    output = ROOT / "results" / "phase3b" / "dev_threshold_sweep.csv"
    write_csv(output, rows)
    save_curves(rows)
    return output


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.split == "test":
        validate_test_gate(config, args.config)
    base_path = ROOT / "results" / "phase3b" / "base_controller.json"
    selected_path = ROOT / "results" / "phase3b" / "selected_thresholds.json"
    temperature_path = ROOT / "results" / "phase3b" / "temperature_scaling.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    temperature_data = json.loads(temperature_path.read_text(encoding="utf-8"))
    predictions = load_predictions(config, base, args.split, float(temperature_data["temperature"]))

    source_split = "dev" if args.split == "dev_policy" else "test"
    source_path = ROOT / "data" / "phase2" / "controller" / config["base_controller"]["variant"] / f"{source_split}.jsonl"
    source_rows = read_jsonl(source_path)
    if args.split == "dev_policy":
        ids = {row["question_id"] for row in predictions}
        source_rows = [row for row in source_rows if row["question_id"] in ids]
    merged = attach_source_fields(predictions, source_rows, config)
    stage_order = [row["name"] for row in config["stages"]]
    grouped = rows_by_question(merged, stage_order)
    row_lookup = {(row["question_id"], row["stage"]): row for row in merged}

    generation_path = repo_path(config["generation"][f"stage_cache_{args.split}"])
    generation_manifest_path = generation_path.with_name("generation_manifest.json")
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(generation_path) != generation_manifest["output_sha256"]:
        raise ValueError("Stage generation cache changed after generation")
    generation_rows = read_jsonl(generation_path)
    generation_lookup = {(row["question_id"], row["stage"]): row for row in generation_rows}
    if set(generation_lookup) != set(row_lookup):
        raise ValueError("Stage generations and Controller trajectories differ")
    latency_dir = ROOT / "results" / "phase3b" / "latency"
    retrieval_latency_path = latency_dir / f"{args.split}_retrieval.jsonl"
    controller_latency_path = latency_dir / f"{args.split}_controller.jsonl"
    retrieval_manifest_path = latency_dir / f"{args.split}_retrieval_manifest.json"
    controller_manifest_path = latency_dir / f"{args.split}_controller_manifest.json"
    for path, manifest_path_value in [
        (retrieval_latency_path, retrieval_manifest_path),
        (controller_latency_path, controller_manifest_path),
    ]:
        manifest_value = json.loads(manifest_path_value.read_text(encoding="utf-8"))
        if file_sha256(path) != manifest_value["output_sha256"]:
            raise ValueError(f"Latency benchmark changed after measurement: {path}")
    retrieval_latency_lookup = {
        row["question_id"]: row for row in read_jsonl(retrieval_latency_path)
    }
    controller_latency_lookup = {
        (row["question_id"], row["stage"]): row["controller_latency_ms"]
        for row in read_jsonl(controller_latency_path)
    }
    if set(retrieval_latency_lookup) != set(grouped):
        raise ValueError("Retrieval latency benchmark does not cover all policy questions")

    output_dir = ROOT / "results" / "phase3b"
    summary_path = output_dir / ("dev_policy_end_to_end.csv" if args.split == "dev_policy" else "test_risk_policy.csv")
    trajectories_path = output_dir / f"{args.split}_policy_trajectories.jsonl"
    manifest_path = output_dir / f"{args.split}_evaluation_manifest.json"
    extra_summary = output_dir / "end_to_end_summary.csv" if args.split == "test" else None
    targets = [summary_path, trajectories_path, manifest_path] + ([extra_summary] if extra_summary else [])
    if args.split == "dev_policy":
        targets.extend(
            [
                output_dir / "dev_threshold_sweep.csv",
                ROOT / "figures" / "phase3b" / "risk_cost_curve.png",
                ROOT / "figures" / "phase3b" / "quality_cost_curve.png",
            ]
        )
    existing = [path for path in targets if path and path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3B evaluation: {existing}")

    definitions = policy_definitions(config, base, selected)
    summaries = []
    all_trajectories = []
    for definition in definitions:
        if definition["kind"] == "adaptive":
            probability_key = definition["probability_key"]
            threshold = float(definition["threshold"])
            trajectories = [
                simulate_trajectory(grouped[qid], probability_key, threshold)
                for qid in sorted(grouped)
            ]
        elif definition["kind"] == "oracle":
            trajectories = [oracle_trajectory(grouped[qid]) for qid in sorted(grouped)]
        else:
            trajectories = [
                fixed_trajectory(grouped[qid], definition["final_stage"], definition["name"])
                for qid in sorted(grouped)
            ]
        metrics = policy_metrics(grouped, definition, trajectories)
        enriched = []
        for trajectory in trajectories:
            trajectory["policy"] = definition["name"]
            trajectory.update(
                question_risk_counts(
                    grouped[trajectory["question_id"]], definition, trajectory
                )
            )
            enriched.append(
                attach_generation_and_cost(
                    trajectory,
                    row_lookup,
                    generation_lookup,
                    retrieval_latency_lookup,
                    controller_latency_lookup,
                )
            )
        all_trajectories.extend(enriched)
        summaries.append(summarize_policy(definition, metrics, enriched))
    mark_pareto(summaries)
    write_csv(summary_path, summaries)
    if extra_summary:
        write_csv(extra_summary, summaries)
    write_jsonl_atomic(trajectories_path, all_trajectories)
    enriched_sweep = (
        enrich_dev_sweep(
            grouped,
            row_lookup,
            generation_lookup,
            retrieval_latency_lookup,
            controller_latency_lookup,
        )
        if args.split == "dev_policy"
        else None
    )
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3B",
            "git_commit": git_commit(ROOT),
            "split": args.split,
            "questions": len(grouped),
            "test_is_evaluation_only": bool(config["test"]["evaluation_only"]),
            "test_threshold_refit": False,
            "test_calibration_refit": False,
            "prior_test_exposure": config["prior_test_exposure"],
            "config_sha256": file_sha256(args.config),
            "selected_thresholds_sha256": file_sha256(selected_path),
            "temperature_scaling_sha256": file_sha256(temperature_path),
            "stage_generation_manifest_sha256": file_sha256(generation_manifest_path),
            "retrieval_benchmark_manifest_sha256": file_sha256(retrieval_manifest_path),
            "controller_benchmark_manifest_sha256": file_sha256(controller_manifest_path),
            "outputs": {
                "summary": portable_path(summary_path, ROOT),
                "trajectories": portable_path(trajectories_path, ROOT),
                "threshold_sweep_end_to_end": portable_path(enriched_sweep, ROOT) if enriched_sweep else None,
            },
        },
    )
    print(f"evaluated {len(summaries)} policies on {args.split}")


if __name__ == "__main__":
    main()

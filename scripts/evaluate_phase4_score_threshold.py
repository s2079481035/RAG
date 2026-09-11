"""Fit and evaluate the preregistered top-1 retrieval-score routing baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from phase3b_metrics import operational_metrics, rows_by_question, simulate_trajectory, threshold_values
from phase4_score_threshold import attach_normalized_scores, fit_stage_minmax
from select_phase3b_risk_policy import bootstrap_fsr, estimated_retrieval_latency_ms, select_threshold
from train_phase2_controller import read_jsonl, write_jsonl_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"


def score_threshold_values(grid: dict) -> list[float]:
    """Include an always-final endpoint for scores clipped to [0, 1]."""
    thresholds = threshold_values(
        float(grid["start"]), float(grid["stop"]), float(grid["step"])
    )
    if math.isclose(thresholds[-1], 1.0, abs_tol=1e-12):
        thresholds.append(math.nextafter(1.0, math.inf))
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fit", "evaluate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def controller_rows(records: list[dict], normalized: dict, ladder: list[str]) -> list[dict]:
    stage_index = {stage: index + 1 for index, stage in enumerate(ladder)}
    output = []
    for record in attach_normalized_scores(records, normalized):
        view = record["cumulative_evidence_memory"]
        output.append(
            {
                "question_id": record["question_id"],
                "stage": record["stage"],
                "stage_index": stage_index[record["stage"]],
                "actual_stop_label": int(view["stop_label"]),
                "actual_three_class_label": view["evidence_state"],
                "true_coverage": float(view["supporting_fact_recall"]),
                "normalized_score": record["normalized_score"],
                "top1_retrieval_score": record["top1_retrieval_score"],
                "retrieved_chunks": len(view["items"]),
                "unique_titles": len(set(view["document_titles"])),
                "reranker_calls": int(record["stage"] == ladder[-1]),
                "estimated_retrieval_latency_ms": estimated_retrieval_latency_ms(record),
                "evidence_chunk_ids": [item["chunk_id"] for item in view["items"]],
            }
        )
    return output


def trajectory_summary(grouped: dict, threshold: float) -> dict:
    trajectories = [
        simulate_trajectory(rows, "normalized_score", threshold)
        for _, rows in sorted(grouped.items())
    ]
    selected = {
        (row["question_id"], row["stage"]): row
        for rows in grouped.values()
        for row in rows
    }
    selected_rows = [selected[(item["question_id"], item["final_stage"])] for item in trajectories]
    return {
        "questions": len(trajectories),
        "average_retrieved_chunks": float(np.mean([row["retrieved_chunks"] for row in selected_rows])),
        "average_unique_titles": float(np.mean([row["unique_titles"] for row in selected_rows])),
        "average_reranker_calls": float(np.mean([row["reranker_calls"] for row in selected_rows])),
        "average_final_stage": float(np.mean([item["final_stage_index"] for item in trajectories])),
        "estimated_average_retrieval_latency_ms": float(
            np.mean([row["estimated_retrieval_latency_ms"] for row in selected_rows])
        ),
        "trajectories": trajectories,
    }


def evaluated_row(grouped: dict, threshold: float) -> tuple[dict, list[dict]]:
    metrics = operational_metrics(grouped, "normalized_score", threshold)
    trajectory = trajectory_summary(grouped, threshold)
    return (
        {
            "threshold": threshold,
            "questions": trajectory["questions"],
            "macro_f1": metrics["macro_f1"],
            "auroc": metrics["auroc"],
            "false_stop_rate": metrics["false_stop_rate"],
            "hard_partial_false_stop_rate": metrics["hard_partial_false_stop_rate"],
            "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
            "average_retrieved_chunks": trajectory["average_retrieved_chunks"],
            "average_unique_titles": trajectory["average_unique_titles"],
            "average_reranker_calls": trajectory["average_reranker_calls"],
            "average_final_stage": trajectory["average_final_stage"],
            "estimated_average_retrieval_latency_ms": trajectory[
                "estimated_average_retrieval_latency_ms"
            ],
        },
        trajectory["trajectories"],
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    score_config = config["score_threshold"]
    ladder = config["controller_ladder"]
    actionable = ladder[:-1]
    split = args.split or (
        score_config["threshold_selection_split"]
        if args.mode == "fit"
        else config["dataset"]["evaluation_split"]
    )
    if args.mode == "fit" and split != score_config["threshold_selection_split"]:
        raise ValueError("Score normalization and threshold selection are locked to dev_policy")
    if args.mode == "evaluate" and split != config["dataset"]["evaluation_split"]:
        raise ValueError("Formal score-baseline evaluation is locked to heldout")

    data_dir = ROOT / config["outputs"]["data_dir"] / "controller" / "sentence_256"
    source_path = data_dir / f"{split}.jsonl"
    records = read_jsonl(source_path)
    output_dir = ROOT / "results" / "phase4" / "score_threshold"
    normalization_path = output_dir / "normalization.json"
    selection_path = output_dir / "selected_thresholds.json"

    if args.mode == "fit":
        normalized = fit_stage_minmax(records, actionable)
        targets = [normalization_path, selection_path, output_dir / "dev_policy_sweep.csv"]
        existing = [path for path in targets if path.exists()]
        if existing and not args.force:
            raise FileExistsError(f"Refusing to overwrite score-baseline fit: {existing}")
        rows = controller_rows(records, normalized, ladder)
        grouped = rows_by_question(rows, ladder)
        grid = score_config["threshold_grid"]
        thresholds = score_threshold_values(grid)
        bootstrap, _ = bootstrap_fsr(
            grouped,
            "normalized_score",
            thresholds,
            replicates=int(config["controller"]["bootstrap_replicates"]),
            seed=42,
        )
        sweep = []
        for threshold in thresholds:
            row, _ = evaluated_row(grouped, threshold)
            samples = bootstrap[threshold]
            valid = samples[np.isfinite(samples)]
            row["fsr_ci95_low"] = float(np.quantile(valid, 0.025))
            row["fsr_ci95_high"] = float(np.quantile(valid, 0.975))
            sweep.append(row)
        policies = []
        for risk_target in config["controller"]["risk_targets"]:
            for selection, key in [
                ("point", "false_stop_rate"),
                ("conservative", "fsr_ci95_high"),
            ]:
                policies.append(
                    {
                        "risk_target": risk_target,
                        "selection": selection,
                        **select_threshold(sweep, float(risk_target), key),
                    }
                )
        write_json_atomic(
            normalization_path,
            {
                "schema_version": 1,
                "created_at_utc": utc_now(),
                "fit_split": split,
                "feature": score_config["feature"],
                "parameters": normalized,
            },
        )
        write_csv(output_dir / "dev_policy_sweep.csv", sweep)
        write_json_atomic(
            selection_path,
            {
                "schema_version": 1,
                "created_at_utc": utc_now(),
                "selection_split": split,
                "heldout_consulted": False,
                "decision_rule": "stop_if_normalized_score_greater_than_or_equal_to_threshold",
                "always_final_threshold": thresholds[-1],
                "policies": policies,
            },
        )
        print(selection_path.relative_to(ROOT))
        return

    if not normalization_path.exists() or not selection_path.exists():
        raise FileNotFoundError("Fit the score baseline on dev_policy before heldout evaluation")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("heldout_consulted") is not False:
        raise ValueError("Invalid score-baseline selection manifest")
    rows = controller_rows(records, normalization["parameters"], ladder)
    grouped = rows_by_question(rows, ladder)
    summaries = []
    all_trajectories = []
    for policy in selection["policies"]:
        if policy["status"] != "selected":
            continue
        summary, trajectories = evaluated_row(grouped, float(policy["threshold"]))
        policy_name = f"score_{policy['selection']}_risk_{int(100 * policy['risk_target']):02d}"
        summaries.append({"policy": policy_name, "risk_target": policy["risk_target"], **summary})
        all_trajectories.extend({"policy": policy_name, **row} for row in trajectories)
    output_path = output_dir / "heldout_summary.csv"
    trajectory_path = output_dir / "heldout_trajectories.jsonl"
    manifest_path = output_dir / "heldout_manifest.json"
    existing = [path for path in [output_path, trajectory_path, manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite heldout score baseline: {existing}")
    write_csv(output_path, summaries)
    write_jsonl_atomic(trajectory_path, all_trajectories)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "git_commit": git_commit(ROOT),
            "source": portable_path(source_path, ROOT),
            "selection": portable_path(selection_path, ROOT),
            "normalization": portable_path(normalization_path, ROOT),
            "split": split,
            "heldout_tuning": False,
            "summaries": summaries,
        },
    )
    print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

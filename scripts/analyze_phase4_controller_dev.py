"""Freeze the Train-derived Phase 4 Controller Dev gate without reading heldout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

try:
    from analyze_phase4_retrieval_ceiling import controller_recoverability_metrics
    from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
    from phase3a_metrics import coverage_bucket
    from phase3b_metrics import (
        actionable_metrics,
        calibration_metrics,
        fit_temperature,
        operational_metrics,
        rows_by_question,
        simulate_trajectory,
        temperature_scale,
        threshold_values,
    )
    from select_phase3b_risk_policy import (
        attach_source_fields,
        bootstrap_fsr,
        select_threshold,
        summarize_trajectories,
    )
except ModuleNotFoundError:  # Imported as scripts.* in tests.
    from scripts.analyze_phase4_retrieval_ceiling import (
        controller_recoverability_metrics,
    )
    from scripts.experiment_utils import (
        git_commit,
        portable_path,
        utc_now,
        write_json_atomic,
    )
    from scripts.phase3a_metrics import coverage_bucket
    from scripts.phase3b_metrics import (
        actionable_metrics,
        calibration_metrics,
        fit_temperature,
        operational_metrics,
        rows_by_question,
        simulate_trajectory,
        temperature_scale,
        threshold_values,
    )
    from scripts.select_phase3b_risk_policy import (
        attach_source_fields,
        bootstrap_fsr,
        select_threshold,
        summarize_trajectories,
    )


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"
DEFAULT_CONTROLLER_CONFIG = ROOT / "configs" / "phase4" / "controller.json"
ALLOWED_SPLITS = {"dev_calibration", "dev_policy"}
GROUPS = {
    "query_stage": "query_stage",
    "final_evidence_aware": "final_evidence_aware",
}
SUMMARY_METRICS = [
    "macro_f1",
    "stop_f1",
    "auroc",
    "false_stop_rate",
    "hard_partial_false_stop_rate",
    "unnecessary_escalation_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_config(ladder: list[str]) -> list[dict]:
    if ladder != ["dense@5", "hybrid@10", "rerank@20"]:
        raise ValueError(f"Unexpected frozen ladder: {ladder}")
    return [
        {"index": 1, "name": "dense@5", "reranker_calls": 0},
        {"index": 2, "name": "hybrid@10", "reranker_calls": 0},
        {"index": 3, "name": "rerank@20", "reranker_calls": 1},
    ]


def validate_split_rows(rows: list[dict], split: str, stage_order: list[str]) -> None:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Phase 4 Dev gate cannot read split {split!r}")
    if not rows:
        raise ValueError(f"No {split} rows were supplied")
    observed_splits = {row["split"] for row in rows}
    if observed_splits != {split}:
        raise ValueError(f"Expected only {split}, got {sorted(observed_splits)}")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["question_id"], []).append(row["stage"])
    for question_id, stages in grouped.items():
        if stages != stage_order:
            raise ValueError(
                f"Invalid stage order for {question_id}: {stages} != {stage_order}"
            )


def merge_predictions(
    predictions: list[dict], source_rows: list[dict], ladder: list[str], split: str
) -> list[dict]:
    validate_split_rows(predictions, split, ladder)
    validate_split_rows(source_rows, split, ladder)
    prediction_keys = {(row["question_id"], row["stage"]) for row in predictions}
    source_keys = {(row["question_id"], row["stage"]) for row in source_rows}
    if prediction_keys != source_keys:
        missing = sorted(source_keys - prediction_keys)[:5]
        extra = sorted(prediction_keys - source_keys)[:5]
        raise ValueError(f"Prediction/source mismatch; missing={missing}, extra={extra}")
    helper_config = {
        "base_controller": {"evidence_mode": "cumulative"},
        "stages": stage_config(ladder),
    }
    return attach_source_fields(predictions, source_rows, helper_config)


def trajectories_and_cost(
    merged: list[dict], probability_key: str, threshold: float, ladder: list[str]
) -> tuple[dict, list[dict]]:
    grouped = rows_by_question(merged, ladder)
    trajectories = [
        simulate_trajectory(grouped[qid], probability_key, threshold, include_evidence=False)
        for qid in sorted(grouped)
    ]
    lookup = {(row["question_id"], row["stage"]): row for row in merged}
    selected = [lookup[(row["question_id"], row["final_stage"])] for row in trajectories]
    return summarize_trajectories(trajectories, selected), trajectories


def metric_row(
    group: str,
    seed: int,
    threshold: float,
    scope: str,
    metrics: dict,
    cost: dict | None,
) -> dict:
    return {
        "group": group,
        "seed": seed,
        "threshold_source": "dev_calibration_selected_during_training",
        "threshold": threshold,
        "evaluation_split": "dev_policy",
        "evaluation_scope": scope,
        "samples": metrics["samples"],
        "macro_f1": metrics["macro_f1"],
        "stop_f1": metrics["stop_f1"],
        "auroc": metrics["auroc"],
        "false_stop_rate": metrics["false_stop_rate"],
        "hard_partial_false_stop_rate": metrics["hard_partial_false_stop_rate"],
        "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
        "average_retrieved_chunks": None if cost is None else cost["average_retrieved_chunks"],
        "average_reranker_calls": None if cost is None else cost["average_reranker_calls"],
        "estimated_average_retrieval_latency_ms": (
            None if cost is None else cost["estimated_average_retrieval_latency_ms"]
        ),
    }


def aggregate_seed_rows(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["group"], row["evaluation_scope"]) for row in rows})
    for group, scope in keys:
        selected = [
            row for row in rows
            if row["group"] == group and row["evaluation_scope"] == scope
        ]
        if len(selected) < 2:
            raise ValueError(f"At least two seeds are required for {group}/{scope}")
        result = {
            "group": group,
            "evaluation_split": "dev_policy",
            "evaluation_scope": scope,
            "seeds": ",".join(str(row["seed"]) for row in selected),
            "runs": len(selected),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            result[f"mean_{metric}"] = statistics.fmean(values) if values else None
            result[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(result)
    return output


def calibration_rows(
    subsets: dict[str, list[dict]], calibration_config: dict
) -> list[dict]:
    bins = int(calibration_config["ece_bins"])
    epsilon = float(calibration_config["probability_epsilon"])
    high = float(calibration_config["high_confidence_stop_probability"])
    output = []
    for split, rows in subsets.items():
        labels = np.asarray([int(row["actual_stop_label"]) for row in rows], dtype=int)
        for method, key in (
            ("raw", "raw_stop_probability"),
            ("temperature", "temperature_stop_probability"),
        ):
            probabilities = np.asarray([float(row[key]) for row in rows], dtype=float)
            hard = [
                row for row in rows
                if int(row["actual_stop_label"]) == 0
                and coverage_bucket(float(row["true_coverage"])) == "hard_partial"
            ]
            high_false = sum(float(row[key]) >= high for row in hard)
            output.append(
                {
                    "split": split,
                    "method": method,
                    "evaluation_scope": "all_actionable_dense_and_hybrid_decisions",
                    **calibration_metrics(labels, probabilities, bins, epsilon),
                    "hard_partial_continue_count": len(hard),
                    "hard_partial_mean_stop_probability": (
                        statistics.fmean(float(row[key]) for row in hard) if hard else None
                    ),
                    "hard_partial_high_confidence_false_stop_count": high_false,
                    "hard_partial_high_confidence_false_stop_rate": (
                        high_false / len(hard) if hard else None
                    ),
                }
            )
    return output


def select_primary(selections: list[dict], primary: dict) -> dict:
    matches = [
        row for row in selections
        if row["calibration_method"] == primary["calibration"]
        and row["selection"] == primary["selection"]
        and math.isclose(float(row["risk_target"]), float(primary["risk_target"]))
    ]
    if len(matches) != 1 or matches[0]["status"] != "selected":
        raise ValueError(f"Primary Dev policy was not uniquely selected: {matches}")
    return matches[0]


def enrich_selection(selection: dict, sweep_rows: list[dict]) -> dict:
    if selection["status"] != "selected":
        return selection
    matches = [
        row for row in sweep_rows
        if row["calibration_method"] == selection["calibration_method"]
        and float(row["threshold"]) == float(selection["threshold"])
    ]
    if len(matches) != 1:
        raise ValueError(f"Selected threshold does not identify one sweep row: {selection}")
    row = matches[0]
    return {
        **selection,
        "macro_f1": row["macro_f1"],
        "auroc": row["auroc"],
        "hard_partial_false_stop_rate": row["hard_partial_false_stop_rate"],
        "unnecessary_escalation_rate": row["unnecessary_escalation_rate"],
        "final_supporting_fact_recall": row["final_supporting_fact_recall"],
        "final_complete_evidence_coverage": row["final_complete_evidence_coverage"],
        "average_unique_titles": row["average_unique_titles"],
        "average_controller_calls": row["average_controller_calls"],
    }


def percent(value) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def number(value, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def render_report(
    summary_rows: list[dict],
    calibration: list[dict],
    temperature: float,
    selections: list[dict],
    primary: dict,
    always_final: dict,
    recoverability: dict | None,
) -> str:
    sequential = [
        row for row in summary_rows
        if row["evaluation_scope"] == "reached_actionable_sequential_decisions"
    ]
    lines = [
        "# 2Wiki In-domain Controller Dev Gate",
        "",
        "This report uses only `dev_calibration` and `dev_policy` derived from the "
        "official 2Wiki Train split. The labeled official Dev (`heldout`) is not read.",
        "",
        "## Multi-seed Stability",
        "",
        "Each run uses its threshold selected on `dev_calibration`. Formal FSR below "
        "scores only sequentially reached `dense@5` and `hybrid@10` decisions; the "
        "forced `rerank@20` terminal state is excluded.",
        "",
        "| Controller | Macro F1 | Stop F1 | AUROC | FSR | Hard Partial FSR | UER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sequential:
        lines.append(
            f"| {row['group']} | {number(row['mean_macro_f1'])} +/- {number(row['std_macro_f1'])} | "
            f"{number(row['mean_stop_f1'])} +/- {number(row['std_stop_f1'])} | "
            f"{number(row['mean_auroc'])} +/- {number(row['std_auroc'])} | "
            f"{percent(row['mean_false_stop_rate'])} +/- {percent(row['std_false_stop_rate'])} | "
            f"{percent(row['mean_hard_partial_false_stop_rate'])} +/- {percent(row['std_hard_partial_false_stop_rate'])} | "
            f"{percent(row['mean_unnecessary_escalation_rate'])} +/- {percent(row['std_unnecessary_escalation_rate'])} |"
        )

    lines.extend(
        [
            "",
            "The operational model is seed 42 by preregistration, not the best "
            "`dev_policy` seed. All-stage training metrics remain diagnostics and are "
            "not substituted for sequential FSR.",
            "",
            "## Calibration",
            "",
            f"Temperature is fitted on seed 42 actionable `dev_calibration` decisions: T = {temperature:.8f}.",
            "",
            "| Split | Method | NLL | Brier | ECE | AUROC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in calibration:
        lines.append(
            f"| {row['split']} | {row['method']} | {number(row['nll'])} | "
            f"{number(row['brier'])} | {number(row['ece'])} | {number(row['auroc'])} |"
        )

    temperature_selections = [
        row for row in selections if row["calibration_method"] == "temperature"
    ]
    lines.extend(
        [
            "",
            "Scalar Temperature Scaling preserves AUROC ranking. Here it improves NLL, "
            "Brier score, and ECE on both Train-derived Dev subsets; this is a "
            "calibration result, not a discrimination gain.",
            "",
            "## Dev Risk-Efficiency Boundary",
            "",
            "| Target | Selection | Threshold | FSR | FSR CI high | Hard Partial FSR | Chunks | Rerank calls | Latency ms | Final coverage |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in temperature_selections:
        if row["status"] != "selected":
            lines.append(
                f"| {percent(row['risk_target'])} | {row['selection']} | infeasible | "
                "N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        lines.append(
            f"| {percent(row['risk_target'])} | {row['selection']} | "
            f"{number(row['threshold'], 3)} | {percent(row['observed_false_stop_rate'])} | "
            f"{percent(row['fsr_ci95_high'])} | "
            f"{percent(row['hard_partial_false_stop_rate'])} | "
            f"{number(row['average_retrieved_chunks'], 3)} | "
            f"{number(row['average_reranker_calls'], 3)} | "
            f"{number(row['estimated_average_retrieval_latency_ms'], 2)} | "
            f"{percent(row['final_complete_evidence_coverage'])} |"
        )

    chunk_reduction = (
        float(always_final["average_retrieved_chunks"])
        - float(primary["average_retrieved_chunks"])
    ) / float(always_final["average_retrieved_chunks"])
    rerank_reduction = (
        float(always_final["average_reranker_calls"])
        - float(primary["average_reranker_calls"])
    ) / float(always_final["average_reranker_calls"])
    latency_reduction = (
        float(always_final["estimated_average_retrieval_latency_ms"])
        - float(primary["estimated_average_retrieval_latency_ms"])
    ) / float(always_final["estimated_average_retrieval_latency_ms"])
    lines.extend(
        [
            "",
            "## Frozen Primary Dev Policy",
            "",
            f"- Policy: `{primary['calibration_method']}_{primary['selection']}_risk_{int(100 * float(primary['risk_target'])):02d}`",
            f"- Threshold: `{primary['threshold']}`",
            f"- Observed sequential Dev-policy FSR: {percent(primary['observed_false_stop_rate'])}",
            f"- Question-bootstrap FSR 95% CI: [{percent(primary['fsr_ci95_low'])}, {percent(primary['fsr_ci95_high'])}]",
            f"- Average retrieved chunks: {number(primary['average_retrieved_chunks'], 3)}",
            f"- Average reranker calls: {number(primary['average_reranker_calls'], 3)}",
            f"- Estimated retrieval latency: {number(primary['estimated_average_retrieval_latency_ms'], 2)} ms",
            f"- Final complete evidence coverage: {percent(primary['final_complete_evidence_coverage'])}",
            f"- Cost reduction vs always-final: chunks {percent(chunk_reduction)}, reranker calls {percent(rerank_reduction)}, estimated retrieval latency {percent(latency_reduction)}",
        ]
    )
    if recoverability is not None:
        lines.extend(
            [
                f"- Recoverable FSR: {percent(recoverability['recoverable_false_stop_rate'])}",
                f"- Unrecoverable early-stop rate: {percent(recoverability['unrecoverable_early_stop_rate'])}",
                f"- Recoverable share of false stops: {percent(recoverability['recoverable_share_of_false_stops'])}",
                f"- Recoverable share of reached Continue states: {percent(recoverability['reached_recoverable_continue_states'] / recoverability['reached_continue_states'])}",
            ]
        )
    lines.extend(
        [
            "",
            "## Gate Boundary",
            "",
            "No heldout metric, model selection, threshold tuning, retrieval change, or "
            "label change is performed here. Heldout remains locked until this Dev gate "
            "and any remaining external-baseline protocol choices are committed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    controller_config = json.loads(args.controller_config.read_text(encoding="utf-8"))
    policy = config["in_domain_policy"]
    ladder = config["controller_ladder"]
    stage_order = [row["name"] for row in stage_config(ladder)]
    if policy["decision_stages"] != stage_order[:-1]:
        raise ValueError("Actionable stages differ from the frozen retrieval ladder")
    if policy["forced_final_stage"] != stage_order[-1]:
        raise ValueError("Forced final stage differs from the frozen retrieval ladder")
    if policy["calibration"]["method"] != "temperature":
        raise ValueError("Phase 4 in-domain calibration must remain temperature scaling")
    if not policy["heldout_tuning_forbidden"]:
        raise ValueError("Phase 4 heldout tuning must remain forbidden")

    seeds = [int(seed) for seed in config["controller"]["seeds"]]
    operational_seed = int(policy["operational_seed"])
    if operational_seed not in seeds:
        raise ValueError("Operational seed must be one of the preregistered seeds")

    data_dir = ROOT / controller_config["data"]["controller_dir"]
    source_paths = {
        split: data_dir / f"{split}.jsonl" for split in ALLOWED_SPLITS
    }
    sources = {split: read_jsonl(path) for split, path in source_paths.items()}
    for split, rows in sources.items():
        validate_split_rows(rows, split, stage_order)

    output_dir = ROOT / "results" / "phase4" / "2wiki" / "controller_dev"
    report_path = ROOT / "docs" / "phase4" / "2wiki_controller_dev_analysis.md"
    seed_results_path = output_dir / "dev_seed_results.csv"
    seed_summary_path = output_dir / "dev_multiseed_summary.csv"
    calibration_metrics_path = output_dir / "calibration_metrics.csv"
    temperature_path = output_dir / "temperature_scaling.json"
    calibrated_path = output_dir / "calibrated_dev_policy_predictions.jsonl"
    sweep_path = output_dir / "dev_policy_threshold_sweep.csv"
    bootstrap_path = output_dir / "dev_policy_bootstrap_fsr.jsonl"
    trajectory_path = output_dir / "dev_policy_threshold_trajectories.jsonl"
    selection_path = output_dir / "selected_thresholds.json"
    recoverability_path = output_dir / "primary_policy_recoverability.json"
    manifest_path = output_dir / "dev_gate_manifest.json"
    targets = [
        report_path,
        seed_results_path,
        seed_summary_path,
        calibration_metrics_path,
        temperature_path,
        calibrated_path,
        sweep_path,
        bootstrap_path,
        trajectory_path,
        selection_path,
        recoverability_path,
        manifest_path,
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 4 Dev gate: {existing}")

    seed_rows = []
    merged_by_run = {}
    run_sources = {}
    for group, directory in GROUPS.items():
        for seed in seeds:
            run_dir = ROOT / "experiments" / "phase4" / "2wiki" / directory / f"seed{seed}"
            manifest_path_run = run_dir / "run_manifest.json"
            prediction_path = run_dir / "evaluation" / "dev_policy" / "original_predictions.jsonl"
            if not manifest_path_run.exists() or not prediction_path.exists():
                raise FileNotFoundError(f"Incomplete run artifacts: {run_dir}")
            manifest = json.loads(manifest_path_run.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise ValueError(f"Run is not complete: {run_dir}")
            predictions = read_jsonl(prediction_path)
            merged = merge_predictions(predictions, sources["dev_policy"], ladder, "dev_policy")
            merged_by_run[(group, seed)] = merged
            run_sources[f"{group}_seed{seed}"] = {
                "run_manifest": portable_path(manifest_path_run, ROOT),
                "run_manifest_sha256": file_sha256(manifest_path_run),
                "dev_policy_predictions": portable_path(prediction_path, ROOT),
                "dev_policy_predictions_sha256": file_sha256(prediction_path),
            }
            threshold = float(manifest["selected_dev_threshold"])
            actionable = [row for row in merged if row["stage"] in policy["decision_stages"]]
            all_metrics = actionable_metrics(actionable, "stop_probability", threshold)
            grouped = rows_by_question(merged, ladder)
            sequential_metrics = operational_metrics(grouped, "stop_probability", threshold)
            cost, _ = trajectories_and_cost(merged, "stop_probability", threshold, ladder)
            seed_rows.append(
                metric_row(
                    group, seed, threshold,
                    "all_actionable_dense_and_hybrid_decisions", all_metrics, None,
                )
            )
            seed_rows.append(
                metric_row(
                    group, seed, threshold,
                    "reached_actionable_sequential_decisions", sequential_metrics, cost,
                )
            )

    seed_summary = aggregate_seed_rows(seed_rows)
    write_csv(seed_results_path, seed_rows)
    write_csv(seed_summary_path, seed_summary)

    operational_dir = (
        ROOT / "experiments" / "phase4" / "2wiki" / "final_evidence_aware"
        / f"seed{operational_seed}"
    )
    dev_calibration_path = operational_dir / "dev_predictions.jsonl"
    dev_calibration_predictions = read_jsonl(dev_calibration_path)
    merged_calibration = merge_predictions(
        dev_calibration_predictions,
        sources["dev_calibration"],
        ladder,
        "dev_calibration",
    )
    merged_policy = merged_by_run[("final_evidence_aware", operational_seed)]
    actionable_calibration = [
        row for row in merged_calibration if row["stage"] in policy["decision_stages"]
    ]
    actionable_policy = [
        row for row in merged_policy if row["stage"] in policy["decision_stages"]
    ]
    calibration_config = policy["calibration"]
    labels = np.asarray(
        [int(row["actual_stop_label"]) for row in actionable_calibration], dtype=int
    )
    raw = np.asarray(
        [float(row["stop_probability"]) for row in actionable_calibration], dtype=float
    )
    epsilon = float(calibration_config["probability_epsilon"])
    temperature, optimized_nll = fit_temperature(
        labels,
        raw,
        log_bounds=tuple(calibration_config["temperature_log_bounds"]),
        iterations=int(calibration_config["optimizer_iterations"]),
        epsilon=epsilon,
    )
    for rows in (merged_calibration, merged_policy):
        scaled = temperature_scale(
            [float(row["stop_probability"]) for row in rows], temperature, epsilon
        )
        for row, probability in zip(rows, scaled):
            row["raw_stop_probability"] = float(row["stop_probability"])
            row["temperature_stop_probability"] = float(probability)
    calibration_by_split = {
        "dev_calibration": [
            row for row in merged_calibration if row["stage"] in policy["decision_stages"]
        ],
        "dev_policy": [
            row for row in merged_policy if row["stage"] in policy["decision_stages"]
        ],
    }
    calibration_output = calibration_rows(calibration_by_split, calibration_config)
    write_csv(calibration_metrics_path, calibration_output)
    write_json_atomic(
        temperature_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "fit_split": "dev_calibration",
            "fit_questions": len({row["question_id"] for row in merged_calibration}),
            "fit_stage_decisions": len(actionable_calibration),
            "fit_stages": policy["decision_stages"],
            "operational_seed": operational_seed,
            "operational_seed_rule": policy["operational_seed_rule"],
            "temperature": temperature,
            "optimized_nll": optimized_nll,
            "probability_epsilon": epsilon,
            "source_predictions": portable_path(dev_calibration_path, ROOT),
            "source_predictions_sha256": file_sha256(dev_calibration_path),
            "heldout_consulted": False,
        },
    )

    calibrated_rows = []
    for row in merged_policy:
        calibrated_rows.append(
            {
                **row,
                "stop_probability": row["temperature_stop_probability"],
                "calibration_method": "temperature",
            }
        )
    write_jsonl_atomic(calibrated_path, calibrated_rows)

    grid = policy["threshold_grid"]
    thresholds = threshold_values(
        float(grid["start"]), float(grid["stop"]), float(grid["step"])
    )
    if grid["include_always_final_endpoint"]:
        thresholds.append(math.nextafter(1.0, math.inf))
    grouped = rows_by_question(merged_policy, ladder)
    sweep_rows = []
    bootstrap_rows = []
    trajectory_rows = []
    probability_keys = {
        "raw": "raw_stop_probability",
        "temperature": "temperature_stop_probability",
    }
    bootstrap_config = policy["bootstrap"]
    for method, probability_key in probability_keys.items():
        bootstrap_values, _ = bootstrap_fsr(
            grouped,
            probability_key,
            thresholds,
            replicates=int(bootstrap_config["replicates"]),
            seed=int(bootstrap_config["seed"]),
        )
        for threshold in thresholds:
            metrics = operational_metrics(grouped, probability_key, threshold)
            cost, trajectories = trajectories_and_cost(
                merged_policy, probability_key, threshold, ladder
            )
            valid = bootstrap_values[threshold][
                np.isfinite(bootstrap_values[threshold])
            ]
            low, high = np.quantile(valid, [0.025, 0.975])
            sweep_rows.append(
                {
                    "split": "dev_policy",
                    "calibration_method": method,
                    "threshold": threshold,
                    **metrics,
                    **cost,
                    "fsr_ci95_low": float(low),
                    "fsr_ci95_high": float(high),
                    "bootstrap_valid_replicates": len(valid),
                    "risk_evaluation_scope": "reached_actionable_controller_decisions",
                }
            )
            for trajectory in trajectories:
                trajectory_rows.append(
                    {
                        **trajectory,
                        "calibration_method": method,
                        "threshold": threshold,
                    }
                )
            bootstrap_rows.extend(
                {
                    "split": "dev_policy",
                    "calibration_method": method,
                    "threshold": threshold,
                    "replicate": index,
                    "false_stop_rate": float(value),
                    "sampling_unit": "question_id",
                }
                for index, value in enumerate(bootstrap_values[threshold])
                if np.isfinite(value)
            )
    write_csv(sweep_path, sweep_rows)
    write_jsonl_atomic(bootstrap_path, bootstrap_rows)
    write_jsonl_atomic(trajectory_path, trajectory_rows)

    selections = []
    for method in probability_keys:
        method_rows = [row for row in sweep_rows if row["calibration_method"] == method]
        for target in (float(value) for value in policy["risk_targets"]):
            selections.append(
                enrich_selection({
                    "calibration_method": method,
                    "selection": "point",
                    "risk_target": target,
                    **select_threshold(method_rows, target, "false_stop_rate"),
                }, sweep_rows)
            )
            selections.append(
                enrich_selection({
                    "calibration_method": method,
                    "selection": "conservative",
                    "risk_target": target,
                    **select_threshold(method_rows, target, "fsr_ci95_high"),
                }, sweep_rows)
            )
    primary = select_primary(selections, policy["primary_report_policy"])
    always_final_matches = [
        row for row in sweep_rows
        if row["calibration_method"] == "temperature"
        and float(row["threshold"]) == math.nextafter(1.0, math.inf)
    ]
    if len(always_final_matches) != 1:
        raise ValueError("Always-final endpoint is missing or duplicated")
    always_final = always_final_matches[0]
    write_json_atomic(
        selection_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "selection_split": "dev_policy",
            "heldout_consulted": False,
            "operational_seed": operational_seed,
            "risk_definition": policy["risk_metric"],
            "decision_stages": policy["decision_stages"],
            "forced_final_stage": policy["forced_final_stage"],
            "always_final_threshold": math.nextafter(1.0, math.inf),
            "primary_report_policy": policy["primary_report_policy"],
            "primary_selected_policy": primary,
            "policies": selections,
        },
    )

    recoverability = None
    diagnostic_path = ROOT / "results" / "phase4" / "2wiki" / "recoverability" / "dev_policy.jsonl"
    if diagnostic_path.exists():
        indexed_predictions = {
            (row["question_id"], row["stage"]): row for row in calibrated_rows
        }
        recoverability = controller_recoverability_metrics(
            diagnostic_path,
            indexed_predictions,
            "in_domain_temperature_conservative_risk_10",
            "dev_policy",
            str(primary["threshold"]),
        )
    write_json_atomic(
        recoverability_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "heldout_consulted": False,
            "diagnostic_available": recoverability is not None,
            "metrics": recoverability,
        },
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            seed_summary,
            calibration_output,
            temperature,
            selections,
            primary,
            always_final,
            recoverability,
        ),
        encoding="utf-8",
        newline="\n",
    )
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "git_commit": git_commit(ROOT),
            "config": portable_path(args.config.resolve(), ROOT),
            "config_sha256": file_sha256(args.config),
            "controller_config": portable_path(args.controller_config.resolve(), ROOT),
            "controller_config_sha256": file_sha256(args.controller_config),
            "allowed_input_splits": sorted(ALLOWED_SPLITS),
            "heldout_consulted": False,
            "operational_seed": operational_seed,
            "run_sources": run_sources,
            "source_data": {
                split: {
                    "path": portable_path(path, ROOT),
                    "sha256": file_sha256(path),
                }
                for split, path in source_paths.items()
            },
            "outputs": {
                "report": portable_path(report_path, ROOT),
                "seed_results": portable_path(seed_results_path, ROOT),
                "seed_summary": portable_path(seed_summary_path, ROOT),
                "calibration_metrics": portable_path(calibration_metrics_path, ROOT),
                "temperature": portable_path(temperature_path, ROOT),
                "calibrated_predictions": portable_path(calibrated_path, ROOT),
                "threshold_sweep": portable_path(sweep_path, ROOT),
                "bootstrap": portable_path(bootstrap_path, ROOT),
                "threshold_trajectories": portable_path(trajectory_path, ROOT),
                "selected_thresholds": portable_path(selection_path, ROOT),
                "recoverability": portable_path(recoverability_path, ROOT),
            },
        },
    )
    print(f"temperature={temperature:.8f}")
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()

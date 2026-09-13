"""Freeze the three-seed Adaptive-RAG-style Router Dev comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from phase4_adaptive_router import (
    ROUTE_NAMES,
    STAGE_NAMES,
    file_sha256,
    grouped_trajectories,
    read_jsonl,
    summarize_routes,
    write_csv,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "adaptive_rag_router.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_index(path: Path, split: str) -> dict[str, list[dict]]:
    return {
        rows[0]["question_id"]: rows
        for rows in grouped_trajectories(path, split)
    }


def controller_predictions(
    rows: list[dict], probability_key: str, threshold: float
) -> dict[str, int]:
    grouped = {}
    for row in rows:
        if row["stage"] not in STAGE_NAMES[:2]:
            continue
        grouped.setdefault(row["question_id"], {})[row["stage"]] = row
    predictions = {}
    for question_id, stages in grouped.items():
        if set(stages) != set(STAGE_NAMES[:2]):
            raise ValueError(f"Incomplete Controller predictions: {question_id}")
        selected = 2
        for index, stage in enumerate(STAGE_NAMES[:2]):
            if float(stages[stage][probability_key]) >= threshold:
                selected = index
                break
        predictions[question_id] = selected
    return predictions


def system_row(system: str, seed, role: str, metrics: dict) -> dict:
    return {
        "system": system,
        "seed": seed,
        "role": role,
        **metrics,
    }


def mean_std(rows: list[dict], metric: str) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows]
    return statistics.fmean(values), statistics.stdev(values)


def fmt(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    args = parse_args()
    config = load(args.config)
    protocol_path = ROOT / "configs" / "phase4" / "protocol.json"
    protocol = load(protocol_path)
    adaptive = protocol["external_baselines"]["adaptive_rag"]
    if adaptive["execution_status"] != "required_before_heldout":
        raise ValueError("Adaptive Router Dev Gate is not required by the protocol")
    result_dir = ROOT / config["output"]["result_dir"]
    router_config_hash = file_sha256(args.config)
    stage_mapping_hash = file_sha256(ROOT / config["stage_mapping"])
    latency_manifest_path = result_dir / "router_latency_manifest.json"
    latency_manifest = load(latency_manifest_path)
    if (
        latency_manifest.get("split") != "dev_policy"
        or latency_manifest.get("heldout_consulted") is not False
        or latency_manifest.get("operational_seed")
        != int(config["training"]["operational_seed"])
    ):
        raise ValueError("Invalid operational Router latency benchmark")
    router_latency_ms = float(latency_manifest["latency"]["mean_ms"])
    summary_path = ROOT / "results" / "phase4" / "2wiki" / "adaptive_rag_dev_summary.csv"
    report_path = ROOT / "docs" / "phase4" / "adaptive_rag_dev_analysis.md"
    gate_path = result_dir / "dev_gate_manifest.json"
    targets = [summary_path, report_path, gate_path]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Adaptive Router Dev Gate: {existing}")

    source_path = (
        ROOT / config["data"]["source_dir"] / "dev_policy.jsonl"
    )
    trajectories = trajectory_index(source_path, "dev_policy")
    rows = []
    run_sources = {}
    router_rows = []
    expected_seeds = [int(value) for value in config["training"]["seeds"]]
    for seed in expected_seeds:
        run_dir = ROOT / config["output"]["experiment_dir"] / f"seed{seed}"
        manifest_path = run_dir / "run_manifest.json"
        evaluation_path = run_dir / "evaluation" / "dev_policy" / "evaluation_manifest.json"
        prediction_path = run_dir / "evaluation" / "dev_policy" / "predictions.jsonl"
        manifest = load(manifest_path)
        evaluation = load(evaluation_path)
        if manifest.get("status") != "complete" or manifest.get("seed") != seed:
            raise ValueError(f"Incomplete Router seed {seed}")
        if manifest.get("heldout_consulted") is not False:
            raise ValueError(f"Router seed {seed} accessed heldout")
        if (
            manifest.get("router_config_sha256") != router_config_hash
            or manifest.get("stage_mapping_sha256") != stage_mapping_hash
            or manifest.get("model_input_fields") != ["question"]
        ):
            raise ValueError(f"Router seed {seed} does not match the frozen protocol")
        if evaluation.get("split") != "dev_policy" or evaluation.get("heldout_consulted") is not False:
            raise ValueError(f"Invalid Router dev_policy evaluation for seed {seed}")
        if evaluation.get("router_config_sha256") != router_config_hash:
            raise ValueError(f"Router evaluation config drift for seed {seed}")
        predictions = list(read_jsonl(prediction_path))
        predicted = {row["question_id"]: int(row["predicted_route_label"]) for row in predictions}
        metrics = {
            **evaluation["metrics"],
            **summarize_routes(trajectories, predicted),
        }
        row = system_row(config["formal_name"], seed, "three_seed_router", metrics)
        rows.append(row)
        router_rows.append(row)
        checkpoint_path = Path(manifest["checkpoint_path"])
        checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash != manifest["checkpoint_sha256"]:
            raise ValueError(f"Router checkpoint hash changed for seed {seed}")
        run_sources[f"seed{seed}"] = {
            "run_manifest": portable_path(manifest_path, ROOT),
            "run_manifest_sha256": file_sha256(manifest_path),
            "checkpoint": portable_path(checkpoint_path, ROOT),
            "checkpoint_sha256": checkpoint_hash,
            "dev_policy_evaluation": portable_path(evaluation_path, ROOT),
            "dev_policy_evaluation_sha256": file_sha256(evaluation_path),
            "dev_policy_predictions_sha256": file_sha256(prediction_path),
        }

    metric_names = [
        "accuracy",
        "macro_f1",
        "light_route_rate",
        "medium_route_rate",
        "heavy_route_rate",
        "average_final_stage",
        "average_retrieved_chunks",
        "average_unique_titles",
        "average_reranker_calls",
        "average_retrieval_latency_ms",
        "supporting_fact_recall",
        "complete_evidence_coverage",
        "terminal_retrieval_failure_rate",
        "recoverable_route_accuracy",
        "recoverable_under_retrieval_rate",
        "unrecoverable_average_chunks",
        "unrecoverable_average_reranker_calls",
        "unrecoverable_average_retrieval_latency_ms",
    ]
    aggregate = {
        "system": config["formal_name"],
        "seed": "mean",
        "role": "three_seed_mean",
    }
    aggregate_std = {
        "system": config["formal_name"],
        "seed": "std",
        "role": "three_seed_std",
    }
    for metric in metric_names:
        mean, std = mean_std(router_rows, metric)
        aggregate[metric] = mean
        aggregate_std[metric] = std
    aggregate["samples"] = router_rows[0]["samples"]
    aggregate_std["samples"] = router_rows[0]["samples"]
    aggregate["questions"] = router_rows[0]["questions"]
    aggregate_std["questions"] = router_rows[0]["questions"]
    aggregate["confusion_matrix"] = ""
    aggregate_std["confusion_matrix"] = ""
    aggregate["recoverable_questions"] = router_rows[0]["recoverable_questions"]
    aggregate_std["recoverable_questions"] = router_rows[0]["recoverable_questions"]
    aggregate["unrecoverable_questions"] = router_rows[0]["unrecoverable_questions"]
    aggregate_std["unrecoverable_questions"] = router_rows[0]["unrecoverable_questions"]
    aggregate["router_latency_ms"] = router_latency_ms
    aggregate["total_routing_retrieval_latency_ms"] = (
        aggregate["average_retrieval_latency_ms"] + router_latency_ms
    )
    aggregate_std["router_latency_ms"] = 0.0
    aggregate_std["total_routing_retrieval_latency_ms"] = aggregate_std[
        "average_retrieval_latency_ms"
    ]
    rows.extend([aggregate, aggregate_std])

    controller_path = (
        ROOT / "results" / "phase4" / "2wiki" / "controller_dev"
        / "calibrated_dev_policy_predictions.jsonl"
    )
    controller_rows = list(read_jsonl(controller_path))
    frozen = protocol["in_domain_policy"]["frozen_dev_gate"]
    selected = load(
        ROOT / "results" / "phase4" / "2wiki" / "controller_dev"
        / "selected_thresholds.json"
    )["primary_selected_policy"]
    comparisons = [
        (
            "EvidenceAwareClassification",
            "raw_stop_probability",
            float(frozen["classification_operating_point"]["threshold"]),
        ),
        (
            "EvidenceAwareRisk10",
            "stop_probability",
            float(selected["threshold"]),
        ),
    ]
    comparison_rows = []
    for name, probability_key, threshold in comparisons:
        predicted = controller_predictions(controller_rows, probability_key, threshold)
        metrics = summarize_routes(trajectories, predicted)
        row = system_row(name, 42, "frozen_evidence_aware", metrics)
        rows.append(row)
        comparison_rows.append(row)

    # Normalize sparse row schemas before writing one auditable table.
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    normalized = [{key: row.get(key) for key in fieldnames} for row in rows]
    write_csv(summary_path, normalized)

    mean_router = aggregate
    classification, risk10 = comparison_rows
    lines = [
        "# 2Wiki Adaptive-RAG-style Router Dev Analysis",
        "",
        "This is an adapted, question-only, pre-retrieval Router and not an official "
        "Adaptive-RAG reproduction. Training uses `train_core`, epoch selection uses "
        "`dev_calibration`, and this report evaluates `dev_policy` once. Heldout is not read.",
        "",
        "## Three-seed Router",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for metric in metric_names:
        lines.append(
            f"| {metric} | {float(mean_router[metric]):.6f} | "
            f"{float(aggregate_std[metric]):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Quality-Cost Comparison",
            "",
            "| System | Final depth | Chunks | Rerank calls | SF recall | Complete coverage | Retrieval latency ms | Recoverable under-retrieval |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [mean_router, classification, risk10]:
        lines.append(
            f"| {row['system']} | {float(row['average_final_stage']):.3f} | "
            f"{float(row['average_retrieved_chunks']):.3f} | "
            f"{float(row['average_reranker_calls']):.3f} | "
            f"{float(row['supporting_fact_recall']):.4f} | "
            f"{float(row['complete_evidence_coverage']):.4f} | "
            f"{float(row['average_retrieval_latency_ms']):.2f} | "
            f"{fmt(float(row['recoverable_under_retrieval_rate']))} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval-limited Diagnostics",
            "",
            f"- Frozen terminal retrieval failure rate: {fmt(float(mean_router['terminal_retrieval_failure_rate']))}.",
            f"- Router route accuracy on recoverable questions: {fmt(float(mean_router['recoverable_route_accuracy']))}.",
            f"- Router recoverable under-retrieval rate: {fmt(float(mean_router['recoverable_under_retrieval_rate']))}.",
            f"- Operational Router inference latency is {router_latency_ms:.2f} ms/question at synchronized batch size one; Router plus retrieval latency is {float(mean_router['total_routing_retrieval_latency_ms']):.2f} ms/question.",
            f"- On terminal-unrecoverable questions, the Router averages {float(mean_router['unrecoverable_average_chunks']):.3f} chunks and {float(mean_router['unrecoverable_average_reranker_calls']):.3f} reranker calls.",
            "",
            "## Research Questions",
            "",
            "- RQ-A is answered by the Router's average chunks, reranker calls, and latency relative to the frozen evidence-aware systems.",
            "- RQ-B is answered by the complete-coverage and supporting-fact-recall differences in the table.",
            "- RQ-C is answered by Recoverable Under-Retrieval Rate, not Controller FSR.",
            "- RQ-D is descriptive: the Router decides once from the query, whereas the proposed Controller observes current evidence and decides sequentially.",
            "",
            "No model, label construction, route mapping, threshold, or retrieval setting is changed from this Dev result.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    leakage_path = result_dir / "leakage_audit.json"
    dataset_manifest_path = result_dir / "dataset_manifest.json"
    leakage = load(leakage_path)
    if leakage.get("leakage_detected") is not False:
        raise ValueError("Adaptive Router leakage audit failed")
    operational_seed = int(config["training"]["operational_seed"])
    operational = run_sources[f"seed{operational_seed}"]
    write_json_atomic(
        gate_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "status": "frozen",
            "formal_name": config["formal_name"],
            "official_reproduction": False,
            "training_split": "train_core",
            "selection_split": "dev_calibration",
            "evaluation_split": "dev_policy",
            "heldout_consulted": False,
            "seeds": expected_seeds,
            "operational_seed": operational_seed,
            "prediction_rule": "three_class_argmax_no_threshold_sweep",
            "router_input": "question_only",
            "leakage_detected": False,
            "router_config": portable_path(args.config, ROOT),
            "router_config_sha256": router_config_hash,
            "protocol_config_sha256": file_sha256(protocol_path),
            "stage_mapping": config["stage_mapping"],
            "stage_mapping_sha256": stage_mapping_hash,
            "dataset_manifest": portable_path(dataset_manifest_path, ROOT),
            "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
            "leakage_audit": portable_path(leakage_path, ROOT),
            "leakage_audit_sha256": file_sha256(leakage_path),
            "router_latency_manifest": portable_path(latency_manifest_path, ROOT),
            "router_latency_manifest_sha256": file_sha256(latency_manifest_path),
            "operational_checkpoint_sha256": operational["checkpoint_sha256"],
            "run_sources": run_sources,
            "summary": portable_path(summary_path, ROOT),
            "summary_sha256": file_sha256(summary_path),
            "report": portable_path(report_path, ROOT),
            "report_sha256": file_sha256(report_path),
            "git_commit": git_commit(ROOT),
        },
    )
    print(report_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

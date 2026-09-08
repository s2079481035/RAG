"""Select point and conservative Phase 3B risk thresholds on dev_policy only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
    from phase2_controller_inputs import evidence_key
    from phase3b_metrics import (
        operational_metrics,
        rows_by_question,
        simulate_trajectory,
        threshold_values,
    )
except ModuleNotFoundError:  # Imported as scripts.select_phase3b_risk_policy in tests.
    from scripts.experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
    from scripts.phase2_controller_inputs import evidence_key
    from scripts.phase3b_metrics import (
        operational_metrics,
        rows_by_question,
        simulate_trajectory,
        threshold_values,
    )


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def estimated_retrieval_latency_ms(source_row: dict) -> float:
    values = source_row["latency_ms"]
    stage = source_row["stage"]
    dense = float(values["query_embedding_run_mean"]) + float(values["dense_search"])
    if stage == "dense@5":
        return dense
    hybrid = dense + float(values["bm25_scoring"]) + float(values["rrf_fusion"])
    if stage == "hybrid@10":
        return hybrid
    if stage == "rerank@20":
        return hybrid + float(values["reranker_allocated"])
    raise ValueError(f"Unknown stage for latency accounting: {stage}")


def attach_source_fields(predictions: list[dict], source_rows: list[dict], config: dict) -> list[dict]:
    source = {(row["question_id"], row["stage"]): row for row in source_rows}
    stage_config = {row["name"]: row for row in config["stages"]}
    merged = []
    mode = config["base_controller"]["evidence_mode"]
    for prediction in predictions:
        key = (prediction["question_id"], prediction["stage"])
        if key not in source:
            raise ValueError(f"Missing Controller source row: {key}")
        row = source[key]
        view = row[evidence_key(mode)]
        if int(prediction["actual_stop_label"]) != int(view["stop_label"]):
            raise ValueError(f"Controller label mismatch: {key}")
        merged.append(
            {
                **prediction,
                "stage_index": int(stage_config[row["stage"]]["index"]),
                "retrieved_chunks": len(view["items"]),
                "unique_titles": len(set(view["document_titles"])),
                "reranker_calls": int(stage_config[row["stage"]]["reranker_calls"]),
                "estimated_retrieval_latency_ms": estimated_retrieval_latency_ms(row),
                "evidence_chunk_ids": [item["chunk_id"] for item in view["items"]],
                "evidence_document_titles": list(view["document_titles"]),
            }
        )
    return merged


def summarize_trajectories(trajectories: list[dict], selected_rows: list[dict]) -> dict:
    return {
        "questions": len(trajectories),
        "stop_rate_before_final": sum(item["final_stage_index"] < 3 for item in trajectories)
        / len(trajectories),
        "average_final_stage": sum(item["final_stage_index"] for item in trajectories)
        / len(trajectories),
        "average_retrieved_chunks": sum(item["retrieved_chunks"] for item in selected_rows)
        / len(selected_rows),
        "average_unique_titles": sum(item["unique_titles"] for item in selected_rows)
        / len(selected_rows),
        "average_reranker_calls": sum(item["reranker_calls"] for item in selected_rows)
        / len(selected_rows),
        "average_controller_calls": sum(item["controller_calls"] for item in trajectories)
        / len(trajectories),
        "estimated_average_retrieval_latency_ms": sum(
            item["estimated_retrieval_latency_ms"] for item in selected_rows
        )
        / len(selected_rows),
        "final_supporting_fact_recall": sum(item["final_true_coverage"] for item in trajectories)
        / len(trajectories),
        "final_complete_evidence_coverage": sum(
            item["final_evidence_state"] == "sufficient" for item in trajectories
        )
        / len(trajectories),
    }


def bootstrap_fsr(
    grouped: dict[str, list[dict]],
    probability_key: str,
    thresholds: list[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    question_ids = sorted(grouped)
    draws = np.random.default_rng(seed).integers(
        0, len(question_ids), size=(replicates, len(question_ids))
    )
    values = {}
    for threshold in thresholds:
        false_by_question = []
        continue_by_question = []
        for qid in question_ids:
            false_stops = 0
            continue_labels = 0
            for row in grouped[qid][:-1]:
                prediction = int(float(row[probability_key]) >= threshold)
                actual = int(row["actual_stop_label"])
                continue_labels += int(actual == 0)
                false_stops += int(actual == 0 and prediction == 1)
                if prediction == 1:
                    break
            false_by_question.append(false_stops)
            continue_by_question.append(continue_labels)
        false_by_question = np.asarray(false_by_question, dtype=int)
        continue_by_question = np.asarray(continue_by_question, dtype=int)
        sampled_false = np.sum(false_by_question[draws], axis=1)
        sampled_denominators = np.sum(continue_by_question[draws], axis=1)
        fsr = np.divide(
            sampled_false,
            sampled_denominators,
            out=np.full(replicates, np.nan, dtype=float),
            where=sampled_denominators > 0,
        )
        values[threshold] = fsr
    return values, draws


def select_threshold(rows: list[dict], target: float, constraint_key: str) -> dict:
    eligible = [row for row in rows if float(row[constraint_key]) <= target]
    if not eligible:
        return {"status": "infeasible", "threshold": None}
    chosen = min(
        eligible,
        key=lambda row: (
            float(row["average_retrieved_chunks"]),
            float(row["estimated_average_retrieval_latency_ms"]),
            float(row["threshold"]),
        ),
    )
    return {
        "status": "selected",
        "threshold": float(chosen["threshold"]),
        "observed_false_stop_rate": float(chosen["false_stop_rate"]),
        "fsr_ci95_low": float(chosen["fsr_ci95_low"]),
        "fsr_ci95_high": float(chosen["fsr_ci95_high"]),
        "average_retrieved_chunks": float(chosen["average_retrieved_chunks"]),
        "estimated_average_retrieval_latency_ms": float(
            chosen["estimated_average_retrieval_latency_ms"]
        ),
        "average_reranker_calls": float(chosen["average_reranker_calls"]),
        "average_final_stage": float(chosen["average_final_stage"]),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = config["policy"]
    protocol_path = ROOT / "results" / "phase3b" / "protocol_manifest.json"
    calibration_manifest_path = ROOT / "results" / "phase3b" / "calibration_manifest.json"
    calibrated_path = ROOT / "results" / "phase3b" / "calibrated_dev_predictions.jsonl"
    if not all(path.exists() for path in [protocol_path, calibration_manifest_path, calibrated_path]):
        raise FileNotFoundError("Run Phase 3B preparation and calibration first")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    policy_ids_path = repo_path(config["dev_split"]["policy_ids"])
    if file_sha256(policy_ids_path) != protocol["dev_policy_ids_sha256"]:
        raise ValueError("Dev policy IDs changed after preparation")
    policy_ids = read_ids(policy_ids_path)
    predictions = [
        row for row in read_jsonl(calibrated_path)
        if row["phase3b_subset"] == "dev_policy"
    ]
    if {row["question_id"] for row in predictions} != policy_ids:
        raise ValueError("Calibrated dev_policy predictions do not match frozen IDs")
    source_path = repo_path(config["dev_split"]["source"])
    source_rows = [row for row in read_jsonl(source_path) if row["question_id"] in policy_ids]
    merged = attach_source_fields(predictions, source_rows, config)
    stage_order = [row["name"] for row in config["stages"]]
    grouped = rows_by_question(merged, stage_order)
    actionable_stages = set(policy["decision_stages"])
    actionable = [row for row in merged if row["stage"] in actionable_stages]
    if len(actionable) != len(policy_ids) * len(actionable_stages):
        raise ValueError("Actionable Dev decisions are incomplete")

    grid = policy["threshold_grid"]
    thresholds = threshold_values(float(grid["start"]), float(grid["stop"]), float(grid["step"]))
    bootstrap_config = policy["bootstrap"]
    output_dir = ROOT / "results" / "phase3b"
    sweep_path = output_dir / "dev_threshold_selection_sweep.csv"
    selected_path = output_dir / "selected_thresholds.json"
    trajectories_path = output_dir / "dev_policy_threshold_trajectories.jsonl"
    bootstrap_path = output_dir / "dev_policy_bootstrap_fsr.jsonl"
    manifest_path = output_dir / "policy_selection_manifest.json"
    targets = [sweep_path, selected_path, trajectories_path, bootstrap_path, manifest_path]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3B policy selection: {existing}")

    probability_keys = {
        "raw": "raw_stop_probability",
        "temperature": "temperature_stop_probability",
    }
    sweep_rows = []
    trajectory_rows = []
    bootstrap_rows = []
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
            trajectories = [
                simulate_trajectory(
                    grouped[qid], probability_key, threshold, include_evidence=False
                )
                for qid in sorted(grouped)
            ]
            row_lookup = {(row["question_id"], row["stage"]): row for row in merged}
            selected_rows = []
            for trajectory in trajectories:
                selected = row_lookup[(trajectory["question_id"], trajectory["final_stage"])]
                trajectory.update(
                    {
                        "calibration_method": method,
                        "threshold": threshold,
                        "retrieved_chunks": selected["retrieved_chunks"],
                        "unique_titles": selected["unique_titles"],
                        "reranker_calls": selected["reranker_calls"],
                        "estimated_retrieval_latency_ms": selected[
                            "estimated_retrieval_latency_ms"
                        ],
                    }
                )
                selected_rows.append(selected)
                trajectory_rows.append(trajectory)
            cost = summarize_trajectories(trajectories, selected_rows)
            valid = bootstrap_values[threshold][np.isfinite(bootstrap_values[threshold])]
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
                    "answer_em": None,
                    "answer_f1": None,
                    "latency_source": "phase2_batched_allocated_estimate_not_official_phase3b_latency",
                    "risk_evaluation_scope": "reached_actionable_controller_decisions",
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

    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sweep_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    temporary.replace(sweep_path)
    write_jsonl_atomic(trajectories_path, trajectory_rows)
    write_jsonl_atomic(bootstrap_path, bootstrap_rows)

    selections = []
    for method in probability_keys:
        method_rows = [row for row in sweep_rows if row["calibration_method"] == method]
        for target in [float(value) for value in policy["risk_targets"]]:
            selections.append(
                {
                    "calibration_method": method,
                    "selection": "point",
                    "risk_target": target,
                    **select_threshold(method_rows, target, "false_stop_rate"),
                }
            )
            selections.append(
                {
                    "calibration_method": method,
                    "selection": "conservative",
                    "risk_target": target,
                    **select_threshold(method_rows, target, "fsr_ci95_high"),
                }
            )
    write_json_atomic(
        selected_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "selection_split": "dev_policy",
            "test_consulted": False,
            "risk_definition": policy["risk_metric"],
            "decision_stages": policy["decision_stages"],
            "forced_final_stage": policy["forced_final_stage"],
            "selection_cost": policy["selection_cost"],
            "risk_targets": policy["risk_targets"],
            "primary_report_policy": policy["primary_report_policy"],
            "policies": selections,
        },
    )
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3B",
            "git_commit": git_commit(ROOT),
            "config": portable_path(args.config.resolve(), ROOT),
            "config_sha256": file_sha256(args.config),
            "split": "dev_policy",
            "questions": len(grouped),
            "actionable_stage_decisions": len(actionable),
            "test_consulted": False,
            "bootstrap_replicates": int(bootstrap_config["replicates"]),
            "bootstrap_unit": "question_id",
            "calibrated_predictions_sha256": file_sha256(calibrated_path),
            "outputs": {
                "threshold_sweep": portable_path(sweep_path, ROOT),
                "selected_thresholds": portable_path(selected_path, ROOT),
                "trajectories": portable_path(trajectories_path, ROOT),
                "bootstrap_fsr": portable_path(bootstrap_path, ROOT),
            },
        },
    )
    print(json.dumps(selections, indent=2))


if __name__ == "__main__":
    main()

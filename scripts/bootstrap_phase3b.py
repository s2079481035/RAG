"""Paired question-level bootstrap for the frozen primary Phase 3B Test policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from generate_phase3b_stage_answers import validate_test_gate
from train_phase2_controller import read_jsonl, write_jsonl_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"


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


def fsr(rows: list[dict]) -> float:
    denominator = sum(int(row["actionable_continue_count"]) for row in rows)
    numerator = sum(int(row["false_stop_count"]) for row in rows)
    return numerator / denominator if denominator else float("nan")


def metrics(rows: list[dict]) -> dict:
    return {
        "answer_f1": float(np.mean([float(row["answer_f1"]) for row in rows])),
        "retrieval_cost": float(np.mean([float(row["retrieved_chunks"]) for row in rows])),
        "latency_ms": float(np.mean([float(row["total_latency_ms"]) for row in rows])),
        "false_stop_rate": fsr(rows),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_test_gate(config, args.config)
    trajectory_path = ROOT / "results" / "phase3b" / "test_policy_trajectories.jsonl"
    evaluation_manifest_path = ROOT / "results" / "phase3b" / "test_evaluation_manifest.json"
    if not trajectory_path.exists() or not evaluation_manifest_path.exists():
        raise FileNotFoundError("Run frozen Phase 3B Test policy evaluation first")
    rows = read_jsonl(trajectory_path)
    by_policy = defaultdict(dict)
    for row in rows:
        qid = row["question_id"]
        if qid in by_policy[row["policy"]]:
            raise ValueError(f"Duplicate trajectory for {row['policy']} {qid}")
        by_policy[row["policy"]][qid] = row
    primary = config["policy"]["primary_report_policy"]
    primary_name = (
        f"{primary['calibration']}_{primary['selection']}_risk_"
        f"{int(round(float(primary['risk_target']) * 100)):02d}"
    )
    baseline_name = "fixed_heavy"
    if primary_name not in by_policy or baseline_name not in by_policy:
        raise ValueError("Primary or Fixed Heavy policy is missing from Test trajectories")
    question_ids = sorted(by_policy[primary_name])
    if set(question_ids) != set(by_policy[baseline_name]):
        raise ValueError("Paired policies cover different Test questions")
    bootstrap = config["policy"]["bootstrap"]
    replicates = int(bootstrap["replicates"])
    draws = np.random.default_rng(int(bootstrap["seed"])).integers(
        0, len(question_ids), size=(replicates, len(question_ids))
    )
    baseline_rows = [by_policy[baseline_name][qid] for qid in question_ids]
    primary_rows = [by_policy[primary_name][qid] for qid in question_ids]
    observed_baseline = metrics(baseline_rows)
    observed_primary = metrics(primary_rows)
    observed_delta = {
        key: observed_primary[key] - observed_baseline[key] for key in observed_primary
    }

    replicate_rows = []
    values = {key: [] for key in observed_delta}
    for replicate, indices in enumerate(draws):
        sampled_baseline = [baseline_rows[int(index)] for index in indices]
        sampled_primary = [primary_rows[int(index)] for index in indices]
        baseline_metrics = metrics(sampled_baseline)
        primary_metrics = metrics(sampled_primary)
        for metric in values:
            delta = primary_metrics[metric] - baseline_metrics[metric]
            values[metric].append(delta)
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "comparison": f"{primary_name}_minus_{baseline_name}",
                    "metric": metric,
                    "baseline": baseline_metrics[metric],
                    "model": primary_metrics[metric],
                    "delta": delta,
                    "sampling_unit": "question_id",
                }
            )
    summary = []
    for metric, metric_values in values.items():
        low, high = np.quantile(np.asarray(metric_values), [0.025, 0.975])
        summary.append(
            {
                "comparison": f"{primary_name}_minus_{baseline_name}",
                "metric": metric,
                "observed_baseline": observed_baseline[metric],
                "observed_model": observed_primary[metric],
                "observed_delta": observed_delta[metric],
                "bootstrap_mean_delta": float(np.mean(metric_values)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                "replicates": replicates,
                "sampling_unit": "question_id",
            }
        )

    output_dir = ROOT / "results" / "phase3b"
    summary_path = output_dir / "bootstrap_ci.csv"
    replicates_path = output_dir / "bootstrap_replicates.jsonl"
    manifest_path = output_dir / "bootstrap_manifest.json"
    existing = [path for path in [summary_path, replicates_path, manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3B bootstrap: {existing}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    temporary.replace(summary_path)
    write_jsonl_atomic(replicates_path, replicate_rows)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3B",
            "git_commit": git_commit(ROOT),
            "split": "test",
            "comparison": f"{primary_name}_minus_{baseline_name}",
            "questions": len(question_ids),
            "replicates": replicates,
            "sampling_unit": "question_id",
            "test_threshold_refit": False,
            "test_calibration_refit": False,
            "prior_test_exposure": config["prior_test_exposure"],
            "source_trajectories": portable_path(trajectory_path, ROOT),
            "source_trajectories_sha256": file_sha256(trajectory_path),
            "outputs": {
                "summary": portable_path(summary_path, ROOT),
                "replicates": portable_path(replicates_path, ROOT),
            },
        },
    )
    print(f"saved {replicates} paired question-level bootstrap replicates")


if __name__ == "__main__":
    main()

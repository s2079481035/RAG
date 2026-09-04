"""Select the final Phase 3A Controller from multi-seed Dev results only."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from experiment_utils import portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3a" / "controller.json"
DEFAULT_OUTPUT = ROOT / "results" / "phase3" / "final_controller_selection.json"
DEFAULT_LAMBDA_SELECTION = ROOT / "results" / "phase3" / "coverage_lambda_selection.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--lambda-selection", type=Path, default=DEFAULT_LAMBDA_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def candidate_group(manifest: dict) -> str:
    auxiliary = bool(manifest["coverage_auxiliary"])
    hard_sampling = manifest["sampling"] == "hard_partial_aware"
    if manifest["baseline"] != "query_evidence" or manifest["representation"] != "score_aware_packing":
        raise ValueError("Final selection candidates must all be Query+Evidence score-aware")
    if not auxiliary and manifest["sampling"] == "natural":
        return "score_aware_baseline"
    if auxiliary and manifest["sampling"] == "natural":
        return "score_aware_coverage_auxiliary"
    if not auxiliary and hard_sampling:
        return "score_aware_hard_partial_sampling"
    if auxiliary and hard_sampling:
        return "score_aware_coverage_auxiliary_hard_partial_sampling"
    raise ValueError(f"Run is not a final-selection ablation candidate: {manifest}")


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite final Controller selection: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lambda_selection = json.loads(args.lambda_selection.read_text(encoding="utf-8"))
    if lambda_selection.get("selection_split") != "dev" or lambda_selection.get("test_consulted"):
        raise ValueError("Coverage lambda was not selected exclusively on Dev")
    selected_lambda = float(lambda_selection["selected_coverage_lambda"])
    selection = config["final_controller_selection"]
    expected_groups = set(selection["candidate_groups"])
    expected_seeds = {int(value) for value in selection["required_seeds"]}
    grouped = defaultdict(list)
    for raw_run in args.runs:
        run = raw_run.resolve()
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "dev_metrics.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete final-selection run: {run}")
        if (run / "evaluation" / "test").exists():
            raise ValueError(f"Test output exists before final Controller selection: {run}")
        if manifest["coverage_auxiliary"] and float(manifest["coverage_lambda"]) != selected_lambda:
            raise ValueError(f"Auxiliary run does not use the Dev-selected lambda {selected_lambda}: {run}")
        grouped[candidate_group(manifest)].append(
            {
                "seed": int(manifest["seed"]),
                "run_dir": portable_path(run, ROOT),
                "coverage_lambda": float(manifest["coverage_lambda"]),
                "sampling": manifest["sampling"],
                "macro_f1": float(metrics["macro_f1"]),
                "hard_partial_false_stop_rate": float(metrics["hard_partial_false_stop_rate"]),
                "auroc": float(metrics["auroc"]),
                "unnecessary_escalation_rate": float(metrics["unnecessary_escalation_rate"]),
            }
        )
    if set(grouped) != expected_groups:
        raise ValueError(f"Expected candidate groups {sorted(expected_groups)}, got {sorted(grouped)}")
    summaries = []
    for order, name in enumerate(selection["candidate_groups"]):
        rows = grouped[name]
        seeds = {row["seed"] for row in rows}
        if seeds != expected_seeds:
            raise ValueError(f"Candidate {name} has seeds {sorted(seeds)}, expected {sorted(expected_seeds)}")
        summaries.append(
            {
                "group": name,
                "complexity_tiebreak_order": order,
                "mean_macro_f1": statistics.fmean(row["macro_f1"] for row in rows),
                "mean_hard_partial_false_stop_rate": statistics.fmean(
                    row["hard_partial_false_stop_rate"] for row in rows
                ),
                "mean_auroc": statistics.fmean(row["auroc"] for row in rows),
                "mean_unnecessary_escalation_rate": statistics.fmean(
                    row["unnecessary_escalation_rate"] for row in rows
                ),
                "runs": sorted(rows, key=lambda row: row["seed"]),
            }
        )
    ranked = sorted(
        summaries,
        key=lambda row: (
            -row["mean_macro_f1"],
            row["mean_hard_partial_false_stop_rate"],
            -row["mean_auroc"],
            row["mean_unnecessary_escalation_rate"],
            row["complexity_tiebreak_order"],
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "selection_split": "dev",
            "test_consulted": False,
            "required_seeds": sorted(expected_seeds),
            "selection_rule": selection["selection_rule"],
            "coverage_lambda_selection": portable_path(args.lambda_selection.resolve(), ROOT),
            "selected_coverage_lambda": selected_lambda,
            "candidates_ranked": ranked,
            "selected_group": ranked[0]["group"],
            "selected_runs": ranked[0]["runs"],
        },
    )
    print(ranked[0]["group"])


if __name__ == "__main__":
    main()

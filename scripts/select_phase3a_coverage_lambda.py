"""Select the coverage-auxiliary lambda using completed dev runs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_utils import portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3a" / "controller.json"
DEFAULT_OUTPUT = ROOT / "results" / "phase3" / "coverage_lambda_selection.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite lambda selection: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_lambdas = {float(value) for value in config["coverage_auxiliary"]["candidate_lambdas"]}
    expected_seed = int(config["coverage_auxiliary"]["selection_seed"])
    candidates = []
    seen_lambdas = set()
    for raw_run in args.runs:
        run = raw_run.resolve()
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "dev_metrics.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete lambda run: {run}")
        if (run / "evaluation" / "test").exists():
            raise ValueError(f"Test output exists before lambda selection: {run}")
        if manifest["seed"] != expected_seed:
            raise ValueError(f"Lambda selection must use seed {expected_seed}: {run}")
        if manifest["baseline"] != "query_evidence" or manifest["representation"] != "score_aware_packing":
            raise ValueError(f"Lambda candidate is not Query+Evidence score-aware: {run}")
        if manifest["sampling"] != "natural" or not manifest["coverage_auxiliary"]:
            raise ValueError(f"Lambda candidate must use natural sampling and auxiliary loss: {run}")
        value = float(manifest["coverage_lambda"])
        if value in seen_lambdas:
            raise ValueError(f"Duplicate lambda candidate: {value}")
        seen_lambdas.add(value)
        candidates.append(
            {
                "coverage_lambda": value,
                "run_dir": portable_path(run, ROOT),
                "macro_f1": metrics["macro_f1"],
                "hard_partial_false_stop_rate": metrics["hard_partial_false_stop_rate"],
                "auroc": metrics["auroc"],
                "stop_f1": metrics["stop_f1"],
                "false_stop_rate": metrics["false_stop_rate"],
                "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
                "coverage_mae": metrics["coverage_regression"]["mae"],
                "coverage_spearman": metrics["coverage_regression"]["spearman"],
            }
        )
    if seen_lambdas != expected_lambdas:
        raise ValueError(
            f"Expected lambda candidates {sorted(expected_lambdas)}, got {sorted(seen_lambdas)}"
        )
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["macro_f1"]),
            float(row["hard_partial_false_stop_rate"]),
            -float(row["auroc"]),
            float(row["coverage_lambda"]),
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
            "selection_seed": expected_seed,
            "selection_rule": config["coverage_auxiliary"]["selection_rule"],
            "candidates_ranked": ranked,
            "selected_coverage_lambda": ranked[0]["coverage_lambda"],
        },
    )
    print(ranked[0]["coverage_lambda"])


if __name__ == "__main__":
    main()

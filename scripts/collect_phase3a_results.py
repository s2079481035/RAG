"""Collect multi-seed Phase 3A Controller stability and ablation results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from experiment_utils import portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3a" / "controller.json"
DEFAULT_OUTPUT = ROOT / "results" / "phase3"
METRICS = [
    "macro_f1",
    "stop_f1",
    "auroc",
    "false_stop_rate",
    "hard_partial_false_stop_rate",
    "unnecessary_escalation_rate",
    "coverage_mae",
    "coverage_spearman",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def group_name(manifest: dict) -> str:
    if manifest["baseline"] == "query_stage":
        return "query_stage"
    if manifest["representation"] == "concat_truncate":
        return "query_evidence_concat"
    auxiliary = bool(manifest["coverage_auxiliary"])
    sampling = manifest["sampling"]
    if not auxiliary and sampling == "natural":
        return "score_aware_baseline"
    if auxiliary and sampling == "natural":
        return f"score_aware_coverage_aux_lambda_{manifest['coverage_lambda']:g}"
    if not auxiliary:
        return f"score_aware_{sampling}"
    return f"score_aware_coverage_aux_lambda_{manifest['coverage_lambda']:g}_{sampling}"


def protocol_signature(config: dict, manifest: dict) -> dict:
    return {
        "variant": manifest["variant"],
        "evidence_mode": manifest["evidence_mode"],
        "backbone": config["resolved"]["backbone"],
        "max_length": config["backbone"]["max_length"],
        "optimizer": config["training"]["optimizer"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_ratio": config["training"]["warmup_ratio"],
        "selection_metric": config["training"]["selection_metric"],
        "threshold_grid": config["training"]["threshold_grid"],
    }


def main() -> None:
    args = parse_args()
    experiment_config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_seeds = {int(seed) for seed in experiment_config["seeds"]}
    output_dir = args.output_dir.resolve()
    targets = [
        output_dir / "seed_results.csv",
        output_dir / "multi_seed_summary.csv",
        output_dir / "ablation_summary.csv",
        output_dir / "sampling_summary.csv",
        output_dir / "phase3a_controller_analysis.md",
        output_dir / "collection_manifest.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3A collection: {existing}")

    seed_rows = []
    signatures = []
    sources = []
    for raw_run in args.runs:
        run = raw_run.resolve()
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        resolved = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
        metrics_path = run / "evaluation" / "test" / "original_metrics.json"
        predictions_path = run / "evaluation" / "test" / "original_predictions.jsonl"
        if manifest["status"] != "complete" or not metrics_path.exists() or not predictions_path.exists():
            raise ValueError(f"Run lacks completed training/test predictions: {run}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        coverage = metrics.get("coverage_regression", {})
        row = {
            "group": group_name(manifest),
            "seed": manifest["seed"],
            "baseline": manifest["baseline"],
            "representation": manifest["representation"],
            "coverage_lambda": manifest["coverage_lambda"],
            "sampling": manifest["sampling"],
            "threshold_selected_on_dev": manifest["selected_dev_threshold"],
            "macro_f1": metrics["macro_f1"],
            "stop_f1": metrics["stop_f1"],
            "auroc": metrics["auroc"],
            "false_stop_rate": metrics["false_stop_rate"],
            "hard_partial_false_stop_rate": metrics["hard_partial_false_stop_rate"],
            "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
            "coverage_mae": coverage.get("mae"),
            "coverage_spearman": coverage.get("spearman"),
            "run_dir": portable_path(run, ROOT),
            "predictions": portable_path(predictions_path, ROOT),
        }
        seed_rows.append(row)
        signatures.append(protocol_signature(resolved, manifest))
        sources.append(portable_path(run, ROOT))
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Core training protocol differs across Phase 3A runs")

    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row["group"]].append(row)
    required_core = {"query_stage", "query_evidence_concat", "score_aware_baseline"}
    missing_core = required_core - set(grouped)
    if missing_core:
        raise ValueError(f"Missing core multi-seed groups: {sorted(missing_core)}")
    for name in required_core:
        seeds = {int(row["seed"]) for row in grouped[name]}
        if seeds != expected_seeds:
            raise ValueError(f"Core group {name} has seeds {sorted(seeds)}, expected {sorted(expected_seeds)}")

    summary_rows = []
    for name, rows in sorted(grouped.items()):
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row[metric] not in (None, "")]
            summary_rows.append(
                {
                    "group": name,
                    "metric": metric,
                    "mean": statistics.fmean(values) if values else None,
                    "std_sample": statistics.stdev(values) if len(values) > 1 else None,
                    "seeds": len(values),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(output_dir / "seed_results.csv", seed_rows)
    write_csv_atomic(output_dir / "multi_seed_summary.csv", summary_rows)
    ablation_groups = {
        name for name in grouped
        if name == "score_aware_baseline"
        or name == "score_aware_hard_partial_aware"
        or name.startswith("score_aware_coverage_aux_lambda_")
    }
    ablation_rows = [row for row in summary_rows if row["group"] in ablation_groups]
    write_csv_atomic(output_dir / "ablation_summary.csv", ablation_rows)
    sampling_groups = {
        "score_aware_baseline",
        "score_aware_balanced_stop_continue",
        "score_aware_hard_partial_aware",
    }
    sampling_rows = [row for row in summary_rows if row["group"] in sampling_groups]
    write_csv_atomic(output_dir / "sampling_summary.csv", sampling_rows)
    lines = [
        "# Phase 3A Controller Analysis",
        "",
        "All test metrics use thresholds selected on dev. Standard deviation is the sample standard deviation across training seeds.",
        "",
        "| Group | Metric | Mean | Std | Seeds |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        mean = "N/A" if row["mean"] is None else f"{row['mean']:.4f}"
        std = "N/A" if row["std_sample"] is None else f"{row['std_sample']:.4f}"
        lines.append(f"| {row['group']} | {row['metric']} | {mean} | {std} | {row['seeds']} |")
    lines.extend(
        [
            "",
            "Hard Partial results must be interpreted with overall Macro F1 and unnecessary escalation rate; a lower False Stop Rate alone can result from overpredicting Continue.",
            "",
        ]
    )
    (output_dir / "phase3a_controller_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    write_json_atomic(
        output_dir / "collection_manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "split": "test",
            "test_threshold_refit": False,
            "expected_core_seeds": sorted(expected_seeds),
            "protocol_signature": signatures[0],
            "runs": sources,
        },
    )


if __name__ == "__main__":
    main()

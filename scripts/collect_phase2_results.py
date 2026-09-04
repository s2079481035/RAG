"""Collect completed Phase 2 runs into paper-facing result tables and analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from critic_metrics import binary_stop_metrics
from experiment_utils import portable_path, utc_now, write_json_atomic
from phase2_packing import gold_fact_bucket, length_bucket


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_run(path: Path, split: str) -> dict:
    run_dir = path.resolve()
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete run: {run_dir}")
    metrics_path = run_dir / "evaluation" / split / "original_metrics.json"
    predictions_path = run_dir / "evaluation" / split / "original_predictions.jsonl"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Run has no {split} original evaluation: {run_dir}")
    return {
        "run_dir": run_dir,
        "config": config,
        "resolved": config["resolved"],
        "manifest": manifest,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "predictions": read_jsonl(predictions_path),
    }


def metric_row(run: dict, split: str) -> dict:
    metrics = run["metrics"]
    resolved = run["resolved"]
    packing = metrics.get("packing", {})
    return {
        "baseline": resolved["baseline"],
        "representation": resolved["representation"],
        "evidence_mode": resolved["evidence_mode"],
        "variant": resolved["variant"],
        "split": split,
        "samples": len(run["predictions"]),
        "threshold_selected_on_dev": run["manifest"]["selected_dev_threshold"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "stop_precision": metrics["stop_precision"],
        "stop_recall": metrics["stop_recall"],
        "stop_f1": metrics["stop_f1"],
        "auroc": metrics["auroc"],
        "false_stop_rate": metrics["false_stop_rate"],
        "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
        "partial_to_stop_false_count": metrics.get("false_stop_by_three_class_label", {}).get(
            "partial", 0
        ),
        "truncation_ratio": packing.get("truncation_ratio"),
        "mean_visible_evidence_chunk_ratio": packing.get(
            "mean_visible_evidence_chunk_ratio",
            metrics.get("mean_visible_evidence_chunk_ratio"),
        ),
        "mean_fully_visible_evidence_chunk_ratio": packing.get(
            "mean_fully_visible_evidence_chunk_ratio"
        ),
        "mean_visible_supporting_fact_ratio_evaluation_only": packing.get(
            "mean_visible_supporting_fact_ratio_evaluation_only"
        ),
        "run_dir": portable_path(run["run_dir"], ROOT),
    }


def raw_tokens(prediction: dict) -> int | None:
    if prediction.get("raw_evidence_tokens") is not None:
        return int(prediction["raw_evidence_tokens"])
    audit = prediction.get("packing_audit")
    return int(audit["raw_evidence_tokens"]) if audit else None


def bucket_metrics(predictions: list[dict], threshold: float) -> dict:
    labels = [item["actual_stop_label"] for item in predictions]
    probabilities = [item["stop_probability"] for item in predictions]
    metrics, _ = binary_stop_metrics(labels, probabilities, threshold)
    return metrics


def build_length_rows(runs: list[dict], split: str) -> list[dict]:
    rows = []
    order = ["<=512", "513-1024", "1025-2048", ">2048"]
    for run in runs:
        if run["resolved"]["baseline"] not in {"evidence_only", "query_evidence"}:
            continue
        grouped = defaultdict(list)
        for prediction in run["predictions"]:
            tokens = raw_tokens(prediction)
            if tokens is not None:
                grouped[length_bucket(tokens)].append(prediction)
        for bucket in order:
            current = grouped[bucket]
            if not current:
                continue
            metrics = bucket_metrics(current, float(run["manifest"]["selected_dev_threshold"]))
            rows.append(
                {
                    "baseline": run["resolved"]["baseline"],
                    "representation": run["resolved"]["representation"],
                    "evidence_mode": run["resolved"]["evidence_mode"],
                    "split": split,
                    "length_bucket": bucket,
                    "samples": len(current),
                    "stop_f1": metrics["stop_f1"],
                    "false_stop_rate": metrics["false_stop_rate"],
                    "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
                    "auroc": metrics["auroc"],
                }
            )
    return rows


def build_support_fact_rows(runs: list[dict], split: str) -> list[dict]:
    rows = []
    source_cache = {}
    for run in runs:
        resolved = run["resolved"]
        variant = resolved["variant"]
        if variant not in source_cache:
            source_path = ROOT / "data" / "phase2" / "controller" / variant / f"{split}.jsonl"
            source_cache[variant] = {
                (item["question_id"], item["stage"]): item for item in read_jsonl(source_path)
            }
        source = source_cache[variant]
        grouped = defaultdict(list)
        for prediction in run["predictions"]:
            grouped[gold_fact_bucket(int(prediction["gold_supporting_fact_count"]))].append(
                prediction
            )
        for bucket in ["2", "3", "4+"]:
            current = grouped[bucket]
            if not current:
                continue
            metrics = bucket_metrics(current, float(run["manifest"]["selected_dev_threshold"]))
            recalls = []
            complete = []
            view_key = (
                "raw_stage_evidence"
                if resolved["evidence_mode"] == "raw"
                else "cumulative_evidence_memory"
            )
            for prediction in current:
                record = source[(prediction["question_id"], prediction["stage"])]
                view = record[view_key]
                recalls.append(view["supporting_fact_recall"])
                complete.append(view["complete_evidence_coverage"])
            rows.append(
                {
                    "baseline": resolved["baseline"],
                    "representation": resolved["representation"],
                    "evidence_mode": resolved["evidence_mode"],
                    "split": split,
                    "gold_supporting_fact_bucket": bucket,
                    "samples": len(current),
                    "supporting_fact_recall": sum(recalls) / len(recalls),
                    "complete_evidence_coverage": sum(complete) / len(complete),
                    "stop_f1": metrics["stop_f1"],
                    "false_stop_rate": metrics["false_stop_rate"],
                    "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
                    "auroc": metrics["auroc"],
                }
            )
    return rows


def build_counterfactual_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        if run["resolved"]["baseline"] not in {"evidence_only", "query_evidence"}:
            continue
        dev_dir = run["run_dir"] / "evaluation" / "dev"
        for condition in [
            "original",
            "evidence_order_shuffle",
            "cross_question_evidence_swap",
            "stage_metadata_removal",
            "title_only_evidence",
        ]:
            path = dev_dir / f"{condition}_metrics.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "baseline": run["resolved"]["baseline"],
                    "representation": run["resolved"]["representation"],
                    "evidence_mode": run["resolved"]["evidence_mode"],
                    "split": "dev",
                    "condition": condition,
                    "samples": metrics["actual_continue_count"] + metrics["actual_stop_count"],
                    "stop_f1": metrics["stop_f1"],
                    "auroc": metrics["auroc"],
                    "false_stop_rate": metrics["false_stop_rate"],
                    "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
                }
            )
    if not rows:
        raise FileNotFoundError("No dev counterfactual metrics found in the supplied evidence runs")
    return rows


def result_lookup(
    rows: list[dict],
    baseline: str,
    representation: str | None = None,
    evidence_mode: str | None = None,
):
    candidates = [row for row in rows if row["baseline"] == baseline]
    if representation is not None:
        candidates = [row for row in candidates if row["representation"] == representation]
    if evidence_mode is not None:
        candidates = [row for row in candidates if row["evidence_mode"] == evidence_mode]
    return max(candidates, key=lambda row: row["stop_f1"]) if candidates else None


def render_analysis(
    controller_rows: list[dict],
    length_rows: list[dict],
    support_rows: list[dict],
    trajectory: dict,
    split: str,
) -> str:
    best_evidence = max(
        (
            row
            for row in controller_rows
            if row["baseline"] == "query_evidence" and row["evidence_mode"] == "cumulative"
        ),
        key=lambda row: row["stop_f1"],
    )
    query_stage = result_lookup(controller_rows, "query_stage")
    query_stats = result_lookup(controller_rows, "query_stage_stats")
    concat = result_lookup(
        controller_rows, "query_evidence", "concat_truncate", "cumulative"
    )
    alternatives = [
        row
        for row in controller_rows
        if row["baseline"] == "query_evidence"
        and row["evidence_mode"] == "cumulative"
        and row["representation"] != "concat_truncate"
    ]
    best_alternative = max(alternatives, key=lambda row: row["stop_f1"]) if alternatives else None
    lines = [
        "# Phase 2 Analysis",
        "",
        f"Main Controller, packing, length, and supporting-fact results below are read from completed `{split}` prediction files. Counterfactual diagnostics are Dev-only. Thresholds were selected on Dev and were not refit on Test.",
        "",
        "## RQ1: Evidence beyond stage shortcuts",
        "",
    ]
    for baseline_name, row in [("Query+Stage", query_stage), ("Query+Stage+Stats", query_stats)]:
        if row is None:
            lines.append(f"- {baseline_name}: required run missing; no conclusion.")
        else:
            delta = best_evidence["stop_f1"] - row["stop_f1"]
            conclusion = "descriptively supports" if delta > 0 else "does not support"
            lines.append(
                f"- Best Query+Evidence Stop F1={best_evidence['stop_f1']:.4f} vs {baseline_name}={row['stop_f1']:.4f} (delta={delta:+.4f}); this {conclusion} added evidence value. No significance test is implied."
            )
    lines.extend(["", "## RQ2: 512-token truncation", ""])
    if concat is None:
        lines.append("Concat-Truncate run missing; no conclusion.")
    else:
        concat_buckets = [
            row
            for row in length_rows
            if row["baseline"] == "query_evidence"
            and row["representation"] == "concat_truncate"
        ]
        lines.append(
            f"Concat-Truncate overall False Stop Rate={concat['false_stop_rate']:.4f}. Length-bucket results are reported in `results/phase2/length_bucket_analysis.csv`; they establish association, not causal attribution."
        )
        for row in concat_buckets:
            lines.append(
                f"- `{row['length_bucket']}`: n={row['samples']}, Stop F1={row['stop_f1']:.4f}, FSR={row['false_stop_rate']:.4f}."
            )
    lines.extend(["", "## RQ3: Chunk-level label distribution", ""])
    selected_split = trajectory["by_split"].get(split)
    if selected_split:
        for mode in ["raw", "cumulative"]:
            rendered = []
            for stage, counts in selected_split["labels"][mode].items():
                rendered.append(
                    f"{stage}: I={counts.get('insufficient', 0)}, P={counts.get('partial', 0)}, S={counts.get('sufficient', 0)}"
                )
            lines.append(f"- {mode}: " + "; ".join(rendered) + ".")
    else:
        lines.append(f"No trajectory distribution was built for `{split}`.")
    lines.extend(["", "## RQ4: Representation", ""])
    if concat and best_alternative:
        delta = best_alternative["false_stop_rate"] - concat["false_stop_rate"]
        lines.append(
            f"Best non-concat representation is `{best_alternative['representation']}`. Its FSR={best_alternative['false_stop_rate']:.4f} vs Concat={concat['false_stop_rate']:.4f} (delta={delta:+.4f}); Partial-to-Stop false counts are in the packing table."
        )
    else:
        lines.append("A Concat run and at least one packing/hierarchical run are required for comparison.")
    lines.extend(["", "## RQ5: Cumulative evidence", ""])
    if selected_split:
        for mode in ["raw", "cumulative"]:
            values = selected_split["trajectory_monotonicity"][mode]
            lines.append(
                f"- {mode}: {values['non_monotonic']}/{values['questions']} non-monotonic trajectories ({values['ratio']:.6f})."
            )
    lines.extend(["", "## RQ6: Number of gold facts", ""])
    best_support = [
        row
        for row in support_rows
        if row["baseline"] == best_evidence["baseline"]
        and row["representation"] == best_evidence["representation"]
        and row["evidence_mode"] == best_evidence["evidence_mode"]
    ]
    for row in best_support:
        lines.append(
            f"- `{row['gold_supporting_fact_bucket']}` facts: n={row['samples']}, SF Recall={row['supporting_fact_recall']:.4f}, complete={row['complete_evidence_coverage']:.4f}, Stop F1={row['stop_f1']:.4f}, FSR={row['false_stop_rate']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Go / No-Go",
            "",
            "The tables support only descriptive Go/No-Go decisions. Confirm practical effect sizes across Query+Stage, Query+Stage+Stats, counterfactual swap, long-evidence buckets, and manual label audit before choosing a Phase 3 method. Risk calibration remains intentionally out of scope.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = [load_run(path, args.split) for path in args.runs]
    variants = {run["resolved"]["variant"] for run in runs}
    if len(variants) != 1:
        raise ValueError(f"Controller runs use different chunk variants: {variants}")
    required_baselines = {"query_only", "query_stage", "query_stage_stats", "evidence_only", "query_evidence"}
    present = {run["resolved"]["baseline"] for run in runs}
    missing = required_baselines - present
    if missing:
        raise ValueError(f"Missing required fair-baseline runs: {sorted(missing)}")

    output_dir = ROOT / "results" / "phase2"
    paths = {
        "retrieval": output_dir / "retrieval_summary.csv",
        "controller": output_dir / "controller_baselines.csv",
        "packing": output_dir / "packing_comparison.csv",
        "length": output_dir / "length_bucket_analysis.csv",
        "support": output_dir / "support_fact_bucket_analysis.csv",
        "counterfactual": output_dir / "counterfactual_analysis.csv",
        "analysis": ROOT / "docs" / "phase2_analysis.md",
        "manifest": output_dir / "collection_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 2 summaries: {existing}")

    controller_rows = [metric_row(run, args.split) for run in runs]
    packing_rows = [
        row
        for row in controller_rows
        if row["baseline"] in {"evidence_only", "query_evidence"}
    ]
    length_rows = build_length_rows(runs, args.split)
    support_rows = build_support_fact_rows(runs, args.split)
    counterfactual_rows = build_counterfactual_rows(runs)
    write_csv_atomic(paths["controller"], controller_rows)
    write_csv_atomic(paths["packing"], packing_rows)
    write_csv_atomic(paths["length"], length_rows)
    write_csv_atomic(paths["support"], support_rows)
    write_csv_atomic(paths["counterfactual"], counterfactual_rows)

    selected_variant = next(iter(variants))
    retrieval_source = ROOT / "results" / "phase2" / "retrieval_eval" / args.split / "retrieval_summary.csv"
    with retrieval_source.open(encoding="utf-8", newline="") as handle:
        retrieval_rows = [
            row for row in csv.DictReader(handle) if row["variant"] == selected_variant
        ]
    write_csv_atomic(paths["retrieval"], retrieval_rows)

    trajectory_candidates = sorted(
        (ROOT / "data" / "phase2" / "controller" / selected_variant).glob(
            "label_distribution_*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not trajectory_candidates:
        raise FileNotFoundError("Controller label distribution is missing")
    trajectory = json.loads(trajectory_candidates[0].read_text(encoding="utf-8"))
    paths["analysis"].write_text(
        render_analysis(controller_rows, length_rows, support_rows, trajectory, args.split),
        encoding="utf-8",
    )
    write_json_atomic(
        paths["manifest"],
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "split": args.split,
            "counterfactual_split": "dev",
            "variant": selected_variant,
            "runs": [portable_path(run["run_dir"], ROOT) for run in runs],
            "retrieval_source": portable_path(retrieval_source, ROOT),
            "trajectory_source": portable_path(trajectory_candidates[0], ROOT),
            "outputs": {name: portable_path(path, ROOT) for name, path in paths.items()},
        },
    )


if __name__ == "__main__":
    main()

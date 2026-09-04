"""Analyze false stops by supporting-fact coverage on frozen predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_utils import portable_path, utc_now, write_json_atomic
from phase2_controller_inputs import evidence_key
from phase3a_metrics import coverage_bucket


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "phase2" / "controller" / "sentence_256" / "test.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "phase3" / "hard_partial"
CONTINUE_BUCKETS = ["easy_continue", "medium_partial", "hard_partial"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Repeat NAME=path/to/original_predictions.jsonl",
    )
    parser.add_argument("--controller-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--evidence-mode", choices=["raw", "cumulative"], default="cumulative")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_named_paths(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prediction must use NAME=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in names:
            raise ValueError(f"Prediction name is empty or repeated: {name!r}")
        names.add(name)
        output.append((name, Path(raw_path)))
    return output


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def probability_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean_stop_probability": float(array.mean()),
        "std_stop_probability": float(array.std(ddof=0)),
        "median_stop_probability": float(np.median(array)),
        "p90_stop_probability": float(np.percentile(array, 90)),
    }


def analyze_model(name: str, prediction_path: Path, source: dict, evidence_mode: str) -> tuple[list[dict], list[dict]]:
    predictions = read_jsonl(prediction_path)
    enriched = []
    seen = set()
    for prediction in predictions:
        key = (prediction["question_id"], prediction["stage"])
        if key in seen or key not in source:
            raise ValueError(f"Duplicate or unknown prediction key: {key}")
        seen.add(key)
        record = source[key]
        view = record[evidence_key(evidence_mode)]
        coverage = float(view["supporting_fact_recall"])
        actual = int(prediction["actual_stop_label"])
        if actual != int(coverage == 1.0):
            raise ValueError(f"Stop label and coverage disagree for {key}")
        enriched.append(
            {
                "model": name,
                "question_id": prediction["question_id"],
                "stage": prediction["stage"],
                "split": prediction.get("split", "test"),
                "actual_stop_label": actual,
                "actual_three_class_label": prediction.get("actual_three_class_label"),
                "predicted_stop_label": int(prediction["predicted_stop_label"]),
                "stop_probability": float(prediction["stop_probability"]),
                "supporting_fact_coverage": coverage,
                "coverage_bucket": coverage_bucket(coverage),
                "predicted_coverage": prediction.get("predicted_coverage"),
            }
        )
    if len(seen) != len(source):
        raise ValueError(f"{name} predictions cover {len(seen)} of {len(source)} source rows")

    stop_rows = [row for row in enriched if int(row["actual_stop_label"]) == 1]
    summaries = []
    for bucket in CONTINUE_BUCKETS:
        rows = [row for row in enriched if row["coverage_bucket"] == bucket]
        probabilities = [float(row["stop_probability"]) for row in rows]
        false_stops = sum(int(row["predicted_stop_label"]) == 1 for row in rows)
        comparison = [*rows, *stop_rows]
        comparison_labels = [int(row["actual_stop_label"]) for row in comparison]
        comparison_probabilities = [float(row["stop_probability"]) for row in comparison]
        stop_vs_bucket_auroc = None
        if rows and stop_rows:
            stop_vs_bucket_auroc = float(
                roc_auc_score(comparison_labels, comparison_probabilities)
            )
        stats = probability_summary(probabilities) if probabilities else {
            "mean_stop_probability": None,
            "std_stop_probability": None,
            "median_stop_probability": None,
            "p90_stop_probability": None,
        }
        summaries.append(
            {
                "model": name,
                "split": "test",
                "coverage_bucket": bucket,
                "samples": len(rows),
                "false_stop_count": false_stops,
                "false_stop_rate": false_stops / len(rows) if rows else None,
                "within_bucket_auroc": None,
                "within_bucket_auroc_reason": "undefined_single_continue_class",
                "stop_vs_bucket_auroc": stop_vs_bucket_auroc,
                **stats,
            }
        )
    return enriched, summaries


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Phase 3A Hard Partial Analysis",
        "",
        "All buckets below contain actual Continue examples only. Consequently, within-bucket AUROC is undefined. `Stop vs bucket AUROC` compares each Continue bucket against all truly Sufficient test examples.",
        "",
        "| Model | Continue bucket | N | False Stop Rate | Mean Stop P | P90 Stop P | Stop vs bucket AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        render = lambda value: "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['model']} | {row['coverage_bucket']} | {row['samples']} | "
            f"{render(row['false_stop_rate'])} | {render(row['mean_stop_probability'])} | "
            f"{render(row['p90_stop_probability'])} | {render(row['stop_vs_bucket_auroc'])} |"
        )
    lines.extend(
        [
            "",
            "The test distribution and coverage values are unchanged. These are frozen-error analyses, not threshold-selection results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    named_paths = parse_named_paths(args.prediction)
    output_dir = args.output_dir.resolve()
    targets = [
        output_dir / "hard_partial_summary.csv",
        output_dir / "hard_partial_analysis.md",
        output_dir / "analysis_manifest.json",
        *[output_dir / f"{name}_samples.jsonl" for name, _ in named_paths],
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Hard Partial outputs: {existing}")
    records = read_jsonl(args.controller_data)
    source = {(row["question_id"], row["stage"]): row for row in records}
    if len(source) != len(records):
        raise ValueError("Controller source contains duplicate question/stage keys")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, path in named_paths:
        enriched, summaries = analyze_model(name, path, source, args.evidence_mode)
        write_jsonl_atomic(output_dir / f"{name}_samples.jsonl", enriched)
        summary_rows.extend(summaries)
    write_csv_atomic(output_dir / "hard_partial_summary.csv", summary_rows)
    (output_dir / "hard_partial_analysis.md").write_text(
        render_markdown(summary_rows), encoding="utf-8"
    )
    write_json_atomic(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "split": "test",
            "evidence_mode": args.evidence_mode,
            "controller_data": portable_path(args.controller_data.resolve(), ROOT),
            "predictions": {
                name: portable_path(path.resolve(), ROOT) for name, path in named_paths
            },
            "within_bucket_auroc": "undefined because each requested bucket contains only Continue labels",
            "alternative_auroc": "each Continue bucket versus all Sufficient examples",
        },
    )


if __name__ == "__main__":
    main()

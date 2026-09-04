"""Paired question-level bootstrap for frozen multi-stage Controller predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from experiment_utils import portable_path, utc_now, write_json_atomic
    from phase3a_metrics import metrics_from_decisions
except ModuleNotFoundError:  # Imported as scripts.bootstrap_phase3a in tests.
    from scripts.experiment_utils import portable_path, utc_now, write_json_atomic
    from scripts.phase3a_metrics import metrics_from_decisions


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results" / "phase3" / "bootstrap"
METRICS = ["macro_f1", "auroc", "false_stop_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True, help="Repeat NAME=PATH")
    parser.add_argument("--compare", help="Optional paired comparison BASELINE,MODEL")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prediction must use NAME=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"Prediction name is empty or repeated: {name!r}")
        output[name] = Path(raw_path)
    return output


def group_by_question(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["question_id"], row["stage"])
        if key in seen:
            raise ValueError(f"Duplicate question/stage prediction: {key}")
        seen.add(key)
        grouped[row["question_id"]].append(row)
    return dict(grouped)


def sampled_metrics(grouped: dict[str, list[dict]], sampled_questions: list[str]) -> dict:
    rows = [row for question_id in sampled_questions for row in grouped[question_id]]
    return metrics_from_decisions(
        [row["actual_stop_label"] for row in rows],
        [row["predicted_stop_label"] for row in rows],
        [row["stop_probability"] for row in rows],
    )


def percentile_interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def main() -> None:
    args = parse_args()
    if args.replicates < 2000:
        raise ValueError("Phase 3A requires at least 2000 bootstrap replicates")
    paths = parse_named_paths(args.prediction)
    grouped = {name: group_by_question(read_jsonl(path)) for name, path in paths.items()}
    question_sets = {name: set(values) for name, values in grouped.items()}
    reference_name = next(iter(grouped))
    reference_questions = sorted(question_sets[reference_name])
    for name, questions in question_sets.items():
        if questions != set(reference_questions):
            raise ValueError(f"Question IDs for {name} do not match {reference_name}")
        reference_stages = {
            qid: {row["stage"] for row in grouped[reference_name][qid]}
            for qid in reference_questions
        }
        current_stages = {
            qid: {row["stage"] for row in grouped[name][qid]}
            for qid in reference_questions
        }
        if current_stages != reference_stages:
            raise ValueError(f"Stage decisions for {name} do not match {reference_name}")
        reference_labels = {
            (row["question_id"], row["stage"]): int(row["actual_stop_label"])
            for rows in grouped[reference_name].values() for row in rows
        }
        current_labels = {
            (row["question_id"], row["stage"]): int(row["actual_stop_label"])
            for rows in grouped[name].values() for row in rows
        }
        if current_labels != reference_labels:
            raise ValueError(f"Ground-truth decisions for {name} do not match {reference_name}")
    comparison = None
    if args.compare:
        parts = [value.strip() for value in args.compare.split(",")]
        if len(parts) != 2 or any(value not in grouped for value in parts):
            raise ValueError("--compare must name two provided predictions")
        comparison = tuple(parts)

    output_dir = args.output_dir.resolve()
    targets = [
        output_dir / "bootstrap_summary.csv",
        output_dir / "bootstrap_replicates.jsonl",
        output_dir / "bootstrap_manifest.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite bootstrap outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    observed = {
        name: sampled_metrics(values, reference_questions) for name, values in grouped.items()
    }
    replicates = []
    metric_values = {name: {metric: [] for metric in METRICS} for name in grouped}
    delta_values = {metric: [] for metric in METRICS}
    for replicate in range(args.replicates):
        sampled = rng.choice(reference_questions, size=len(reference_questions), replace=True).tolist()
        result = {name: sampled_metrics(values, sampled) for name, values in grouped.items()}
        for name, metrics in result.items():
            for metric in METRICS:
                if metrics[metric] is not None:
                    metric_values[name][metric].append(float(metrics[metric]))
        deltas = None
        if comparison:
            baseline, model = comparison
            deltas = {}
            for metric in METRICS:
                if result[model][metric] is None or result[baseline][metric] is None:
                    deltas[metric] = None
                else:
                    deltas[metric] = float(result[model][metric] - result[baseline][metric])
            for metric, value in deltas.items():
                if value is not None:
                    delta_values[metric].append(value)
        replicates.append({"replicate": replicate, "models": result, "paired_delta": deltas})

    summary_rows = []
    for name, values in metric_values.items():
        for metric, samples in values.items():
            low, high = percentile_interval(samples)
            summary_rows.append(
                {
                    "result_type": "model",
                    "name": name,
                    "metric": metric,
                    "observed": observed[name][metric],
                    "bootstrap_mean": float(np.mean(samples)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "replicates": args.replicates,
                    "valid_replicates": len(samples),
                    "sampling_unit": "question_id",
                }
            )
    if comparison:
        baseline, model = comparison
        for metric, samples in delta_values.items():
            low, high = percentile_interval(samples)
            observed_delta = None
            if observed[model][metric] is not None and observed[baseline][metric] is not None:
                observed_delta = observed[model][metric] - observed[baseline][metric]
            summary_rows.append(
                {
                    "result_type": "paired_delta",
                    "name": f"{model}_minus_{baseline}",
                    "metric": metric,
                    "observed": observed_delta,
                    "bootstrap_mean": float(np.mean(samples)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "replicates": args.replicates,
                    "valid_replicates": len(samples),
                    "sampling_unit": "question_id",
                }
            )
    write_csv_atomic(output_dir / "bootstrap_summary.csv", summary_rows)
    write_jsonl_atomic(output_dir / "bootstrap_replicates.jsonl", replicates)
    write_json_atomic(
        output_dir / "bootstrap_manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "seed": args.seed,
            "replicates": args.replicates,
            "sampling_unit": "question_id",
            "questions_per_replicate": len(reference_questions),
            "all_stage_decisions_kept_with_question": True,
            "predictions": {
                name: portable_path(path.resolve(), ROOT) for name, path in paths.items()
            },
            "paired_comparison": comparison,
        },
    )


if __name__ == "__main__":
    main()

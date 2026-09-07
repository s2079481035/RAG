"""Derive preregistered paired comparisons from saved bootstrap replicates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from experiment_utils import portable_path, utc_now, write_json_atomic
except ModuleNotFoundError:  # Imported as scripts.summarize_phase3a_bootstrap_pairs in tests.
    from scripts.experiment_utils import portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "results" / "phase3" / "bootstrap"
DEFAULT_COMPARISONS = [
    "QueryStage,ScoreAwareBaseline",
    "QueryStage,FinalController",
    "ScoreAwareBaseline,FinalController",
]
METRICS = ["macro_f1", "auroc", "false_stop_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replicates", type=Path, default=DEFAULT_DIR / "bootstrap_replicates.jsonl"
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_DIR / "bootstrap_summary.csv")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DIR / "bootstrap_manifest.json")
    parser.add_argument("--comparison", action="append", help="Repeat BASELINE,MODEL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_comparisons(values: list[str]) -> list[tuple[str, str]]:
    comparisons = []
    for value in values:
        parts = tuple(part.strip() for part in value.split(","))
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            raise ValueError(f"Comparison must use distinct BASELINE,MODEL names: {value!r}")
        if parts in comparisons:
            raise ValueError(f"Duplicate comparison: {value!r}")
        comparisons.append(parts)
    return comparisons


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def observed_metrics(path: Path) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["result_type"] != "model" or row["metric"] not in METRICS:
                continue
            output.setdefault(row["name"], {})[row["metric"]] = float(row["observed"])
    return output


def paired_comparison_rows(
    replicates: list[dict],
    observed: dict[str, dict[str, float]],
    comparisons: list[tuple[str, str]],
) -> list[dict]:
    if not replicates:
        raise ValueError("Bootstrap replicate file is empty")
    available = set(replicates[0].get("models", {}))
    requested = {name for pair in comparisons for name in pair}
    missing = requested - available
    if missing:
        raise ValueError(f"Bootstrap replicates lack models: {sorted(missing)}")
    if requested - set(observed):
        raise ValueError("Bootstrap summary lacks observed metrics for a requested model")

    rows = []
    for baseline, model in comparisons:
        for metric in METRICS:
            values = []
            for replicate in replicates:
                models = replicate["models"]
                baseline_value = models[baseline].get(metric)
                model_value = models[model].get(metric)
                if baseline_value is not None and model_value is not None:
                    values.append(float(model_value) - float(baseline_value))
            if not values:
                raise ValueError(f"No valid bootstrap values for {baseline},{model}:{metric}")
            low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
            rows.append(
                {
                    "comparison": f"{model}_minus_{baseline}",
                    "baseline": baseline,
                    "model": model,
                    "metric": metric,
                    "observed_delta": observed[model][metric] - observed[baseline][metric],
                    "bootstrap_mean_delta": float(np.mean(values)),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "valid_replicates": len(values),
                    "sampling_unit": "question_id",
                }
            )
    return rows


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Phase 3A Paired Bootstrap Comparisons",
        "",
        "All differences are `model - baseline` and reuse the saved question-level bootstrap draws. No model inference or Test tuning is performed.",
        "",
        "| Comparison | Metric | Observed delta | 95% CI | Excludes zero |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['observed_delta']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{'yes' if row['ci_excludes_zero'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "paired_comparison_summary.csv"
    report_path = output_dir / "paired_comparison_summary.md"
    output_manifest = output_dir / "paired_comparison_manifest.json"
    existing = [path for path in [csv_path, report_path, output_manifest] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite paired bootstrap summaries: {existing}")

    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if source_manifest.get("sampling_unit") != "question_id":
        raise ValueError("Source bootstrap did not sample at question level")
    if not source_manifest.get("all_stage_decisions_kept_with_question"):
        raise ValueError("Source bootstrap did not keep each question's stages together")
    if int(source_manifest.get("replicates", 0)) < 2000:
        raise ValueError("Source bootstrap has fewer than 2000 replicates")

    comparisons = parse_comparisons(args.comparison or DEFAULT_COMPARISONS)
    replicates = read_jsonl(args.replicates)
    if len(replicates) != int(source_manifest["replicates"]):
        raise ValueError("Bootstrap replicate count differs from its manifest")
    rows = paired_comparison_rows(
        replicates, observed_metrics(args.summary), comparisons
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(csv_path, rows)
    report_path.write_text(render_markdown(rows), encoding="utf-8")
    write_json_atomic(
        output_manifest,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "source_replicates": portable_path(args.replicates.resolve(), ROOT),
            "source_summary": portable_path(args.summary.resolve(), ROOT),
            "source_manifest": portable_path(args.manifest.resolve(), ROOT),
            "replicates": len(replicates),
            "sampling_unit": "question_id",
            "all_stage_decisions_kept_with_question": True,
            "comparisons": [
                {"baseline": baseline, "model": model}
                for baseline, model in comparisons
            ],
            "test_predictions_recomputed": False,
            "test_threshold_refit": False,
        },
    )


if __name__ == "__main__":
    main()

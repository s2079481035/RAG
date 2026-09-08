"""Fit Dev-only scalar temperature and report Phase 3B calibration quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from phase3a_metrics import coverage_bucket
from phase3b_metrics import calibration_metrics, fit_temperature, temperature_scale
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


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def reliability_points(labels, probabilities, bins: int) -> list[dict]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    points = []
    for index in range(bins):
        mask = (
            (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
            if index == bins - 1
            else (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        )
        if np.any(mask):
            points.append(
                {
                    "bin": index,
                    "count": int(np.sum(mask)),
                    "mean_probability": float(np.mean(probabilities[mask])),
                    "empirical_stop_rate": float(np.mean(labels[mask])),
                }
            )
    return points


def save_reliability(path: Path, points: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="perfect")
    axis.plot(
        [row["mean_probability"] for row in points],
        [row["empirical_stop_rate"] for row in points],
        marker="o",
        color="#1f77b4",
        label="observed",
    )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted Stop probability", ylabel="Observed Stop frequency", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_reliability_comparison(
    path: Path, raw_points: list[dict], temperature_points: list[dict]
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="perfect")
    for label, points, color in [
        ("raw", raw_points, "#1f77b4"),
        ("temperature", temperature_points, "#d62728"),
    ]:
        axis.plot(
            [row["mean_probability"] for row in points],
            [row["empirical_stop_rate"] for row in points],
            marker="o",
            color=color,
            label=label,
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted Stop probability",
        ylabel="Observed Stop frequency",
        title="Controller reliability (dev_calibration)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    calibration_config = config["calibration"]
    base_path = ROOT / "results" / "phase3b" / "base_controller.json"
    protocol_path = ROOT / "results" / "phase3b" / "protocol_manifest.json"
    if not base_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("Run scripts/prepare_phase3b.py first")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    split_config = config["dev_split"]
    calibration_ids_path = repo_path(split_config["calibration_ids"])
    policy_ids_path = repo_path(split_config["policy_ids"])
    if file_sha256(calibration_ids_path) != protocol["dev_calibration_ids_sha256"]:
        raise ValueError("Dev calibration IDs changed after preparation")
    if file_sha256(policy_ids_path) != protocol["dev_policy_ids_sha256"]:
        raise ValueError("Dev policy IDs changed after preparation")
    calibration_ids = read_ids(calibration_ids_path)
    policy_ids = read_ids(policy_ids_path)
    if calibration_ids & policy_ids:
        raise ValueError("Question-ID leakage between calibration and policy sets")

    predictions_path = repo_path(base["dev_predictions"])
    if file_sha256(predictions_path) != base["dev_predictions_sha256"]:
        raise ValueError("Frozen Phase 3A Dev predictions changed")
    predictions = read_jsonl(predictions_path)
    all_ids = {row["question_id"] for row in predictions}
    if all_ids != calibration_ids | policy_ids:
        raise ValueError("Frozen Dev predictions and Phase 3B split IDs differ")
    for row in predictions:
        row["phase3b_subset"] = (
            "dev_calibration" if row["question_id"] in calibration_ids else "dev_policy"
        )
        row["raw_stop_probability"] = float(row["stop_probability"])

    actionable_stages = set(config["policy"]["decision_stages"])
    fit_rows = [
        row
        for row in predictions
        if row["phase3b_subset"] == "dev_calibration" and row["stage"] in actionable_stages
    ]
    labels = np.asarray([row["actual_stop_label"] for row in fit_rows], dtype=int)
    raw = np.asarray([row["raw_stop_probability"] for row in fit_rows], dtype=float)
    epsilon = float(calibration_config["probability_epsilon"])
    temperature, optimized_nll = fit_temperature(
        labels,
        raw,
        log_bounds=tuple(calibration_config["temperature_log_bounds"]),
        iterations=int(calibration_config["optimizer_iterations"]),
        epsilon=epsilon,
    )
    for row in predictions:
        row["temperature_stop_probability"] = float(
            temperature_scale([row["raw_stop_probability"]], temperature, epsilon)[0]
        )

    output_dir = ROOT / "results" / "phase3b"
    metrics_path = output_dir / "calibration_metrics.csv"
    calibrator_path = output_dir / "temperature_scaling.json"
    predictions_output = output_dir / "calibrated_dev_predictions.jsonl"
    manifest_path = output_dir / "calibration_manifest.json"
    raw_figure = ROOT / "figures" / "phase3b" / "reliability_raw.png"
    temperature_figure = ROOT / "figures" / "phase3b" / "reliability_temperature.png"
    comparison_figure = ROOT / "figures" / "phase3b" / "reliability_diagram.png"
    targets = [
        metrics_path,
        calibrator_path,
        predictions_output,
        manifest_path,
        raw_figure,
        temperature_figure,
        comparison_figure,
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3B calibration: {existing}")

    metric_rows = []
    bins = int(calibration_config["ece_bins"])
    high = float(calibration_config["high_confidence_stop_probability"])
    for subset in ["dev_calibration", "dev_policy"]:
        subset_rows = [
            row
            for row in predictions
            if row["phase3b_subset"] == subset and row["stage"] in actionable_stages
        ]
        subset_labels = np.asarray([row["actual_stop_label"] for row in subset_rows], dtype=int)
        for method, key in [
            ("raw", "raw_stop_probability"),
            ("temperature", "temperature_stop_probability"),
        ]:
            values = np.asarray([row[key] for row in subset_rows], dtype=float)
            metrics = calibration_metrics(subset_labels, values, bins, epsilon)
            hard = [
                row for row in subset_rows
                if int(row["actual_stop_label"]) == 0
                and coverage_bucket(float(row["true_coverage"])) == "hard_partial"
            ]
            high_false = sum(float(row[key]) >= high for row in hard)
            metric_rows.append(
                {
                    "split": subset,
                    "method": method,
                    "evaluation_scope": "actionable_controller_stages",
                    **metrics,
                    "hard_partial_continue_count": len(hard),
                    "hard_partial_mean_stop_probability": (
                        sum(float(row[key]) for row in hard) / len(hard) if hard else None
                    ),
                    "hard_partial_high_confidence_false_stop_count": high_false,
                    "hard_partial_high_confidence_false_stop_rate": (
                        high_false / len(hard) if hard else None
                    ),
                }
            )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metrics_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    temporary.replace(metrics_path)
    write_jsonl_atomic(predictions_output, predictions)

    raw_points = reliability_points(labels, raw, bins)
    calibrated_fit = temperature_scale(raw, temperature, epsilon)
    temperature_points = reliability_points(labels, calibrated_fit, bins)
    save_reliability(raw_figure, raw_points, "Raw Controller reliability (dev_calibration)")
    save_reliability(
        temperature_figure,
        temperature_points,
        "Temperature-scaled reliability (dev_calibration)",
    )
    save_reliability_comparison(comparison_figure, raw_points, temperature_points)
    write_json_atomic(
        calibrator_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "fit_split": "dev_calibration",
            "fit_questions": len(calibration_ids),
            "fit_stage_decisions": len(fit_rows),
            "fit_stages": sorted(actionable_stages),
            "temperature": temperature,
            "optimized_nll": optimized_nll,
            "probability_epsilon": epsilon,
            "source_predictions": portable_path(predictions_path, ROOT),
            "source_predictions_sha256": file_sha256(predictions_path),
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
            "base_controller": portable_path(base_path, ROOT),
            "base_controller_sha256": file_sha256(base_path),
            "calibrator": portable_path(calibrator_path, ROOT),
            "calibrator_sha256": file_sha256(calibrator_path),
            "test_consulted": False,
            "evaluation_scope": "actionable_controller_stages",
            "outputs": {
                "metrics": portable_path(metrics_path, ROOT),
                "predictions": portable_path(predictions_output, ROOT),
                "reliability_raw": portable_path(raw_figure, ROOT),
                "reliability_temperature": portable_path(temperature_figure, ROOT),
                "reliability_diagram": portable_path(comparison_figure, ROOT),
            },
        },
    )
    print(f"temperature={temperature:.8f}")


if __name__ == "__main__":
    main()

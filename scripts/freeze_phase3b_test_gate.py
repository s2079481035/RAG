"""Freeze all Dev-selected Phase 3B artifacts before any Phase 3B Test run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config["test"]["evaluation_only"]:
        raise ValueError("Phase 3B Test must remain evaluation-only")
    gate_path = repo_path(config["test"]["gate"])
    if gate_path.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 3B Test gate: {gate_path}")
    test_outputs = [
        repo_path(config["generation"]["stage_cache_test"]),
        ROOT / "results" / "phase3b" / "test_risk_policy.csv",
        ROOT / "results" / "phase3b" / "test_policy_trajectories.jsonl",
        ROOT / "results" / "phase3b" / "test_evaluation_manifest.json",
    ]
    existing_test = [path for path in test_outputs if path.exists()]
    if existing_test:
        raise ValueError(f"Phase 3B Test outputs already exist before gate freeze: {existing_test}")

    artifacts = {
        "protocol_manifest": ROOT / "results" / "phase3b" / "protocol_manifest.json",
        "base_controller": ROOT / "results" / "phase3b" / "base_controller.json",
        "calibration_manifest": ROOT / "results" / "phase3b" / "calibration_manifest.json",
        "temperature_scaling": ROOT / "results" / "phase3b" / "temperature_scaling.json",
        "policy_selection_manifest": ROOT / "results" / "phase3b" / "policy_selection_manifest.json",
        "selected_thresholds": ROOT / "results" / "phase3b" / "selected_thresholds.json",
        "dev_generation_manifest": ROOT / "results" / "phase3b" / "generation" / "dev_policy" / "generation_manifest.json",
        "dev_retrieval_benchmark_manifest": ROOT / "results" / "phase3b" / "latency" / "dev_policy_retrieval_manifest.json",
        "dev_controller_benchmark_manifest": ROOT / "results" / "phase3b" / "latency" / "dev_policy_controller_manifest.json",
        "dev_evaluation_manifest": ROOT / "results" / "phase3b" / "dev_policy_evaluation_manifest.json",
        "dev_end_to_end_summary": ROOT / "results" / "phase3b" / "dev_policy_end_to_end.csv",
        "dev_threshold_sweep": ROOT / "results" / "phase3b" / "dev_threshold_sweep.csv",
    }
    missing = [path for path in artifacts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Complete all Phase 3B Dev steps before freezing Test: {missing}")
    selected = json.loads(artifacts["selected_thresholds"].read_text(encoding="utf-8"))
    if selected.get("selection_split") != "dev_policy" or selected.get("test_consulted") is not False:
        raise ValueError("Risk thresholds were not selected on dev_policy only")
    primary = config["policy"]["primary_report_policy"]
    matching_primary = [
        row for row in selected["policies"]
        if row["calibration_method"] == primary["calibration"]
        and row["selection"] == primary["selection"]
        and float(row["risk_target"]) == float(primary["risk_target"])
    ]
    if len(matching_primary) != 1:
        raise ValueError("Primary report policy is missing or ambiguous")
    if matching_primary[0].get("status") != "selected":
        raise ValueError("Primary report policy is infeasible on dev_policy")

    gate_path.parent.mkdir(parents=True, exist_ok=True)
    hashes = {f"{name}_sha256": file_sha256(path) for name, path in artifacts.items()}
    write_json_atomic(
        gate_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "status": "frozen",
            "phase": "3B",
            "git_commit": git_commit(ROOT),
            "selection_split": "dev_policy",
            "calibration_split": "dev_calibration",
            "test_consulted": False,
            "prior_test_exposure": config["prior_test_exposure"],
            "config": portable_path(args.config.resolve(), ROOT),
            "config_sha256": file_sha256(args.config),
            "selected_thresholds": portable_path(artifacts["selected_thresholds"], ROOT),
            "temperature_scaling": portable_path(artifacts["temperature_scaling"], ROOT),
            "base_controller": portable_path(artifacts["base_controller"], ROOT),
            "primary_report_policy": primary,
            "primary_selected_threshold": matching_primary[0],
            "infeasible_preregistered_policies": [
                row for row in selected["policies"] if row.get("status") != "selected"
            ],
            "frozen_artifacts": {
                name: portable_path(path, ROOT) for name, path in artifacts.items()
            },
            **hashes,
        },
    )
    print("frozen")


if __name__ == "__main__":
    main()

"""Freeze the Phase 3A Controller and create leakage-safe Phase 3B Dev splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidence_utils import assert_disjoint_question_ids
from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from phase3b_metrics import deterministic_question_split


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_ids(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    temporary.replace(path)


def validate_base_controller(config: dict) -> dict:
    expected = config["base_controller"]
    selection_path = resolve_repo_path(expected["selection_manifest"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selection_split"] != "dev" or selection.get("test_consulted") is not False:
        raise ValueError("Phase 3A Controller selection was not Dev-only")
    if selection["selected_group"] != expected["expected_selected_group"]:
        raise ValueError("Configured Base Controller differs from Phase 3A selection")
    selected = [row for row in selection["selected_runs"] if int(row["seed"]) == int(expected["seed"])]
    if len(selected) != 1 or selected[0]["run_dir"] != expected["run_dir"]:
        raise ValueError("Configured seed run is not the selected Phase 3A run")

    run_dir = resolve_repo_path(expected["run_dir"])
    resolved_path = run_dir / "resolved_config.json"
    run_manifest_path = run_dir / "run_manifest.json"
    resolved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest["status"] != "complete":
        raise ValueError("Base Controller training run is incomplete")
    resolved = resolved_config["resolved"]
    checks = {
        "variant": resolved["variant"],
        "baseline": resolved["baseline"],
        "representation": resolved["representation"],
        "evidence_mode": resolved["evidence_mode"],
        "seed": int(resolved["seed"]),
        "coverage_auxiliary": bool(resolved["coverage_auxiliary"]),
        "coverage_lambda": float(resolved["coverage_lambda"]),
        "sampling": resolved["sampling"],
        "backbone": resolved["backbone"],
    }
    for key, actual in checks.items():
        expected_value = expected[key]
        if actual != expected_value:
            raise ValueError(f"Base Controller {key} mismatch: {actual!r} != {expected_value!r}")

    checkpoint = Path(run_manifest["checkpoint_path"])
    if not checkpoint.exists():
        checkpoint = run_dir / checkpoint.name
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing Base Controller checkpoint: {checkpoint}")
    dev_predictions = run_dir / "dev_predictions.jsonl"
    test_predictions = run_dir / "evaluation" / "test" / "original_predictions.jsonl"
    if not dev_predictions.exists():
        raise FileNotFoundError(f"Missing frozen Controller Dev predictions: {dev_predictions}")
    return {
        "selection_manifest": portable_path(selection_path, ROOT),
        "selection_manifest_sha256": file_sha256(selection_path),
        "run_dir": portable_path(run_dir, ROOT),
        "resolved_config": portable_path(resolved_path, ROOT),
        "resolved_config_sha256": file_sha256(resolved_path),
        "run_manifest": portable_path(run_manifest_path, ROOT),
        "run_manifest_sha256": file_sha256(run_manifest_path),
        "checkpoint": portable_path(checkpoint, ROOT),
        "checkpoint_sha256": file_sha256(checkpoint),
        "dev_predictions": portable_path(dev_predictions, ROOT),
        "dev_predictions_sha256": file_sha256(dev_predictions),
        "test_predictions": portable_path(test_predictions, ROOT),
        "test_predictions_sha256": None,
        "test_predictions_note": "Existence and content are not inspected before the Phase 3B Test gate.",
        "frozen_fields": checks,
        "phase3a_selected_dev_threshold": float(run_manifest["selected_dev_threshold"]),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split_config = config["dev_split"]
    calibration_path = resolve_repo_path(split_config["calibration_ids"])
    policy_path = resolve_repo_path(split_config["policy_ids"])
    base_output = ROOT / "results" / "phase3b" / "base_controller.json"
    protocol_output = ROOT / "results" / "phase3b" / "protocol_manifest.json"
    targets = [calibration_path, policy_path, base_output, protocol_output]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3B preparation: {existing}")

    closing_path = resolve_repo_path(config["phase3a_closing_report"])
    if not closing_path.exists():
        raise FileNotFoundError(f"Missing Phase 3A closing report: {closing_path}")
    base = validate_base_controller(config)

    source_path = resolve_repo_path(split_config["source"])
    records = read_jsonl(source_path)
    question_ids = sorted({row["question_id"] for row in records})
    stages_by_question = {}
    for qid in question_ids:
        stages_by_question[qid] = {row["stage"] for row in records if row["question_id"] == qid}
    expected_stages = {row["name"] for row in config["stages"]}
    invalid = [qid for qid, stages in stages_by_question.items() if stages != expected_stages]
    if invalid:
        raise ValueError(f"Dev questions with incomplete stage trajectories: {invalid[:5]}")
    expected_total = int(split_config["calibration_questions"]) + int(split_config["policy_questions"])
    if len(question_ids) != expected_total:
        raise ValueError(f"Expected {expected_total} Dev questions, found {len(question_ids)}")
    calibration_ids, policy_ids = deterministic_question_split(
        question_ids,
        int(split_config["calibration_questions"]),
        int(split_config["seed"]),
    )
    assert_disjoint_question_ids(
        {"dev_calibration": calibration_ids, "dev_policy": policy_ids}
    )
    write_ids(calibration_path, calibration_ids)
    write_ids(policy_path, policy_ids)
    write_json_atomic(
        base_output,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3B",
            **base,
        },
    )
    write_json_atomic(
        protocol_output,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3B",
            "git_commit": git_commit(ROOT),
            "phase3a_frozen_commit": config["phase3a_frozen_commit"],
            "phase3a_closing_report": portable_path(closing_path, ROOT),
            "phase3a_closing_report_sha256": file_sha256(closing_path),
            "config": portable_path(args.config.resolve(), ROOT),
            "config_sha256": file_sha256(args.config),
            "source": portable_path(source_path, ROOT),
            "source_sha256": file_sha256(source_path),
            "split_seed": int(split_config["seed"]),
            "split_unit": "question_id",
            "dev_calibration_questions": len(calibration_ids),
            "dev_policy_questions": len(policy_ids),
            "dev_calibration_ids_sha256": file_sha256(calibration_path),
            "dev_policy_ids_sha256": file_sha256(policy_path),
            "test_is_evaluation_only": bool(config["test"]["evaluation_only"]),
            "prior_test_exposure": config["prior_test_exposure"],
        },
    )
    print(f"prepared {len(calibration_ids)} calibration and {len(policy_ids)} policy questions")


if __name__ == "__main__":
    main()

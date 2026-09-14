"""One-time authorization and freeze checks for Phase 4 heldout access."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVALUATION_CONFIG = ROOT / "configs" / "phase4" / "heldout_evaluation.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def git_status() -> str:
    return git_output("status", "--short", "--untracked-files=all")


def require_clean_worktree() -> None:
    status = git_status()
    if status:
        raise ValueError(
            "Phase 4 heldout requires a clean worktree before access. "
            f"Current changes:\n{status}"
        )


def require_ancestor(commit: str, descendant: str = "HEAD") -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, descendant],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Frozen commit {commit} is not an ancestor of {descendant}")


def validate_unlock(config: dict) -> tuple[dict, Path]:
    unlock_path = repo_path(config["authorization"]["unlock_manifest"])
    unlock = load_json(unlock_path)
    required = {
        "ready_for_heldout": True,
        "ready_for_heldout_label": "YES",
        "heldout_consulted": False,
        "human_confirmation_required": True,
        "human_confirmation_recorded": False,
        "heldout_execution_started": False,
    }
    for key, expected in required.items():
        if unlock.get(key) != expected:
            raise ValueError(
                f"Heldout unlock requirement failed: {key}={unlock.get(key)!r}, "
                f"expected {expected!r}"
            )
    for relative, expected_hash in {
        **unlock["protected_code"],
        **unlock["protected_adaptive_artifacts"],
    }.items():
        path = ROOT / relative
        if not path.exists() or file_sha256(path) != expected_hash:
            raise ValueError(f"Frozen Phase 4 artifact changed: {relative}")
    return unlock, unlock_path


def validate_authorization(
    evaluation_config: Path = DEFAULT_EVALUATION_CONFIG,
    *,
    require_environment: bool = True,
) -> tuple[dict, dict, Path]:
    config = load_json(evaluation_config)
    frozen_commit = str(config["frozen_choices_commit"])
    require_ancestor(frozen_commit)
    unlock, unlock_path = validate_unlock(config)
    authorization_path = repo_path(config["authorization"]["manifest"])
    if not authorization_path.exists():
        raise ValueError(
            "Missing committed human authorization. Run "
            "scripts/record_phase4_heldout_authorization.py first."
        )
    authorization = load_json(authorization_path)
    expected = {
        "status": "authorized",
        "run_heldout": "YES",
        "researcher": "sunjb",
        "frozen_choices_commit": frozen_commit,
        "heldout_consulted_at_authorization": False,
        "one_time_evaluation": True,
        "heldout_tuning_forbidden": True,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise ValueError(f"Invalid heldout authorization field: {key}")
    if authorization.get("unlock_manifest_sha256") != file_sha256(unlock_path):
        raise ValueError("Heldout unlock manifest changed after human authorization")
    if require_environment:
        required_value = config["authorization"]["required_environment_value"]
        if os.environ.get("RUN_HELDOUT") != required_value:
            raise ValueError(f"Set RUN_HELDOUT={required_value} for formal heldout access")
    return config, authorization, authorization_path


def guard_heldout_access(
    evaluation_config: Path = DEFAULT_EVALUATION_CONFIG,
    *,
    allow_running_resume: bool = True,
) -> tuple[dict, dict]:
    config, authorization, _ = validate_authorization(evaluation_config)
    run_manifest_path = repo_path(config["run_manifest"])
    completion_path = repo_path(config["completion_manifest"])
    if completion_path.exists():
        completion = load_json(completion_path)
        if completion.get("status") == "complete":
            raise ValueError("Formal Phase 4 heldout evaluation is already complete")
    if run_manifest_path.exists():
        run = load_json(run_manifest_path)
        if run.get("heldout_consulted") is True or run.get("status") == "complete":
            raise ValueError("Formal Phase 4 heldout evaluation has already been consumed")
        if not allow_running_resume or run.get("status") != "running":
            raise ValueError("Unexpected existing heldout run manifest")
        if run.get("authorization_sha256") != file_sha256(
            repo_path(config["authorization"]["manifest"])
        ):
            raise ValueError("Heldout authorization changed during the run")
    return config, authorization

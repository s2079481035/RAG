"""Persist the researcher's explicit one-time Phase 4 heldout authorization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from phase4_heldout_guard import (
    DEFAULT_EVALUATION_CONFIG,
    ROOT,
    file_sha256,
    load_json,
    repo_path,
    require_ancestor,
    require_clean_worktree,
    validate_unlock,
)


CONFIRMATION = (
    "I confirm that all Phase 4 Dev choices are frozen and authorize one-time "
    "heldout evaluation without heldout tuning."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--researcher", required=True)
    parser.add_argument("--frozen-commit", required=True)
    parser.add_argument("--confirmation", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    expected_commit = str(config["frozen_choices_commit"])
    if os.environ.get("RUN_HELDOUT") != "YES":
        raise ValueError("Explicit authorization requires RUN_HELDOUT=YES")
    if args.researcher != "sunjb":
        raise ValueError("Researcher must match the approved researcher: sunjb")
    if args.frozen_commit != expected_commit:
        raise ValueError(
            f"Frozen commit must be exactly {expected_commit}, got {args.frozen_commit}"
        )
    if args.confirmation.strip() != CONFIRMATION:
        raise ValueError("Confirmation text does not match the approved statement")
    require_ancestor(expected_commit)
    require_clean_worktree()
    unlock, unlock_path = validate_unlock(config)
    output_path = repo_path(config["authorization"]["manifest"])
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite authorization: {output_path}")
    write_json_atomic(
        output_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "status": "authorized",
            "run_heldout": "YES",
            "researcher": args.researcher,
            "confirmation": CONFIRMATION,
            "frozen_choices_commit": expected_commit,
            "authorization_recorded_at_git_commit": git_commit(ROOT),
            "heldout_consulted_at_authorization": False,
            "one_time_evaluation": True,
            "heldout_tuning_forbidden": True,
            "unlock_manifest": portable_path(unlock_path, ROOT),
            "unlock_manifest_sha256": file_sha256(unlock_path),
            "unlock_status": unlock["status"],
            "evaluation_config": portable_path(args.config, ROOT),
            "evaluation_config_sha256": file_sha256(args.config),
        },
    )
    print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

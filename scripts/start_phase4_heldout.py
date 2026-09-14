"""Create the immutable pre-run manifest immediately before heldout access."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
from phase4_heldout_guard import (
    DEFAULT_EVALUATION_CONFIG,
    ROOT,
    file_sha256,
    git_status,
    load_json,
    repo_path,
    require_clean_worktree,
    validate_authorization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, authorization, authorization_path = validate_authorization(args.config)
    run_path = repo_path(config["run_manifest"])
    completion_path = repo_path(config["completion_manifest"])
    if completion_path.exists() and load_json(completion_path).get("status") == "complete":
        raise ValueError("The one-time heldout evaluation is already complete")
    if run_path.exists():
        run = load_json(run_path)
        if not args.resume:
            raise FileExistsError(f"Heldout run already started: {run_path}")
        if run.get("status") != "running" or run.get("heldout_consulted") is not False:
            raise ValueError("Existing heldout run is not resumable")
        if run.get("authorization_sha256") != file_sha256(authorization_path):
            raise ValueError("Authorization changed after heldout execution began")
        print(run_path.relative_to(ROOT))
        return

    require_clean_worktree()
    protocol_path = ROOT / "configs" / "phase4" / "protocol.json"
    controller_path = ROOT / "configs" / "phase4" / "controller.json"
    router_path = ROOT / "configs" / "phase4" / "adaptive_rag_router.json"
    prompt_path = ROOT / "configs" / "phase4" / "llm_judge_prompt.txt"
    freeze_path = ROOT / "results" / "phase4" / "freeze_manifest.json"
    controller_gate_path = (
        ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "dev_gate_manifest.json"
    )
    router_gate_path = (
        ROOT / "results" / "phase4" / "2wiki" / "adaptive_rag" / "dev_gate_manifest.json"
    )
    freeze = load_json(freeze_path)
    controller_gate = load_json(controller_gate_path)
    router_gate = load_json(router_gate_path)
    write_json_atomic(
        run_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "status": "running",
            "one_time_formal_heldout_evaluation": True,
            "split": config["split"],
            "expected_questions": config["expected_questions"],
            "heldout_consulted_before_run": False,
            "heldout_consulted": False,
            "heldout_tuning_forbidden": True,
            "frozen_choices_commit": config["frozen_choices_commit"],
            "execution_code_commit": git_commit(ROOT),
            "git_status_before_run": git_status(),
            "authorization": portable_path(authorization_path, ROOT),
            "authorization_sha256": file_sha256(authorization_path),
            "researcher": authorization["researcher"],
            "environment": collect_environment(ROOT),
            "frozen_values": {
                "controller": controller_gate["operational_seed"],
                "classification_threshold": load_json(
                    ROOT / "configs" / "phase4" / "protocol.json"
                )["in_domain_policy"]["frozen_dev_gate"]["classification_operating_point"]["threshold"],
                "temperature": load_json(
                    ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "temperature_scaling.json"
                )["temperature"],
                "primary_policy": load_json(
                    ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "selected_thresholds.json"
                )["primary_selected_policy"],
                "adaptive_router_operational_seed": router_gate["operational_seed"],
                "adaptive_router_checkpoint_sha256": router_gate["operational_checkpoint_sha256"],
                "generator_model": freeze["models"]["generator"],
            },
            "frozen_file_sha256": {
                portable_path(path, ROOT): file_sha256(path)
                for path in [
                    protocol_path,
                    controller_path,
                    router_path,
                    prompt_path,
                    freeze_path,
                    controller_gate_path,
                    router_gate_path,
                    args.config,
                ]
            },
        },
    )
    print(run_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

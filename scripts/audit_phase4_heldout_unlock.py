"""Audit Phase 4 freeze conditions without reading or running heldout data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

try:
    from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
except ModuleNotFoundError:  # Imported as scripts.* in tests.
    from scripts.experiment_utils import git_commit, portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_is_committed(paths: list[str]) -> bool:
    for path in paths:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0:
            return False
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    args = parse_args()
    protocol_path = ROOT / "configs" / "phase4" / "protocol.json"
    protocol = load(protocol_path)
    phase4_freeze_path = ROOT / "results" / "phase4" / "freeze_manifest.json"
    phase4_freeze = load(phase4_freeze_path)
    controller_gate_path = ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "dev_gate_manifest.json"
    controller_gate = load(controller_gate_path)
    thresholds_path = ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "selected_thresholds.json"
    thresholds = load(thresholds_path)
    temperature_path = ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "temperature_scaling.json"
    temperature = load(temperature_path)
    score_path = ROOT / "results" / "phase4" / "score_threshold" / "selected_thresholds.json"
    score = load(score_path)
    judge_prompt_path = ROOT / "results" / "phase4" / "2wiki" / "llm_judge" / "prompt_freeze_manifest.json"
    judge_prompt = load(judge_prompt_path)
    judge_analysis_path = ROOT / "results" / "phase4" / "2wiki" / "llm_judge" / "analysis_manifest.json"
    judge_analysis = load(judge_analysis_path)
    adaptive_path = ROOT / "results" / "phase4" / "2wiki" / "adaptive_rag_protocol_manifest.json"
    adaptive = load(adaptive_path)

    primary = thresholds["primary_selected_policy"]
    expected_primary = protocol["in_domain_policy"]["primary_report_policy"]
    frozen_gate = protocol["in_domain_policy"]["frozen_dev_gate"]
    frozen_risk = frozen_gate["risk_controlled_operating_point"]
    generation_config_path = ROOT / phase4_freeze["files"]["generation_config"]["path"]
    protected_code = [
        "configs/phase4/protocol.json",
        "configs/phase4/controller.json",
        "configs/phase4/llm_judge_prompt.txt",
        "scripts/analyze_phase4_controller_dev.py",
        "scripts/analyze_phase4_retrieval_ceiling.py",
        "scripts/analyze_phase4_llm_judge.py",
        "scripts/evaluate_phase4_score_threshold.py",
        "scripts/run_phase4_llm_judge.py",
    ]
    checks = {
        "controller_frozen": (
            controller_gate.get("heldout_consulted") is False
            and controller_gate.get("operational_seed") == protocol["in_domain_policy"]["operational_seed"]
            and set(controller_gate.get("allowed_input_splits", [])) == {"dev_calibration", "dev_policy"}
            and controller_gate.get("controller_config_sha256")
            == sha256(ROOT / "configs" / "phase4" / "controller.json")
        ),
        "controller_threshold_frozen": (
            thresholds.get("heldout_consulted") is False
            and primary.get("status") == "selected"
            and primary.get("calibration_method") == expected_primary["calibration"]
            and primary.get("selection") == expected_primary["selection"]
            and float(primary.get("risk_target")) == float(expected_primary["risk_target"])
            and float(primary.get("threshold")) == float(frozen_risk["threshold"])
            and float(primary.get("risk_target")) == float(frozen_risk["risk_target"])
            and float(temperature["temperature"]) == float(frozen_gate["temperature"])
        ),
        "retrieval_frozen": (
            protocol["retrieval"] == phase4_freeze["retrieval"]
            and protocol["chunking"] == phase4_freeze["chunking"]
            and protocol["controller_ladder"] == ["dense@5", "hybrid@10", "rerank@20"]
        ),
        "generator_frozen": (
            phase4_freeze["models"]["generator"]["configured_name"]
            == protocol["llm_judge"]["model"]
        ),
        "generation_prompt_frozen": (
            sha256(generation_config_path)
            == phase4_freeze["files"]["generation_config"]["sha256"]
        ),
        "llm_judge_prompt_frozen": (
            judge_prompt.get("status") == "frozen"
            and judge_prompt.get("heldout_consulted") is False
            and judge_analysis.get("analysis_split") == "dev_policy"
            and judge_analysis.get("heldout_consulted") is False
            and judge_analysis.get("judge_threshold_sweep_performed") is False
        ),
        "adaptive_rag_protocol_frozen": (
            adaptive.get("status") == "frozen_protocol_execution_deferred"
            and adaptive.get("official_reproduction") is False
            and adaptive.get("heldout_consulted") is False
            and adaptive.get("config_sha256") == sha256(protocol_path)
        ),
        "score_threshold_baseline_frozen": (
            score.get("selection_split") == "dev_policy"
            and score.get("heldout_consulted") is False
        ),
        "metrics_frozen": code_is_committed(protected_code),
    }
    checks["no_outstanding_dev_tuning"] = (
        checks["llm_judge_prompt_frozen"]
        and checks["adaptive_rag_protocol_frozen"]
        and checks["metrics_frozen"]
    )
    consumed = [controller_gate, thresholds, score, judge_prompt, judge_analysis, adaptive]
    heldout_consulted = any(item.get("heldout_consulted") is not False for item in consumed)
    ready = all(checks.values()) and not heldout_consulted

    output_path = ROOT / "results" / "phase4" / "heldout_unlock_manifest.json"
    report_path = ROOT / "docs" / "phase4" / "heldout_unlock_checklist.md"
    existing = [path for path in (output_path, report_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite heldout unlock audit: {existing}")
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": 4,
        **checks,
        "heldout_consulted": heldout_consulted,
        "ready_for_heldout": ready,
        "ready_for_heldout_label": "YES" if ready else "NO",
        "human_confirmation_required": True,
        "human_confirmation_recorded": False,
        "heldout_execution_started": False,
        "status": "ready_for_human_confirmation" if ready else "not_ready",
        "frozen_primary_policy": primary,
        "protected_code": {
            path: sha256(ROOT / path) for path in protected_code
        },
        "sources": {
            "phase4_freeze": portable_path(phase4_freeze_path, ROOT),
            "controller_dev_gate": portable_path(controller_gate_path, ROOT),
            "controller_thresholds": portable_path(thresholds_path, ROOT),
            "score_thresholds": portable_path(score_path, ROOT),
            "llm_judge_prompt_gate": portable_path(judge_prompt_path, ROOT),
            "llm_judge_analysis": portable_path(judge_analysis_path, ROOT),
            "adaptive_rag_protocol": portable_path(adaptive_path, ROOT),
        },
        "git_commit": git_commit(ROOT),
    }
    write_json_atomic(output_path, payload)
    lines = [
        "# Phase 4 Heldout Unlock Checklist",
        "",
        "This audit reads only frozen configuration and Train-derived Dev manifests. It does not read or run heldout data.",
        "",
        "| Requirement | Status |",
        "|---|---|",
    ]
    for key, passed in checks.items():
        lines.append(f"| `{key}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            f"| `heldout_consulted == false` | {'PASS' if not heldout_consulted else 'FAIL'} |",
            "",
            f"## READY_FOR_HELDOUT = {'YES' if ready else 'NO'}",
            "",
            "Even when this checklist is fully green, heldout remains operationally locked until the researcher records explicit human confirmation. This script never starts heldout evaluation.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"READY_FOR_HELDOUT = {'YES' if ready else 'NO'}")


if __name__ == "__main__":
    main()

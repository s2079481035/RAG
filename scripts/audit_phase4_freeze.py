"""Verify and record the immutable Phase 3B inputs used by Phase 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dense-model", type=Path)
    parser.add_argument("--reranker-model", type=Path)
    parser.add_argument("--generator-model", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tagged_commit(tag: str) -> str:
    completed = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def model_record(path: Path | None, configured_name: str) -> dict:
    record = {"configured_name": configured_name, "local_path": None, "config_sha256": None}
    if path is None:
        return record
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Configured local model does not exist: {resolved}")
    config_path = resolved / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Local model lacks config.json: {resolved}")
    record.update(
        {
            "local_path": str(resolved),
            "config_sha256": sha256(config_path),
        }
    )
    return record


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = config["phase3b_freeze"]
    actual_tag_commit = tagged_commit(frozen["tag"])
    expected_prefix = frozen["commit"]
    if not actual_tag_commit.startswith(expected_prefix):
        raise ValueError(
            f"Freeze tag {frozen['tag']} points to {actual_tag_commit}, expected {expected_prefix}"
        )

    final_selection_path = ROOT / frozen["final_controller_selection"]
    risk_gate_path = ROOT / frozen["risk_policy"]
    generation_path = ROOT / frozen["generation_config"]
    final_selection = json.loads(final_selection_path.read_text(encoding="utf-8"))
    risk_gate = json.loads(risk_gate_path.read_text(encoding="utf-8"))
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if final_selection["selected_group"] != "score_aware_coverage_auxiliary":
        raise ValueError("Unexpected frozen Phase 3A Controller selection")
    primary = risk_gate["primary_selected_threshold"]
    if (
        primary["calibration_method"] != "temperature"
        or primary["selection"] != "conservative"
        or float(primary["risk_target"]) != 0.1
        or float(primary["threshold"]) != 0.93
    ):
        raise ValueError("Unexpected frozen Phase 3B primary policy")
    if risk_gate["test_consulted"] is not False:
        raise ValueError("Phase 3B gate must show that Test was not consulted during selection")

    checkpoint = (
        args.checkpoint
        or ROOT / config["controller"]["hotpotqa_checkpoint"]
    ).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Frozen Controller checkpoint is required for the audit: {checkpoint}"
        )
    temperature_path = ROOT / "results" / "phase3b" / "temperature_scaling.json"
    temperature = json.loads(temperature_path.read_text(encoding="utf-8"))
    if float(temperature["temperature"]) != float(config["controller"]["hotpotqa_temperature"]):
        raise ValueError("Phase 4 temperature differs from frozen Phase 3B")

    output_dir = ROOT / "results" / "phase4"
    manifest_path = output_dir / "freeze_manifest.json"
    report_path = ROOT / "docs" / "phase4" / "freeze_audit.md"
    if (manifest_path.exists() or report_path.exists()) and not args.force:
        raise FileExistsError("Refusing to overwrite Phase 4 freeze audit")
    files = {
        "phase4_protocol": args.config.resolve(),
        "final_controller_selection": final_selection_path,
        "risk_gate": risk_gate_path,
        "temperature_scaling": temperature_path,
        "generation_config": generation_path,
        "answer_evaluator": ROOT / "scripts" / "phase3a_generation_utils.py",
        "latency_retrieval": ROOT / "scripts" / "benchmark_phase3b_retrieval.py",
        "latency_controller": ROOT / "scripts" / "benchmark_phase3b_controller.py",
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": 4,
        "git_commit": git_commit(ROOT),
        "phase3b_tag": frozen["tag"],
        "phase3b_tag_commit": actual_tag_commit,
        "checkpoint": {
            "path": portable_path(checkpoint, ROOT),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "controller": {
            "selected_group": final_selection["selected_group"],
            "packing": config["controller"]["representation"],
            "evidence_mode": config["controller"]["evidence_mode"],
            "coverage_auxiliary": config["controller"]["coverage_auxiliary"],
            "coverage_lambda": config["controller"]["coverage_lambda"],
            "sampling": config["controller"]["sampling"],
            "temperature": temperature["temperature"],
            "primary_policy": primary,
        },
        "retrieval": config["retrieval"],
        "chunking": config["chunking"],
        "generation": generation,
        "models": {
            "dense": model_record(args.dense_model, config["retrieval"]["dense_model"]),
            "reranker": model_record(
                args.reranker_model, config["retrieval"]["reranker_model"]
            ),
            "generator": model_record(args.generator_model, generation["model"]),
        },
        "files": {
            name: {"path": portable_path(path, ROOT), "sha256": sha256(path)}
            for name, path in files.items()
        },
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(manifest_path, manifest)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Phase 3B Freeze Audit for Phase 4",
                "",
                f"- Frozen tag: `{frozen['tag']}`",
                f"- Frozen commit: `{actual_tag_commit}`",
                f"- Controller checkpoint SHA-256: `{manifest['checkpoint']['sha256']}`",
                f"- Final Controller: `{final_selection['selected_group']}`",
                f"- Packing: `{config['controller']['representation']}`",
                f"- Coverage auxiliary lambda: `{config['controller']['coverage_lambda']}`",
                f"- Temperature: `{temperature['temperature']}`",
                f"- Primary threshold: `{primary['threshold']}`",
                f"- Generator: `{generation['model']}`",
                f"- Chunk variant: `{config['chunking']['default_candidate']}`",
                "",
                "Phase 4 must consume these values without rewriting any Phase 1-3B formal result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(manifest_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

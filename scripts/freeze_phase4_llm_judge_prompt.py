"""Freeze the Phase 4 LLM Judge prompt after a format-only calibration check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
except ModuleNotFoundError:  # Imported as scripts.* in tests.
    from scripts.experiment_utils import git_commit, portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"


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


def object_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_prompt_check(manifest: dict, representation: str, judge: dict) -> dict:
    if manifest.get("split") != "dev_calibration":
        raise ValueError("Prompt format checks may only use dev_calibration")
    if manifest.get("heldout_consulted") is not False:
        raise ValueError("Prompt check manifest does not prove heldout isolation")
    if manifest.get("purpose") != "prompt_check":
        raise ValueError("Unexpected LLM Judge run purpose")
    if manifest.get("representation") != representation:
        raise ValueError(f"Wrong representation manifest for {representation}")
    if manifest.get("questions") != int(judge["prompt_check_questions"]):
        raise ValueError("Prompt check question count changed")
    if manifest.get("actionable_stages") != judge["actionable_stages"]:
        raise ValueError("Prompt check actionable stages changed")
    if manifest.get("llm_judge_config_sha256") != object_sha256(judge):
        raise ValueError("LLM Judge configuration changed during prompt check")
    parse_rate = float(manifest["strict_parse_rate"])
    if parse_rate < float(judge["minimum_parse_rate"]):
        raise ValueError(
            f"{representation} strict parse rate {parse_rate:.4f} is below "
            f"{float(judge['minimum_parse_rate']):.4f}"
        )
    return {
        "representation": representation,
        "questions": manifest["questions"],
        "actionable_states": manifest["actionable_states"],
        "strict_parse_rate": parse_rate,
        "invalid_outputs": manifest["invalid_outputs"],
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    judge = config["llm_judge"]
    if judge["prompt_check_split"] != "dev_calibration":
        raise ValueError("Prompt freeze is restricted to dev_calibration")
    output_dir = ROOT / "results" / "phase4" / "2wiki" / "llm_judge"
    output_path = output_dir / "prompt_freeze_manifest.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite frozen prompt gate: {output_path}")

    prompt_path = ROOT / judge["prompt_path"]
    checks = []
    source_manifests = []
    model_config_hashes = set()
    for representation in judge["representations"]:
        manifest_path = output_dir / "prompt_check" / f"{representation}_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("prompt_sha256") != file_sha256(prompt_path):
            raise ValueError("Prompt changed between format checks")
        checks.append(validate_prompt_check(manifest, representation, judge))
        source_manifests.append(
            {
                "representation": representation,
                "path": portable_path(manifest_path, ROOT),
                "sha256": file_sha256(manifest_path),
            }
        )
        model_config_hashes.add(manifest["model_config_sha256"])
    if len(model_config_hashes) != 1:
        raise ValueError("Prompt checks did not use one frozen Judge model")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "status": "frozen",
            "freeze_basis": "format_parseability_only_not_label_quality",
            "selection_split": "dev_calibration",
            "heldout_consulted": False,
            "minimum_parse_rate": float(judge["minimum_parse_rate"]),
            "prompt": portable_path(prompt_path, ROOT),
            "prompt_sha256": file_sha256(prompt_path),
            "llm_judge_config_sha256": object_sha256(judge),
            "model_config_sha256": next(iter(model_config_hashes)),
            "invalid_output_action": judge["invalid_output_action"],
            "checks": checks,
            "source_manifests": source_manifests,
            "git_commit": git_commit(ROOT),
        },
    )
    print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

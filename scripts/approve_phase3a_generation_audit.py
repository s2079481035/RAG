"""Validate the human Dev generation audit and create the immutable test gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiment_utils import portable_path, utc_now, write_json_atomic
from phase3a_generation_utils import canonical_rows_sha256, file_sha256, load_config


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3a" / "generation.json"
DEFAULT_AUDIT = ROOT / "results" / "phase3" / "generation" / "dev" / "manual_audit_50.csv"
DEFAULT_AUDIT_MANIFEST = DEFAULT_AUDIT.with_suffix(".manifest.json")
DEFAULT_GENERATION_MANIFEST = ROOT / "results" / "phase3" / "generation" / "dev" / "generation_manifest.json"
DEFAULT_OUTPUT = ROOT / "results" / "phase3" / "generation" / "dev" / "test_gate_approval.json"
YES = {"yes", "y", "1", "true"}
NO = {"no", "n", "0", "false"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST)
    parser.add_argument("--generation-manifest", type=Path, default=DEFAULT_GENERATION_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalized_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in YES:
        return True
    if normalized in NO:
        return False
    raise ValueError(f"Expected yes/no audit value, got {value!r}")


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite generation test gate: {args.output}")
    config = load_config(args.config)
    creation = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    generation_manifest = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
    config_hash = file_sha256(args.config)
    if creation["generation_config_sha256"] != config_hash:
        raise ValueError("Generation config changed after the Dev audit was sampled")
    if generation_manifest.get("split") != "dev":
        raise ValueError("Generation approval must be tied to a Dev generation manifest")
    if generation_manifest.get("generation_config_sha256") != config_hash:
        raise ValueError("Dev generation manifest used a different configuration")
    details_path = Path(creation["evaluation_details"])
    if not details_path.is_absolute():
        details_path = ROOT / details_path
    if not details_path.exists() or file_sha256(details_path) != creation["evaluation_details_sha256"]:
        raise ValueError("Dev evaluation details changed after the audit was sampled")
    with args.audit.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_count = int(config["manual_dev_audit_count"])
    if len(rows) != required_count:
        raise ValueError(f"Expected {required_count} audit rows, got {len(rows)}")
    immutable_fields = creation["immutable_fields"]
    if canonical_rows_sha256(rows, immutable_fields) != creation["immutable_fields_sha256"]:
        raise ValueError("Immutable generation audit fields changed during human review")
    incomplete = [
        row["audit_id"] for row in rows
        if not row["human_extraction_correct"].strip()
        or not row["human_scoring_reasonable"].strip()
        or not row["reviewer"].strip()
    ]
    if incomplete:
        raise ValueError(f"Generation audit has {len(incomplete)} incomplete rows")
    extraction_failures = [
        row["audit_id"] for row in rows if not normalized_boolean(row["human_extraction_correct"])
    ]
    scoring_failures = [
        row["audit_id"] for row in rows if not normalized_boolean(row["human_scoring_reasonable"])
    ]
    status = "approved" if not extraction_failures and not scoring_failures else "rejected"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "status": status,
            "selection_split": "dev",
            "test_consulted": False,
            "reviewed_samples": len(rows),
            "generation_config_sha256": config_hash,
            "audit": portable_path(args.audit.resolve(), ROOT),
            "audit_sha256_after_review": file_sha256(args.audit),
            "source_evaluation_details_sha256": creation["evaluation_details_sha256"],
            "dev_generation_manifest": portable_path(args.generation_manifest.resolve(), ROOT),
            "dev_generation_manifest_sha256": file_sha256(args.generation_manifest),
            "model": generation_manifest["model"],
            "model_revision": generation_manifest.get("model_revision"),
            "extraction_failure_ids": extraction_failures,
            "scoring_failure_ids": scoring_failures,
            "reviewers": sorted({row["reviewer"].strip() for row in rows}),
        },
    )
    print(status)
    if status != "approved":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

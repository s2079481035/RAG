"""Record the audited, adapted Adaptive-RAG protocol without running heldout."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    adaptive = config["external_baselines"]["adaptive_rag"]
    expected_mapping = {
        "light": "dense@5",
        "medium": "hybrid@10",
        "heavy": "rerank@20",
    }
    if adaptive["formal_name"] != "Adaptive-RAG-style Query Complexity Router":
        raise ValueError("Adapted baseline must not be named an official reproduction")
    if adaptive["strict_official_reproduction"] is not False:
        raise ValueError("Current ladder cannot be marked as an official reproduction")
    if adaptive["router_input"] != "question_only" or adaptive["route_mapping"] != expected_mapping:
        raise ValueError("Adapted router input or mapping changed")
    if adaptive["development_splits"] != ["dev_calibration", "dev_policy"]:
        raise ValueError("Adaptive protocol may only use Train-derived Dev splits")
    if adaptive["heldout_tuning_forbidden"] is not True:
        raise ValueError("Adaptive protocol does not protect heldout")

    audit_path = ROOT / "docs" / "phase4" / "adaptive_rag_protocol_audit.md"
    output_path = ROOT / "results" / "phase4" / "2wiki" / "adaptive_rag_protocol_manifest.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite adaptive protocol freeze: {output_path}")
    write_json_atomic(
        output_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "status": "frozen_protocol_execution_deferred",
            "formal_name": adaptive["formal_name"],
            "official_reproduction": False,
            "adaptation_reason": "official no/single/iterative ladder is not equivalent to dense/hybrid/rerank",
            "router_input": adaptive["router_input"],
            "route_mapping": adaptive["route_mapping"],
            "training_label": adaptive["training_label"],
            "training_split": adaptive["training_split"],
            "development_splits": adaptive["development_splits"],
            "heldout_consulted": False,
            "execution_status": adaptive["execution_status"],
            "audit": portable_path(audit_path, ROOT),
            "audit_sha256": sha256(audit_path),
            "config": portable_path(args.config, ROOT),
            "config_sha256": sha256(args.config),
            "git_commit": git_commit(ROOT),
        },
    )
    print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

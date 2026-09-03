"""Record the pre-registered dev-only Phase 2 chunk selection and unlock test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase2" / "chunk_retrieval.json"
DEFAULT_SUMMARY = ROOT / "results" / "phase2" / "retrieval_eval" / "dev" / "retrieval_summary.csv"
DEFAULT_DECISION = ROOT / "results" / "phase2" / "chunk_selection.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.decision.exists():
        raise FileExistsError(f"Refusing to overwrite chunk selection: {args.decision}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluation = config["retrieval_evaluation"]
    if evaluation["selected_variant_after_dev"] is not None:
        raise ValueError("A chunk variant has already been recorded in the config")
    rule = evaluation["selection_rule"]
    with args.summary.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in all_rows
        if row["split"] == evaluation["selection_split"]
        and row["method"] == rule["method"]
        and int(row["k"]) == int(rule["k"])
        and row["variant"] in evaluation["selection_variants"]
    ]
    if {row["variant"] for row in candidates} != set(evaluation["selection_variants"]):
        raise ValueError("Dev summary does not contain every pre-registered chunk variant")
    candidates.sort(
        key=lambda row: (
            -float(row[rule["primary_maximize"]]),
            -float(row[rule["secondary_maximize"]]),
            float(row[rule["tertiary_minimize"]]),
            row["variant"],
        )
    )
    selected = candidates[0]["variant"]
    decision = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "git_commit_before_selection": git_commit(ROOT),
        "selection_split": "dev",
        "test_consulted": False,
        "selection_rule": rule,
        "candidate_rows": candidates,
        "selected_variant": selected,
        "source_summary": portable_path(args.summary, ROOT),
    }
    write_json_atomic(args.decision, decision)
    evaluation["selected_variant_after_dev"] = selected
    write_json_atomic(args.config, config)
    print(selected)


if __name__ == "__main__":
    main()

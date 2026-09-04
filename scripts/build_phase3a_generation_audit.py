"""Build a deterministic 50-example Dev audit for generation and scoring."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path

from experiment_utils import portable_path
from phase3a_generation_utils import canonical_rows_sha256, file_sha256, load_config


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3a" / "generation.json"
DEFAULT_INPUT = ROOT / "results" / "phase3" / "generation" / "dev" / "evaluation_details.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "phase3" / "generation" / "dev" / "manual_audit_50.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_stratum(row: dict) -> str:
    if row["has_multiline_raw_output"] or row["has_long_raw_output"]:
        return "verbose_or_multiline"
    if row["exact_match"]:
        return "exact_match"
    if float(row["token_f1"]) == 0.0:
        return "zero_f1"
    return "partial_f1"


def stratified_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[audit_stratum(row)].append(row)
    queues = []
    for name in sorted(groups):
        values = groups[name]
        random.Random(f"{seed}:{name}").shuffle(values)
        queues.append((name, deque(values)))
    selected = []
    while queues and len(selected) < min(count, len(rows)):
        remaining = []
        for name, queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                remaining.append((name, queue))
        queues = remaining
    return selected


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite generation audit: {args.output}")
    config = load_config(args.config)
    rows = read_jsonl(args.input)
    if not rows or {row["split"] for row in rows} != {"dev"}:
        raise ValueError("Generation manual audit must be built from Dev evaluation details")
    count = int(config["manual_dev_audit_count"])
    if len(rows) < count:
        raise ValueError(f"Need at least {count} Dev generations, got {len(rows)}")
    selected = stratified_sample(rows, count, int(config["seed"]))
    fields = [
        "audit_id",
        "question_id",
        "stratum",
        "question",
        "gold_answer",
        "raw_output",
        "extracted_answer",
        "exact_match",
        "token_f1",
        "output_tokens",
        "raw_output_lines",
        "human_extraction_correct",
        "human_scoring_reasonable",
        "human_notes",
        "reviewer",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        audit_rows = []
        for index, row in enumerate(selected, start=1):
            audit_row = {
                "audit_id": index,
                "question_id": row["question_id"],
                "stratum": audit_stratum(row),
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "raw_output": row["raw_output"],
                "extracted_answer": row["extracted_answer"],
                "exact_match": row["exact_match"],
                "token_f1": row["token_f1"],
                "output_tokens": row["output_tokens"],
                "raw_output_lines": row["raw_output_lines"],
                "human_extraction_correct": "",
                "human_scoring_reasonable": "",
                "human_notes": "",
                "reviewer": "",
            }
            audit_rows.append(audit_row)
            writer.writerow(audit_row)
    temporary.replace(args.output)
    metadata = {
        "generation_config_sha256": file_sha256(args.config),
        "evaluation_details": portable_path(args.input.resolve(), ROOT),
        "evaluation_details_sha256": file_sha256(args.input),
        "audit_csv_sha256_at_creation": file_sha256(args.output),
        "immutable_fields": fields[:11],
        "immutable_fields_sha256": canonical_rows_sha256(audit_rows, fields[:11]),
        "sample_count": len(selected),
        "seed": config["seed"],
        "strata": sorted({audit_stratum(row) for row in selected}),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

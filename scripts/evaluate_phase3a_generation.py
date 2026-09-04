"""Evaluate frozen Phase 3A HotpotQA generations with standard EM and F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_utils import portable_path, utc_now, write_json_atomic
from phase3a_generation_utils import exact_match_score, extract_short_answer, normalize_answer, token_f1_score


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("Generation prediction file is empty")
    output_dir = (args.output_dir or args.input.parent).resolve()
    details_path = output_dir / "evaluation_details.jsonl"
    metrics_path = output_dir / "evaluation_metrics.json"
    report_path = output_dir / "evaluation.md"
    existing = [path for path in [details_path, metrics_path, report_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite generation evaluation: {existing}")
    details = []
    for row in rows:
        extracted = extract_short_answer(row["raw_output"])
        if extracted != row["extracted_answer"]:
            raise ValueError(f"Stored extraction differs from frozen rule: {row['question_id']}")
        em = exact_match_score(extracted, row["gold_answer"])
        f1 = token_f1_score(extracted, row["gold_answer"])
        details.append(
            {
                **row,
                "normalized_prediction": normalize_answer(extracted),
                "normalized_gold": normalize_answer(row["gold_answer"]),
                "exact_match": em,
                "token_f1": f1,
                "has_multiline_raw_output": row["raw_output_lines"] > 1,
                "has_long_raw_output": row["output_tokens"] > 16,
            }
        )
    metrics = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "split": rows[0]["split"],
        "samples": len(details),
        "exact_match": sum(row["exact_match"] for row in details) / len(details),
        "token_f1": sum(row["token_f1"] for row in details) / len(details),
        "multiline_raw_output_count": sum(row["has_multiline_raw_output"] for row in details),
        "long_raw_output_count": sum(row["has_long_raw_output"] for row in details),
        "empty_extracted_answer_count": sum(not row["extracted_answer"] for row in details),
        "context_truncated_count": sum(row["context_truncated"] for row in details),
        "input": portable_path(args.input.resolve(), ROOT),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(details_path, details)
    write_json_atomic(metrics_path, metrics)
    report_path.write_text(
        "\n".join(
            [
                f"# Phase 3A Generation Evaluation ({metrics['split']})",
                "",
                f"- Samples: {metrics['samples']}",
                f"- Exact Match: {metrics['exact_match']:.4f}",
                f"- Token-level F1: {metrics['token_f1']:.4f}",
                f"- Multiline raw outputs: {metrics['multiline_raw_output_count']}",
                f"- Raw outputs longer than 16 tokens: {metrics['long_raw_output_count']}",
                f"- Empty extracted answers: {metrics['empty_extracted_answer_count']}",
                f"- Truncated contexts: {metrics['context_truncated_count']}",
                "",
                "Scoring uses the frozen first-line extraction rule and standard HotpotQA lower/punctuation/article/whitespace normalization.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

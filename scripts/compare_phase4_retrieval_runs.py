"""Verify that a throughput-optimized retrieval run preserves saved retrieval outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_by_question(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["question_id"]
            if qid in rows:
                raise ValueError(f"Duplicate question_id {qid!r} in {path}")
            rows[qid] = row
    return rows


def compare_value(left, right, path: str, atol: float, differences: list[str]) -> None:
    if isinstance(left, float) or isinstance(right, float):
        try:
            matches = math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            differences.append(f"{path}: {left!r} != {right!r}")
        return
    if type(left) is not type(right):
        differences.append(f"{path}: types {type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, dict):
        if set(left) != set(right):
            differences.append(f"{path}: keys {sorted(left)} != {sorted(right)}")
            return
        for key in sorted(left):
            compare_value(left[key], right[key], f"{path}.{key}", atol, differences)
        return
    if isinstance(left, list):
        if len(left) != len(right):
            differences.append(f"{path}: lengths {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_value(left_item, right_item, f"{path}[{index}]", atol, differences)
        return
    if left != right:
        differences.append(f"{path}: {left!r} != {right!r}")


def main() -> None:
    args = parse_args()
    if args.atol < 0:
        raise ValueError("--atol must be non-negative")
    left_rows = read_by_question(args.left)
    right_rows = read_by_question(args.right)
    if set(left_rows) != set(right_rows):
        missing_left = sorted(set(right_rows) - set(left_rows))
        missing_right = sorted(set(left_rows) - set(right_rows))
        raise ValueError(
            f"Question sets differ; missing_left={missing_left[:5]}, "
            f"missing_right={missing_right[:5]}"
        )
    differences: list[str] = []
    for qid in sorted(left_rows):
        left = {key: value for key, value in left_rows[qid].items() if key != "latency_ms"}
        right = {key: value for key, value in right_rows[qid].items() if key != "latency_ms"}
        compare_value(left, right, qid, args.atol, differences)
        if len(differences) >= 20:
            break
    result = {
        "schema_version": 1,
        "left": str(args.left),
        "right": str(args.right),
        "questions": len(left_rows),
        "absolute_float_tolerance": args.atol,
        "latency_ignored": True,
        "matches": not differences,
        "first_differences": differences[:20],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

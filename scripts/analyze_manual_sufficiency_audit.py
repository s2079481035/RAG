"""Analyze a completed Phase 2 manual sufficiency audit without changing labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "docs" / "manual_sufficiency_audit.csv"
DEFAULT_REPORT = ROOT / "docs" / "manual_audit_analysis.md"
DEFAULT_CONFUSION = ROOT / "results" / "phase3" / "manual_audit_confusion.csv"
THREE_CLASS = ["insufficient", "partial", "sufficient"]
YES = {"yes", "y", "1", "true"}
NO = {"no", "n", "0", "false"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--confusion", type=Path, default=DEFAULT_CONFUSION)
    parser.add_argument("--expected-count", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_human_stop(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in YES:
        return 1
    if normalized in NO:
        return 0
    raise ValueError(f"Expected yes/no human sufficiency value, got {value!r}")


def read_completed_audit(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manual audit is empty")
    required = {
        "audit_id",
        "automatic_three_class_label",
        "automatic_stop_label",
        "human_sufficient_to_answer",
        "human_label",
        "human_notes",
        "reviewer",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manual audit is missing columns: {sorted(missing)}")
    incomplete = [
        row["audit_id"]
        for row in rows
        if not row["human_sufficient_to_answer"].strip()
        or not row["human_label"].strip()
        or not row["reviewer"].strip()
    ]
    if incomplete:
        preview = ", ".join(incomplete[:10])
        raise ValueError(f"Manual audit has {len(incomplete)} incomplete rows; first IDs: {preview}")
    for row in rows:
        human_three = row["human_label"].strip().lower()
        if human_three not in THREE_CLASS:
            raise ValueError(f"audit_id={row['audit_id']} has invalid human_label={human_three!r}")
        automatic_three = row["automatic_three_class_label"].strip().lower()
        if automatic_three not in THREE_CLASS:
            raise ValueError(
                f"audit_id={row['audit_id']} has invalid automatic label={automatic_three!r}"
            )
        human_stop = normalize_human_stop(row["human_sufficient_to_answer"])
        if human_stop != int(human_three == "sufficient"):
            raise ValueError(
                f"audit_id={row['audit_id']} has inconsistent human binary and three-class labels"
            )
        row["human_label"] = human_three
        row["human_stop_label"] = human_stop
        row["automatic_three_class_label"] = automatic_three
        row["automatic_stop_label"] = int(row["automatic_stop_label"])
        if row["automatic_stop_label"] != int(automatic_three == "sufficient"):
            raise ValueError(
                f"audit_id={row['audit_id']} has inconsistent automatic binary and three-class labels"
            )
        if automatic_three != human_three and not row["human_notes"].strip():
            raise ValueError(
                f"audit_id={row['audit_id']} disagrees but has no human_notes explanation"
            )
    return rows


def classification_summary(actual: list, predicted: list, labels: list) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def write_confusions(path: Path, binary: dict, three_class: dict) -> None:
    rows = []
    for task, summary, labels in [
        ("stop_binary", binary, ["continue", "stop"]),
        ("three_class", three_class, THREE_CLASS),
    ]:
        for actual_index, actual in enumerate(labels):
            for predicted_index, predicted in enumerate(labels):
                rows.append(
                    {
                        "task": task,
                        "human_label": actual,
                        "automatic_label": predicted,
                        "count": summary["confusion_matrix"][actual_index][predicted_index],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def render_report(rows: list[dict], binary: dict, three_class: dict) -> str:
    automatic_sufficient_human_not = [
        row for row in rows
        if row["automatic_three_class_label"] == "sufficient" and row["human_label"] != "sufficient"
    ]
    automatic_continue_human_sufficient = [
        row for row in rows
        if row["automatic_stop_label"] == 0 and row["human_stop_label"] == 1
    ]
    transitions = Counter(
        (row["automatic_three_class_label"], row["human_label"])
        for row in rows
        if row["automatic_three_class_label"] != row["human_label"]
    )
    sufficient = three_class["per_class"]["sufficient"]
    visible_groups = {"all_gold_visible": [], "some_gold_not_visible": [], "unknown": []}
    for row in rows:
        raw = row.get("visible_supporting_fact_ratio_evaluation_only", "").strip()
        key = "unknown"
        if raw:
            key = "all_gold_visible" if np.isclose(float(raw), 1.0) else "some_gold_not_visible"
        visible_groups[key].append(int(row["automatic_stop_label"] == row["human_stop_label"]))
    lines = [
        "# Manual Sufficiency Audit Analysis",
        "",
        "This report evaluates whether automatic gold supporting-fact coverage is a reasonable proxy for human answerability. It does not modify any dev or test label.",
        "",
        "## Audit Scope",
        "",
        f"- Completed dev audit records: {len(rows)}",
        "- Human stop label is treated as reference; the automatic coverage-derived label is treated as prediction.",
        "- `retrieved_evidence` contains full retrieved chunks. The Controller saw packed evidence, which may be shorter.",
        "",
        "## Agreement Metrics",
        "",
        "| Comparison | Accuracy | Macro F1 | Sufficient precision | Sufficient recall |",
        "|---|---:|---:|---:|---:|",
        f"| Automatic Stop vs Human Stop | {binary['accuracy']:.4f} | {binary['macro_f1']:.4f} | {binary['per_class']['1']['precision']:.4f} | {binary['per_class']['1']['recall']:.4f} |",
        f"| Automatic three-class vs Human three-class | {three_class['accuracy']:.4f} | {three_class['macro_f1']:.4f} | {sufficient['precision']:.4f} | {sufficient['recall']:.4f} |",
        "",
        "## Priority Disagreements",
        "",
        f"- Automatic Sufficient -> Human Not Sufficient: {len(automatic_sufficient_human_not)}",
        f"- Automatic Continue -> Human Sufficient: {len(automatic_continue_human_sufficient)}",
        "",
        "| Automatic label | Human label | Count |",
        "|---|---|---:|",
    ]
    for (automatic, human), count in sorted(transitions.items()):
        lines.append(f"| {automatic} | {human} | {count} |")
    lines.extend(["", "## Full Evidence vs Model-visible Evidence", ""])
    for name, values in visible_groups.items():
        agreement = sum(values) / len(values) if values else None
        rendered = "N/A" if agreement is None else f"{agreement:.4f}"
        lines.append(f"- `{name}`: n={len(values)}, binary agreement={rendered}")
    lines.extend(
        [
            "",
            "## Human Disagreement Notes",
            "",
            "The following notes are evidence for the researcher's qualitative synthesis; they are not generated label corrections.",
            "",
            "| Audit ID | Automatic -> Human | Human note |",
            "|---:|---|---|",
        ]
    )
    for row in rows:
        if row["automatic_three_class_label"] == row["human_label"]:
            continue
        note = row["human_notes"].replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()
        lines.append(
            f"| {row['audit_id']} | {row['automatic_three_class_label']} -> {row['human_label']} | {note or 'No note supplied'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These results validate or challenge the answerability proxy on dev only. They must not be used to relabel test, tune a test threshold, or claim that the model saw every token shown in the audit sheet.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    existing = [path for path in [args.report, args.confusion] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite manual audit analysis: {existing}")
    rows = read_completed_audit(args.input)
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} audit records, got {len(rows)}")
    human_stop = [row["human_stop_label"] for row in rows]
    automatic_stop = [row["automatic_stop_label"] for row in rows]
    human_three = [row["human_label"] for row in rows]
    automatic_three = [row["automatic_three_class_label"] for row in rows]
    binary = classification_summary(human_stop, automatic_stop, [0, 1])
    three_class = classification_summary(human_three, automatic_three, THREE_CLASS)
    write_confusions(args.confusion, binary, three_class)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(rows, binary, three_class), encoding="utf-8")


if __name__ == "__main__":
    main()

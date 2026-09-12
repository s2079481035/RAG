"""Compare frozen LLM Judges with the 2Wiki evidence-aware Controller on Dev."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from analyze_phase4_retrieval_ceiling import controller_recoverability_metrics
    from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
    from phase3a_metrics import coverage_bucket, metrics_from_decisions
except ModuleNotFoundError:  # Imported as scripts.* in tests.
    from scripts.analyze_phase4_retrieval_ceiling import controller_recoverability_metrics
    from scripts.experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
    from scripts.phase3a_metrics import coverage_bucket, metrics_from_decisions


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def hard_actionable_metrics(rows: list[dict]) -> dict:
    labels = np.asarray([int(row["actual_stop_label"]) for row in rows], dtype=int)
    predictions = np.asarray([int(row["predicted_stop_label"]) for row in rows], dtype=int)
    metrics = metrics_from_decisions(labels, predictions, predictions.astype(float))
    metrics["auroc"] = None
    hard = np.asarray(
        [
            int(row["actual_stop_label"]) == 0
            and coverage_bucket(float(row["true_coverage"])) == "hard_partial"
            for row in rows
        ],
        dtype=bool,
    )
    hard_false = int(np.sum(hard & (predictions == 1)))
    metrics["hard_partial_continue_count"] = int(np.sum(hard))
    metrics["hard_partial_false_stop_count"] = hard_false
    metrics["hard_partial_false_stop_rate"] = (
        hard_false / int(np.sum(hard)) if np.any(hard) else None
    )
    return metrics


def hard_sequential_metrics(rows: list[dict], ladder: list[str]) -> tuple[dict, list[dict]]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["question_id"]][row["stage"]] = row
    reached = []
    for question_id in sorted(grouped):
        for stage in ladder[:-1]:
            row = grouped[question_id][stage]
            reached.append(row)
            if int(row["predicted_stop_label"]):
                break
    return hard_actionable_metrics(reached), reached


def benchmark_index(path: Path, latency_key: str) -> dict:
    return {
        (row["question_id"], row["stage"]): {
            "latency_ms": float(row[latency_key]),
            "input_tokens": float(row["input_tokens"]),
            "output_tokens": float(row.get("output_tokens", 0)),
        }
        for row in read_jsonl(path)
    }


def reached_cost(reached: list[dict], benchmark: dict) -> dict:
    per_question = defaultdict(lambda: {"latency": 0.0, "input": 0.0, "output": 0.0})
    for row in reached:
        key = (row["question_id"], row["stage"])
        measured = benchmark[key]
        totals = per_question[row["question_id"]]
        totals["latency"] += measured["latency_ms"]
        totals["input"] += measured["input_tokens"]
        totals["output"] += measured["output_tokens"]
    decisions = len(reached)
    questions = len(per_question)
    return {
        "mean_decision_latency_ms": sum(v["latency"] for v in per_question.values()) / decisions,
        "mean_latency_per_question_ms": sum(v["latency"] for v in per_question.values()) / questions,
        "mean_input_tokens_per_decision": sum(v["input"] for v in per_question.values()) / decisions,
        "mean_input_tokens_per_question": sum(v["input"] for v in per_question.values()) / questions,
        "mean_output_tokens_per_decision": sum(v["output"] for v in per_question.values()) / decisions,
        "mean_output_tokens_per_question": sum(v["output"] for v in per_question.values()) / questions,
    }


def critic_rows(path: Path, probability_key: str, threshold: float, actionable: list[str]) -> list[dict]:
    rows = []
    for source in read_jsonl(path):
        if source["stage"] not in actionable:
            continue
        row = dict(source)
        row["analysis_probability"] = float(source[probability_key])
        row["predicted_stop_label"] = int(row["analysis_probability"] >= threshold)
        rows.append(row)
    return rows


def summary_row(
    name: str,
    role: str,
    scope: str,
    metrics: dict,
    recoverability: dict | None,
    cost: dict,
    parse_rate: float | None,
    probability_available: bool,
    threshold: float | None,
    terminal_failure_rate: float,
) -> dict:
    return {
        "system": name,
        "comparison_role": role,
        "decision_scope": scope,
        "threshold": threshold,
        "questions": 2000,
        "decisions": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "stop_precision": metrics["stop_precision"],
        "stop_recall": metrics["stop_recall"],
        "stop_f1": metrics["stop_f1"],
        "false_stop_rate": metrics["false_stop_rate"],
        "hard_partial_false_stop_rate": metrics["hard_partial_false_stop_rate"],
        "unnecessary_escalation_rate": metrics["unnecessary_escalation_rate"],
        "recoverable_false_stop_rate": (
            recoverability["recoverable_false_stop_rate"] if recoverability else None
        ),
        "unrecoverable_early_stop_rate": (
            recoverability["unrecoverable_early_stop_rate"] if recoverability else None
        ),
        "terminal_retrieval_failure_rate": terminal_failure_rate,
        "mean_decision_latency_ms": cost["mean_decision_latency_ms"],
        "mean_latency_per_question_ms": cost["mean_latency_per_question_ms"],
        "mean_input_tokens_per_decision": cost["mean_input_tokens_per_decision"],
        "mean_input_tokens_per_question": cost["mean_input_tokens_per_question"],
        "mean_output_tokens_per_decision": cost["mean_output_tokens_per_decision"],
        "mean_output_tokens_per_question": cost["mean_output_tokens_per_question"],
        "strict_parse_rate": parse_rate,
        "label_probability_available": probability_available,
        "auroc": metrics["auroc"] if probability_available else None,
    }


def fmt(value, digits: int = 4) -> str:
    return "N/A" if value in (None, "") else f"{float(value):.{digits}f}"


def percent(value) -> str:
    return "N/A" if value in (None, "") else f"{100 * float(value):.2f}%"


def write_report(path: Path, rows: list[dict], full_gain: dict) -> None:
    sequential = [row for row in rows if row["decision_scope"] == "sequential_reached"]
    lines = [
        "# Phase 4 2Wiki LLM-as-a-Judge Analysis",
        "",
        "All formal comparisons use the same 2,000 `dev_policy` questions, cumulative evidence, and actionable stages (`dense@5`, `hybrid@10`). `rerank@20` is terminal and is excluded from Judge and Controller FSR.",
        "",
        "## Sequential Results",
        "",
        "| System | Accuracy | Macro F1 | Stop P | Stop R | FSR | Hard Partial FSR | RFSR | UER | Latency/question ms | Input tokens/question | Output tokens/question | Parse rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sequential:
        lines.append(
            f"| {row['system']} | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {fmt(row['stop_precision'])} | "
            f"{fmt(row['stop_recall'])} | {percent(row['false_stop_rate'])} | "
            f"{percent(row['hard_partial_false_stop_rate'])} | "
            f"{percent(row['recoverable_false_stop_rate'])} | "
            f"{percent(row['unnecessary_escalation_rate'])} | "
            f"{fmt(row['mean_latency_per_question_ms'], 2)} | "
            f"{fmt(row['mean_input_tokens_per_question'], 1)} | "
            f"{fmt(row['mean_output_tokens_per_question'], 1)} | "
            f"{percent(row['strict_parse_rate'])} |"
        )
    lines.extend(
        [
            "",
            "`EvidenceAwareClassification` uses the frozen seed-42 classification threshold; `EvidenceAwareRisk10` uses the separately frozen temperature-scaled risk-policy threshold. These are different operating concepts.",
            "",
            "Judge outputs are hard labels. No Judge AUROC or artificial threshold sweep is reported. Invalid output, if any, follows the preregistered safe action `Continue`.",
            "",
            "## Context Effect",
            "",
            f"Relative to Judge-512, Judge-Full changes sequential Macro F1 by {full_gain['macro_f1']:+.4f}, FSR by {100 * full_gain['false_stop_rate']:+.2f} percentage points, and mean input tokens per question by {full_gain['input_tokens_per_question']:+.1f}.",
            "",
            "## Retrieval-Limited Boundary",
            "",
            f"The frozen final-stage terminal retrieval failure rate is {percent(sequential[0]['terminal_retrieval_failure_rate'])}. It is a retriever ceiling diagnostic, not a Controller false stop. RFSR measures only incorrect early stops whose evidence could actually become sufficient later in the frozen ladder.",
            "",
            "## Interpretation",
            "",
            "- Judge accuracy and FSR must be interpreted jointly with latency and token use; a larger model is not automatically a better adaptive policy.",
            "- Judge-512 is the primary fair representation comparison. Judge-Full isolates the additional benefit of the larger context budget.",
            "- This Dev analysis freezes the baseline result and does not authorize heldout access or any retuning.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    judge = config["llm_judge"]
    ladder = config["controller_ladder"]
    actionable = config["in_domain_policy"]["decision_stages"]
    if actionable != ladder[:-1] or judge["actionable_stages"] != actionable:
        raise ValueError("Actionable stages changed from the frozen ladder")
    output_dir = ROOT / "results" / "phase4" / "2wiki" / "llm_judge"
    formal_dir = output_dir / "dev_policy"
    summary_path = ROOT / "results" / "phase4" / "2wiki" / "llm_judge_summary.csv"
    report_path = ROOT / "docs" / "phase4" / "llm_judge_analysis.md"
    manifest_path = output_dir / "analysis_manifest.json"
    existing = [path for path in (summary_path, report_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite LLM Judge analysis: {existing}")
    gate_path = output_dir / "prompt_freeze_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "frozen" or gate.get("heldout_consulted") is not False:
        raise ValueError("LLM Judge prompt is not frozen")

    with (ROOT / "results" / "phase4" / "2wiki" / "retrieval_ceiling_summary.csv").open(newline="") as handle:
        ceiling = next(row for row in csv.DictReader(handle) if row["split"] == "dev_policy")
    terminal_failure_rate = float(ceiling["terminal_retrieval_failure_rate"])
    diagnostic_path = ROOT / "results" / "phase4" / "2wiki" / "recoverability" / "dev_policy.jsonl"
    critic_prediction_path = ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "calibrated_dev_policy_predictions.jsonl"
    critic_latency_path = formal_dir / "critic_latency.jsonl"
    critic_benchmark = benchmark_index(critic_latency_path, "controller_latency_ms")

    operational_seed = int(config["in_domain_policy"]["operational_seed"])
    run_manifest_path = ROOT / "experiments" / "phase4" / "2wiki" / "final_evidence_aware" / f"seed{operational_seed}" / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    classification_threshold = float(run_manifest["selected_dev_threshold"])
    selected = json.loads(
        (ROOT / "results" / "phase4" / "2wiki" / "controller_dev" / "selected_thresholds.json").read_text(encoding="utf-8")
    )["primary_selected_policy"]
    risk_threshold = float(selected["threshold"])
    frozen_gate = config["in_domain_policy"]["frozen_dev_gate"]
    if classification_threshold != float(
        frozen_gate["classification_operating_point"]["threshold"]
    ):
        raise ValueError("Classification threshold changed after the 2Wiki Dev gate")
    frozen_risk = frozen_gate["risk_controlled_operating_point"]
    if risk_threshold != float(frozen_risk["threshold"]):
        raise ValueError("Risk-policy threshold changed after the 2Wiki Dev gate")
    if float(selected["risk_target"]) != float(frozen_risk["risk_target"]):
        raise ValueError("Risk-policy target changed after the 2Wiki Dev gate")

    systems = []
    source_files = {}
    for representation, display in (("judge_512", "Judge-512"), ("judge_full", "Judge-Full")):
        path = formal_dir / f"{representation}.jsonl"
        manifest = json.loads((formal_dir / f"{representation}_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("split") != "dev_policy" or manifest.get("heldout_consulted") is not False:
            raise ValueError(f"Invalid formal Judge manifest: {representation}")
        rows = read_jsonl(path)
        if len(rows) != 4000:
            raise ValueError(f"Expected 4000 actionable Judge states for {representation}")
        systems.append((display, "hard_label_baseline", rows, None, False, float(manifest["strict_parse_rate"]), benchmark_index(path, "judge_latency_ms")))
        source_files[representation] = {"path": portable_path(path, ROOT), "sha256": file_sha256(path)}

    systems.extend(
        [
            (
                "EvidenceAwareClassification",
                "frozen_classification_operating_point",
                critic_rows(critic_prediction_path, "raw_stop_probability", classification_threshold, actionable),
                classification_threshold,
                True,
                None,
                critic_benchmark,
            ),
            (
                "EvidenceAwareRisk10",
                "frozen_risk_controlled_operating_point",
                critic_rows(critic_prediction_path, "stop_probability", risk_threshold, actionable),
                risk_threshold,
                True,
                None,
                critic_benchmark,
            ),
        ]
    )

    output_rows = []
    sequential_by_name = {}
    for name, role, rows, threshold, probability_available, parse_rate, benchmark in systems:
        if probability_available:
            all_metrics = metrics_from_decisions(
                [int(row["actual_stop_label"]) for row in rows],
                [int(row["predicted_stop_label"]) for row in rows],
                [float(row["analysis_probability"]) for row in rows],
            )
            hard_metrics = hard_actionable_metrics(rows)
            all_metrics["hard_partial_false_stop_rate"] = hard_metrics["hard_partial_false_stop_rate"]
            prepared = []
            indexed = defaultdict(dict)
            for row in rows:
                indexed[row["question_id"]][row["stage"]] = row
            for question_id in sorted(indexed):
                for stage in actionable:
                    row = indexed[question_id][stage]
                    prepared.append(row)
                    if row["predicted_stop_label"]:
                        break
            seq_metrics = metrics_from_decisions(
                [int(row["actual_stop_label"]) for row in prepared],
                [int(row["predicted_stop_label"]) for row in prepared],
                [float(row["analysis_probability"]) for row in prepared],
            )
            seq_hard = hard_actionable_metrics(prepared)
            seq_metrics["hard_partial_false_stop_rate"] = seq_hard["hard_partial_false_stop_rate"]
        else:
            all_metrics = hard_actionable_metrics(rows)
            seq_metrics, prepared = hard_sequential_metrics(rows, ladder)
        predictions = {(row["question_id"], row["stage"]): row for row in rows}
        recoverability = controller_recoverability_metrics(
            diagnostic_path, predictions, name, "dev_policy", "saved"
        )
        zero_cost = reached_cost(rows, benchmark)
        seq_cost = reached_cost(prepared, benchmark)
        output_rows.append(summary_row(name, role, "all_actionable", all_metrics, None, zero_cost, parse_rate, probability_available, threshold, terminal_failure_rate))
        output_rows.append(summary_row(name, role, "sequential_reached", seq_metrics, recoverability, seq_cost, parse_rate, probability_available, threshold, terminal_failure_rate))
        sequential_by_name[name] = output_rows[-1]

    judge_512 = sequential_by_name["Judge-512"]
    judge_full = sequential_by_name["Judge-Full"]
    full_gain = {
        "macro_f1": judge_full["macro_f1"] - judge_512["macro_f1"],
        "false_stop_rate": judge_full["false_stop_rate"] - judge_512["false_stop_rate"],
        "input_tokens_per_question": judge_full["mean_input_tokens_per_question"] - judge_512["mean_input_tokens_per_question"],
    }
    write_csv(summary_path, output_rows)
    write_report(report_path, output_rows, full_gain)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "analysis_split": "dev_policy",
            "heldout_consulted": False,
            "actionable_stages": actionable,
            "terminal_stage_excluded_from_fsr": ladder[-1],
            "judge_probability_available": False,
            "judge_auroc_reported": False,
            "judge_threshold_sweep_performed": False,
            "classification_threshold": classification_threshold,
            "risk_policy_threshold": risk_threshold,
            "prompt_gate": portable_path(gate_path, ROOT),
            "prompt_gate_sha256": file_sha256(gate_path),
            "sources": source_files,
            "critic_predictions": portable_path(critic_prediction_path, ROOT),
            "critic_predictions_sha256": file_sha256(critic_prediction_path),
            "summary": portable_path(summary_path, ROOT),
            "report": portable_path(report_path, ROOT),
            "git_commit": git_commit(ROOT),
        },
    )
    print(report_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

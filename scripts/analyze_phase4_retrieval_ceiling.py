"""Separate Phase 4 Controller errors from the frozen retrieval ladder ceiling."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

try:
    from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
except ModuleNotFoundError:  # Imported as scripts.* in tests.
    from scripts.experiment_utils import git_commit, portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"
ALLOWED_SPLITS = {"train_core", "dev_calibration", "dev_policy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--splits", default="train_core,dev_calibration,dev_policy"
    )
    parser.add_argument(
        "--prediction",
        nargs=3,
        action="append",
        default=[],
        metavar=("LABEL", "PATH", "THRESHOLD_OR_SAVED"),
        help="Optional Controller predictions. Use a numeric threshold or 'saved'.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def trajectories(path: Path, split: str, ladder: list[str]) -> Iterator[list[dict]]:
    current_id = None
    current = []
    seen = set()
    for row in read_jsonl(path):
        if row["split"] != split:
            raise ValueError(f"Unexpected split in {path}: {row['split']}")
        question_id = row["question_id"]
        if current_id is not None and question_id != current_id:
            if current_id in seen:
                raise ValueError(f"Non-contiguous duplicate question: {current_id}")
            seen.add(current_id)
            validate_trajectory(current, ladder)
            yield current
            current = []
        current_id = question_id
        current.append(row)
    if current:
        if current_id in seen:
            raise ValueError(f"Non-contiguous duplicate question: {current_id}")
        validate_trajectory(current, ladder)
        yield current


def validate_trajectory(rows: list[dict], ladder: list[str]) -> None:
    stages = [row["stage"] for row in rows]
    if stages != ladder:
        raise ValueError(
            f"Invalid trajectory for {rows[0]['question_id']}: {stages} != {ladder}"
        )
    coverage = [float(snapshot(row)["coverage"]) for row in rows]
    if any(right < left for left, right in zip(coverage, coverage[1:])):
        raise ValueError(f"Cumulative coverage decreased for {rows[0]['question_id']}")


def stage_latency_ms(row: dict) -> float:
    values = row["latency_ms"]
    dense = float(values["query_embedding_run_mean"]) + float(values["dense_search"])
    if row["stage"] == "dense@5":
        return dense
    hybrid = dense + float(values["bm25_scoring"]) + float(values["rrf_fusion"])
    if row["stage"] == "hybrid@10":
        return hybrid
    if row["stage"] == "rerank@20":
        return hybrid + float(values["reranker_allocated"])
    raise ValueError(f"Unknown retrieval stage: {row['stage']}")


def snapshot(row: dict) -> dict:
    view = row["cumulative_evidence_memory"]
    return {
        "sufficient": int(view["stop_label"]),
        "state": view["evidence_state"],
        "coverage": float(view["supporting_fact_recall"]),
        "chunks": len(view["items"]),
        "latency_ms": stage_latency_ms(row),
    }


def fact_bucket(count: int) -> str:
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    return "4+"


def empty_transition(source: str, target: str) -> dict:
    return {
        "source_stage": source,
        "target_stage": target,
        "eligible_current_incomplete": 0,
        "rescued": 0,
        "eligible_incremental_chunks": 0.0,
        "eligible_incremental_latency_ms": 0.0,
        "rescued_incremental_chunks": 0.0,
        "rescued_incremental_latency_ms": 0.0,
    }


def update_transition(stats: dict, source: dict, target: dict) -> None:
    if source["sufficient"]:
        return
    stats["eligible_current_incomplete"] += 1
    chunk_delta = target["chunks"] - source["chunks"]
    latency_delta = target["latency_ms"] - source["latency_ms"]
    if chunk_delta < 0 or latency_delta < -1e-9:
        raise ValueError("Frozen cumulative retrieval cost must not decrease")
    stats["eligible_incremental_chunks"] += chunk_delta
    stats["eligible_incremental_latency_ms"] += latency_delta
    if target["sufficient"]:
        stats["rescued"] += 1
        stats["rescued_incremental_chunks"] += chunk_delta
        stats["rescued_incremental_latency_ms"] += latency_delta


def finalized_transition(split: str, name: str, stats: dict, questions: int) -> dict:
    eligible = stats["eligible_current_incomplete"]
    rescued = stats["rescued"]
    return {
        "split": split,
        "analysis": name,
        "source_stage": stats["source_stage"],
        "target_stage": stats["target_stage"],
        "questions": questions,
        "eligible_current_incomplete": eligible,
        "rescued": rescued,
        "rescue_rate": rescued / eligible if eligible else None,
        "complete_coverage_gain": rescued / questions if questions else None,
        "eligible_extra_chunks_per_rescued_question": (
            stats["eligible_incremental_chunks"] / rescued if rescued else None
        ),
        "eligible_extra_latency_ms_per_rescued_question": (
            stats["eligible_incremental_latency_ms"] / rescued if rescued else None
        ),
        "rescued_only_mean_incremental_chunks": (
            stats["rescued_incremental_chunks"] / rescued if rescued else None
        ),
        "rescued_only_mean_incremental_latency_ms": (
            stats["rescued_incremental_latency_ms"] / rescued if rescued else None
        ),
    }


def analyze_split(
    source: Iterable[list[dict]],
    split: str,
    diagnostic_handle,
) -> tuple[dict, list[dict], list[dict]]:
    transitions = {
        "Dense_to_Hybrid_Rescue": empty_transition("dense@5", "hybrid@10"),
        "Dense_to_Final_Rescue": empty_transition("dense@5", "rerank@20"),
        "Hybrid_to_Final_Rescue": empty_transition("hybrid@10", "rerank@20"),
    }
    questions = 0
    sufficient_counts = [0, 0, 0]
    oracle_chunks = oracle_rerank = oracle_latency = 0.0
    bucket_counts = defaultdict(lambda: {"questions": 0, "sufficient": [0, 0, 0]})

    for rows in source:
        questions += 1
        states = [snapshot(row) for row in rows]
        for index, state in enumerate(states):
            sufficient_counts[index] += state["sufficient"]

        update_transition(transitions["Dense_to_Hybrid_Rescue"], states[0], states[1])
        update_transition(transitions["Dense_to_Final_Rescue"], states[0], states[2])
        update_transition(transitions["Hybrid_to_Final_Rescue"], states[1], states[2])

        selected_index = next(
            (index for index, state in enumerate(states) if state["sufficient"]), 2
        )
        selected = states[selected_index]
        oracle_chunks += selected["chunks"]
        oracle_rerank += int(selected_index == 2)
        oracle_latency += selected["latency_ms"]

        bucket = fact_bucket(int(rows[0]["gold_supporting_fact_count"]))
        bucket_counts[bucket]["questions"] += 1
        for index, state in enumerate(states):
            bucket_counts[bucket]["sufficient"][index] += state["sufficient"]

        for index, row in enumerate(rows[:-1]):
            current = states[index]
            diagnostic = {
                "question_id": row["question_id"],
                "split": split,
                "stage": row["stage"],
                "current_sufficient": current["sufficient"],
                "final_sufficient": states[2]["sufficient"],
                "later_recoverable": int(
                    not current["sufficient"]
                    and any(state["sufficient"] for state in states[index + 1 :])
                ),
                "current_evidence_state": current["state"],
                "current_supporting_fact_recall": current["coverage"],
                "final_supporting_fact_recall": states[2]["coverage"],
            }
            diagnostic_handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")

    if not questions:
        raise ValueError(f"No trajectories found for {split}")
    final_coverage = sufficient_counts[2] / questions
    ceiling = {
        "split": split,
        "questions": questions,
        "dense_complete_evidence_coverage": sufficient_counts[0] / questions,
        "hybrid_complete_evidence_coverage": sufficient_counts[1] / questions,
        "final_complete_evidence_coverage": final_coverage,
        "terminal_retrieval_failure_count": questions - sufficient_counts[2],
        "terminal_retrieval_failure_rate": 1.0 - final_coverage,
        "oracle_average_chunks": oracle_chunks / questions,
        "oracle_reranker_calls": oracle_rerank / questions,
        "oracle_average_retrieval_latency_ms": oracle_latency / questions,
        "oracle_unresolved_rate": 1.0 - final_coverage,
    }
    rescue = [
        finalized_transition(split, name, stats, questions)
        for name, stats in transitions.items()
    ]
    buckets = []
    if split == "dev_policy":
        for bucket in ["2", "3", "4+"]:
            values = bucket_counts[bucket]
            count = values["questions"]
            buckets.append(
                {
                    "split": split,
                    "gold_supporting_fact_bucket": bucket,
                    "questions": count,
                    "dense_complete_evidence_coverage": (
                        values["sufficient"][0] / count if count else None
                    ),
                    "hybrid_complete_evidence_coverage": (
                        values["sufficient"][1] / count if count else None
                    ),
                    "final_complete_evidence_coverage": (
                        values["sufficient"][2] / count if count else None
                    ),
                }
            )
    return ceiling, rescue, buckets


def load_predictions(path: Path) -> tuple[dict, str]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No predictions found: {path}")
    splits = {row["split"] for row in rows}
    if len(splits) != 1 or not splits <= ALLOWED_SPLITS:
        raise ValueError(f"Predictions must use one Train-derived split, got {splits}")
    indexed = {}
    for row in rows:
        key = (row["question_id"], row["stage"])
        if key in indexed:
            raise ValueError(f"Duplicate prediction: {key}")
        indexed[key] = row
    return indexed, next(iter(splits))


def controller_recoverability_metrics(
    diagnostic_path: Path,
    predictions: dict,
    label: str,
    split: str,
    threshold: str,
) -> dict:
    grouped = defaultdict(dict)
    for row in read_jsonl(diagnostic_path):
        grouped[row["question_id"]][row["stage"]] = row
    expected_keys = {
        (question_id, stage)
        for question_id, stages in grouped.items()
        for stage in stages
    }
    if not expected_keys <= set(predictions):
        missing = sorted(expected_keys - set(predictions))[:5]
        raise ValueError(f"Predictions are incomplete for {label}: {missing}")

    counts = defaultdict(int)
    numeric_threshold = None if threshold == "saved" else float(threshold)
    for question_id in sorted(grouped):
        for stage in ["dense@5", "hybrid@10"]:
            diagnostic = grouped[question_id][stage]
            prediction = predictions[(question_id, stage)]
            stop = (
                int(prediction["predicted_stop_label"])
                if numeric_threshold is None
                else int(float(prediction["stop_probability"]) >= numeric_threshold)
            )
            counts["reached_actionable_decisions"] += 1
            if not diagnostic["current_sufficient"]:
                counts["reached_continue_states"] += 1
                if diagnostic["later_recoverable"]:
                    counts["reached_recoverable_continue_states"] += 1
                    counts["recoverable_false_stops"] += stop
                else:
                    counts["reached_unrecoverable_continue_states"] += 1
                    counts["unrecoverable_early_stops"] += stop
                counts["false_stops"] += stop
            if stop:
                break

    def ratio(numerator: str, denominator: str):
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    return {
        "controller": label,
        "split": split,
        "threshold": threshold,
        "evaluation_scope": "reached_dense_and_hybrid_decisions_only",
        **dict(counts),
        "false_stop_rate": ratio("false_stops", "reached_continue_states"),
        "recoverable_false_stop_rate": ratio(
            "recoverable_false_stops", "reached_recoverable_continue_states"
        ),
        "unrecoverable_early_stop_rate": ratio(
            "unrecoverable_early_stops", "reached_unrecoverable_continue_states"
        ),
        "recoverable_share_of_false_stops": ratio(
            "recoverable_false_stops", "false_stops"
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def percent(value) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def number(value, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def render_report(
    ceilings: list[dict],
    rescue_rows: list[dict],
    buckets: list[dict],
    controller_rows: list[dict],
) -> str:
    lines = [
        "# 2Wiki Retrieval Ceiling Analysis",
        "",
        "This analysis uses only `train_core`, `dev_calibration`, and `dev_policy`. "
        "The official labeled Dev (`heldout`) is not read.",
        "",
        "## False Stop Scope Audit",
        "",
        "Formal sequential-policy FSR is computed only over reached `dense@5` and "
        "`hybrid@10` decisions. `rerank@20` is removed with `[:-1]` in "
        "`phase3b_metrics.operational_metrics`, policy selection/bootstrap, and policy "
        "evaluation. A terminal incomplete state is therefore a Terminal Retrieval "
        "Failure, not a Controller False Stop.",
        "",
        "Training-time `dev_metrics.json` is an all-stage classification diagnostic and "
        "must not be reported as sequential-policy FSR. No Phase 3B result requires "
        "recomputation.",
        "",
        "## Retrieval Ceiling And Oracle",
        "",
        "| Split | Questions | Dense coverage | Hybrid coverage | Final coverage | TRFR | Oracle chunks | Oracle reranker calls | Oracle latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ceilings:
        lines.append(
            f"| {row['split']} | {row['questions']} | "
            f"{percent(row['dense_complete_evidence_coverage'])} | "
            f"{percent(row['hybrid_complete_evidence_coverage'])} | "
            f"{percent(row['final_complete_evidence_coverage'])} | "
            f"{percent(row['terminal_retrieval_failure_rate'])} | "
            f"{number(row['oracle_average_chunks'])} | "
            f"{number(row['oracle_reranker_calls'], 3)} | "
            f"{number(row['oracle_average_retrieval_latency_ms'])} |"
        )
    lines.extend(
        [
            "",
            "TRFR is checked as exactly `1 - Final Complete Evidence Coverage`. The "
            "Oracle stops at the first sufficient cumulative state and otherwise reaches "
            "the forced final stage; it is an analysis bound, not a deployment method.",
            "",
            "## Stage Rescue",
            "",
            "| Split | Rescue | Eligible incomplete | Rescued | Rescue rate | Coverage gain | Extra chunks / rescue | Extra latency ms / rescue |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rescue_rows:
        lines.append(
            f"| {row['split']} | {row['analysis']} | "
            f"{row['eligible_current_incomplete']} | {row['rescued']} | "
            f"{percent(row['rescue_rate'])} | {percent(row['complete_coverage_gain'])} | "
            f"{number(row['eligible_extra_chunks_per_rescued_question'])} | "
            f"{number(row['eligible_extra_latency_ms_per_rescued_question'])} |"
        )
    lines.extend(
        [
            "",
            "Extra cost per rescue divides the total incremental cost paid by all "
            "eligible incomplete questions by the number actually rescued.",
            "",
            "## Dev Policy By Gold Supporting-Fact Count",
            "",
            "| Gold facts | Questions | Dense coverage | Hybrid coverage | Final coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in buckets:
        lines.append(
            f"| {row['gold_supporting_fact_bucket']} | {row['questions']} | "
            f"{percent(row['dense_complete_evidence_coverage'])} | "
            f"{percent(row['hybrid_complete_evidence_coverage'])} | "
            f"{percent(row['final_complete_evidence_coverage'])} |"
        )
    lines.extend(["", "Small 3-fact or 4+-fact groups are descriptive only."])

    lines.extend(["", "## Controller Recoverability Diagnostics", ""])
    if controller_rows:
        lines.extend(
            [
                "| Controller | Split | FSR | RFSR | Unrecoverable early-stop rate | Recoverable share of false stops |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in controller_rows:
            lines.append(
                f"| {row['controller']} | {row['split']} | "
                f"{percent(row['false_stop_rate'])} | "
                f"{percent(row['recoverable_false_stop_rate'])} | "
                f"{percent(row['unrecoverable_early_stop_rate'])} | "
                f"{percent(row['recoverable_share_of_false_stops'])} |"
            )
    else:
        lines.append(
            "RFSR is pending frozen Train-derived Controller predictions. No placeholder "
            "value is inferred while multi-seed training is running. Re-run this analysis "
            "with `--prediction LABEL PATH THRESHOLD_OR_SAVED` after dev-policy inference."
        )

    dev = next(row for row in ceilings if row["split"] == "dev_policy")
    dev_rescue = {row["analysis"]: row for row in rescue_rows if row["split"] == "dev_policy"}
    dense = dev_rescue["Dense_to_Final_Rescue"]
    hybrid = dev_rescue["Hybrid_to_Final_Rescue"]
    dense_unrecoverable = dense["eligible_current_incomplete"] - dense["rescued"]
    hybrid_unrecoverable = hybrid["eligible_current_incomplete"] - hybrid["rescued"]
    lines.extend(
        [
            "",
            "## Answers To The Audit Questions",
            "",
            f"1. Dev-policy final retrieval coverage is {percent(dev['final_complete_evidence_coverage'])}; the retrieval ceiling failure rate is {percent(dev['terminal_retrieval_failure_rate'])}.",
            f"2. Of Dense Continue questions, {dense['rescued']}/{dense['eligible_current_incomplete']} ({percent(dense['rescue_rate'])}) become sufficient by a later stage.",
            f"3. Of Hybrid Continue questions, {hybrid['rescued']}/{hybrid['eligible_current_incomplete']} ({percent(hybrid['rescue_rate'])}) are rescued by Rerank.",
            (
                "4. Recoverable false-stop attribution is reported above from supplied "
                "Controller predictions."
                if controller_rows
                else "4. Recoverable false-stop attribution is pending completed Controller predictions; it is not fabricated from retrieval labels."
            ),
            f"5. On dev_policy, {dense_unrecoverable}/{dense['eligible_current_incomplete']} Dense Continue states and {hybrid_unrecoverable}/{hybrid['eligible_current_incomplete']} Hybrid Continue states are unrecoverable within the frozen ladder; these correspond to {dev['terminal_retrieval_failure_count']}/{dev['questions']} terminal-incomplete questions.",
            "6. Final forced stops are excluded from formal sequential FSR; terminal incompleteness is reported as TRFR.",
            "7. This analysis does not change the frozen Phase 4 protocol, model, retrieval parameters, labels, or thresholds, and it does not consult heldout.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not splits or set(splits) - ALLOWED_SPLITS:
        raise ValueError(f"Analysis is restricted to Train-derived splits: {splits}")
    if "dev_policy" not in splits:
        raise ValueError("dev_policy is required for the preregistered descriptive buckets")

    ladder = config["controller_ladder"]
    if ladder != ["dense@5", "hybrid@10", "rerank@20"]:
        raise ValueError(f"Unexpected frozen ladder: {ladder}")
    data_dir = ROOT / config["outputs"]["data_dir"] / "controller" / "sentence_256"
    output_dir = ROOT / "results" / "phase4" / "2wiki"
    diagnostics_dir = output_dir / "recoverability"
    report_path = ROOT / "docs" / "phase4" / "2wiki_retrieval_ceiling_analysis.md"
    rescue_path = output_dir / "stage_rescue_analysis.csv"
    ceiling_path = output_dir / "retrieval_ceiling_summary.csv"
    bucket_path = output_dir / "supporting_fact_retrievability.csv"
    controller_path = output_dir / "controller_recoverability_metrics.csv"
    manifest_path = output_dir / "retrieval_ceiling_manifest.json"
    diagnostic_paths = {
        split: diagnostics_dir / f"{split}.jsonl" for split in splits
    }
    targets = [
        report_path,
        rescue_path,
        ceiling_path,
        bucket_path,
        manifest_path,
        *diagnostic_paths.values(),
    ]
    if args.prediction:
        targets.append(controller_path)
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite retrieval-ceiling outputs: {existing}")

    ceilings = []
    rescue_rows = []
    buckets = []
    source_paths = {}
    for split in splits:
        source_path = data_dir / f"{split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_paths[split] = portable_path(source_path, ROOT)
        diagnostic_path = diagnostic_paths[split]
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = diagnostic_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            ceiling, rescue, split_buckets = analyze_split(
                trajectories(source_path, split, ladder), split, handle
            )
        temporary.replace(diagnostic_path)
        ceilings.append(ceiling)
        rescue_rows.extend(rescue)
        buckets.extend(split_buckets)

    for row in ceilings:
        if abs(
            row["terminal_retrieval_failure_rate"]
            - (1.0 - row["final_complete_evidence_coverage"])
        ) > 1e-12:
            raise AssertionError("TRFR complement check failed")

    controller_rows = []
    prediction_sources = []
    for label, raw_path, threshold in args.prediction:
        prediction_path = Path(raw_path)
        if not prediction_path.is_absolute():
            prediction_path = ROOT / prediction_path
        predictions, split = load_predictions(prediction_path)
        if split not in diagnostic_paths:
            raise ValueError(f"Prediction split was not analyzed: {split}")
        controller_rows.append(
            controller_recoverability_metrics(
                diagnostic_paths[split], predictions, label, split, threshold
            )
        )
        prediction_sources.append(
            {
                "label": label,
                "path": portable_path(prediction_path, ROOT),
                "split": split,
                "threshold": threshold,
            }
        )

    write_csv(rescue_path, rescue_rows)
    write_csv(ceiling_path, ceilings)
    write_csv(bucket_path, buckets)
    if controller_rows:
        write_csv(controller_path, controller_rows)
    elif controller_path.exists() and args.force:
        controller_path.unlink()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(ceilings, rescue_rows, buckets, controller_rows), encoding="utf-8"
    )
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "git_commit": git_commit(ROOT),
            "analysis_scope": splits,
            "heldout_consulted": False,
            "ladder": ladder,
            "actionable_stages": ladder[:-1],
            "terminal_stage": ladder[-1],
            "terminal_incomplete_classification": "terminal_retrieval_failure",
            "recoverability_is_evaluation_only": True,
            "recoverability_used_as_training_feature": False,
            "source_controller_data": source_paths,
            "prediction_sources": prediction_sources,
            "outputs": {
                "report": portable_path(report_path, ROOT),
                "stage_rescue": portable_path(rescue_path, ROOT),
                "retrieval_ceiling": portable_path(ceiling_path, ROOT),
                "supporting_fact_retrievability": portable_path(bucket_path, ROOT),
                "recoverability_diagnostics": {
                    split: portable_path(path, ROOT)
                    for split, path in diagnostic_paths.items()
                },
                "controller_recoverability_metrics": (
                    portable_path(controller_path, ROOT) if controller_rows else None
                ),
            },
        },
    )
    print(report_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

"""Build binary Controller examples with both raw and cumulative evidence definitions."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from evidence_utils import supporting_fact_metrics, unique_document_titles
from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from phase2_retrieval import cumulative_unique, parse_stage, score_statistics


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RETRIEVAL_CONFIG = ROOT / "configs" / "phase2" / "chunk_retrieval.json"
DEFAULT_CONTROLLER_CONFIG = ROOT / "configs" / "phase2" / "controller.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER_CONFIG)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def row_score(row: dict, method: str) -> float:
    return float(row["rrf_score"] if method == "hybrid" else row["score"])


def retrieval_statistics(rankings: dict, stage_index: int, stage_name: str) -> dict:
    current_method, current_k = parse_stage(stage_name)
    available_methods = ["dense"]
    if stage_index >= 1:
        available_methods.extend(["bm25", "hybrid"])
    if stage_index >= 2:
        available_methods.append("rerank")
    stats = {}
    for method in available_methods:
        rows = rankings[method][:current_k]
        stats[method] = score_statistics([row_score(row, method) for row in rows])
    stats["current_method"] = current_method
    stats["current_k"] = current_k
    return stats


def evidence_items(
    chunk_ids: list[str], rankings: dict, methods_seen: list[str], chunk_by_id: dict
) -> list[dict]:
    latest_rows = {}
    for method in methods_seen:
        for row in rankings[method]:
            latest_rows[row["chunk_id"]] = (method, row)
    items = []
    for chunk_id in chunk_ids:
        source = latest_rows.get(chunk_id)
        method, row = source if source else (None, None)
        chunk = chunk_by_id[chunk_id]
        items.append(
            {
                "chunk_id": chunk_id,
                "document_title": chunk["document_title"],
                "sentence_ids": chunk["sentence_ids"],
                "retrieval_score": row_score(row, method) if row else None,
                "retrieval_rank": int(row["rank"]) if row else None,
                "score_source": method,
            }
        )
    return items


def evidence_view(chunk_ids: list[str], chunk_by_id: dict, gold: list[dict]) -> dict:
    metrics = supporting_fact_metrics(chunk_ids, chunk_by_id, gold)
    return {
        "chunk_ids": chunk_ids,
        "document_titles": unique_document_titles(chunk_ids, chunk_by_id),
        **metrics,
        "stop_label": metrics["complete_evidence_coverage"],
        "action_label": "stop" if metrics["complete_evidence_coverage"] else "continue",
    }


def render_trajectory_report(distribution: dict) -> str:
    lines = [
        "# Phase 2 Evidence Trajectory Audit",
        "",
        "The Controller ladder is evaluated under both raw-stage evidence and cumulative evidence memory. Cumulative coverage is asserted to be non-decreasing during data construction.",
    ]
    for split, details in distribution["by_split"].items():
        lines.extend(
            [
                "",
                f"## {split}",
                "",
                "| Evidence definition | Questions | Non-monotonic | Ratio |",
                "|---|---:|---:|---:|",
            ]
        )
        for mode in ["raw", "cumulative"]:
            values = details["trajectory_monotonicity"][mode]
            lines.append(
                f"| {mode} | {values['questions']} | {values['non_monotonic']} | {values['ratio']:.6f} |"
            )
        lines.extend(
            [
                "",
                "| Mode | Stage | Insufficient | Partial | Sufficient | Continue | Stop |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ["raw", "cumulative"]:
            for stage, counts in details["labels"][mode].items():
                lines.append(
                    f"| {mode} | {stage} | {counts.get('insufficient', 0)} | "
                    f"{counts.get('partial', 0)} | {counts.get('sufficient', 0)} | "
                    f"{counts.get('continue', 0)} | {counts.get('stop', 0)} |"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    retrieval_config = json.loads(args.retrieval_config.read_text(encoding="utf-8"))
    controller_config = json.loads(args.controller_config.read_text(encoding="utf-8"))
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if set(splits) - {"train", "dev", "test"} or not splits:
        raise ValueError(f"Invalid split list: {splits}")
    known_variants = {item["name"] for item in retrieval_config["chunking"]["variants"]}
    if args.variant not in known_variants:
        raise ValueError(f"Unknown variant: {args.variant}")
    if "test" in splits:
        selected = retrieval_config["retrieval_evaluation"]["selected_variant_after_dev"]
        if selected != args.variant:
            raise ValueError("Test Controller data is locked to the dev-selected chunk variant")

    chunks = read_jsonl(
        ROOT / "data" / "phase2" / "chunks" / f"{args.variant}.jsonl"
    )
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    questions = json.loads(
        (ROOT / "data" / "phase2" / "questions.json").read_text(encoding="utf-8")
    )
    ladder = controller_config["controller_ladder"]
    parsed_ladder = [parse_stage(stage) for stage in ladder]
    output_dir = ROOT / "data" / "phase2" / "controller" / args.variant
    output_paths = {split: output_dir / f"{split}.jsonl" for split in splits}
    distribution_path = output_dir / f"label_distribution_{'_'.join(splits)}.json"
    report_path = ROOT / "docs" / "phase2_trajectory_audit.md"
    manifest_path = output_dir / f"dataset_manifest_{'_'.join(splits)}.json"
    targets = [*output_paths.values(), distribution_path, report_path, manifest_path]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Controller data outputs: {existing}")

    labels = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    monotonicity = defaultdict(
        lambda: {
            "raw": {"questions": 0, "non_monotonic": 0},
            "cumulative": {"questions": 0, "non_monotonic": 0},
        }
    )
    excluded_invalid = Counter()
    ranking_sources = {}

    for split in splits:
        ranking_path = (
            ROOT / "results" / "phase2" / "retrieval" / args.variant / f"{split}.jsonl"
        )
        if not ranking_path.exists():
            raise FileNotFoundError(f"Run Phase 2 retrieval first: {ranking_path}")
        ranking_sources[split] = portable_path(ranking_path, ROOT)
        ranking_records = read_jsonl(ranking_path)
        output_records = []
        for retrieval_record in ranking_records:
            qid = retrieval_record["question_id"]
            question = questions[qid]
            if not question.get("gold_annotation_valid", True):
                if split != "train":
                    raise ValueError(f"Invalid gold annotation outside train: {qid}")
                excluded_invalid[split] += 1
                continue
            rankings = retrieval_record["rankings"]
            raw_rankings = [
                [row["chunk_id"] for row in rankings[method][:k]]
                for method, k in parsed_ladder
            ]
            cumulative_rankings = cumulative_unique(raw_rankings)
            raw_coverage = []
            cumulative_coverage = []
            methods_seen = []
            for stage_index, ((method, k), stage_name, raw_ids, cumulative_ids) in enumerate(
                zip(parsed_ladder, ladder, raw_rankings, cumulative_rankings)
            ):
                if method == "hybrid":
                    methods_seen.extend(["bm25", "hybrid"])
                else:
                    methods_seen.append(method)
                raw_view = evidence_view(raw_ids, chunk_by_id, question["gold_supporting_facts"])
                cumulative_view = evidence_view(
                    cumulative_ids, chunk_by_id, question["gold_supporting_facts"]
                )
                raw_view["items"] = evidence_items(
                    raw_ids, rankings, methods_seen, chunk_by_id
                )
                cumulative_view["items"] = evidence_items(
                    cumulative_ids, rankings, methods_seen, chunk_by_id
                )
                raw_coverage.append(raw_view["supporting_fact_recall"])
                cumulative_coverage.append(cumulative_view["supporting_fact_recall"])
                for mode, view in [("raw", raw_view), ("cumulative", cumulative_view)]:
                    labels[split][mode][stage_name][view["evidence_state"]] += 1
                    labels[split][mode][stage_name][view["action_label"]] += 1
                output_records.append(
                    {
                        "question_id": qid,
                        "split": split,
                        "question": question["question"],
                        "answer": question["answer"],
                        "question_type": question["type"],
                        "gold_supporting_facts": question["gold_supporting_facts"],
                        "gold_supporting_fact_count": len(
                            {
                                (fact["title"], fact["sentence_id"])
                                for fact in question["gold_supporting_facts"]
                            }
                        ),
                        "chunk_variant": args.variant,
                        "stage_index": stage_index,
                        "stage": stage_name,
                        "retrieval_method": method,
                        "k": k,
                        "retrieval_statistics": retrieval_statistics(
                            rankings, stage_index, stage_name
                        ),
                        "raw_stage_evidence": raw_view,
                        "cumulative_evidence_memory": cumulative_view,
                        "latency_ms": retrieval_record["latency_ms"],
                    }
                )

            for mode, trajectory in [
                ("raw", raw_coverage),
                ("cumulative", cumulative_coverage),
            ]:
                non_monotonic = any(
                    right < left for left, right in zip(trajectory, trajectory[1:])
                )
                monotonicity[split][mode]["questions"] += 1
                monotonicity[split][mode]["non_monotonic"] += int(non_monotonic)
            if any(
                right < left
                for left, right in zip(cumulative_coverage, cumulative_coverage[1:])
            ):
                raise AssertionError(f"Cumulative evidence coverage decreased for {qid}")

        write_jsonl_atomic(output_paths[split], output_records)
        logger.info("saved %s samples -> %s", len(output_records), output_paths[split])

    by_split = {}
    for split in splits:
        split_monotonicity = {}
        for mode, values in monotonicity[split].items():
            split_monotonicity[mode] = {
                **values,
                "ratio": values["non_monotonic"] / values["questions"],
            }
        by_split[split] = {
            "labels": {
                mode: {
                    stage: dict(labels[split][mode][stage]) for stage in ladder
                }
                for mode in ["raw", "cumulative"]
            },
            "trajectory_monotonicity": split_monotonicity,
            "invalid_gold_questions_excluded": excluded_invalid[split],
        }
    distribution = {
        "schema_version": 1,
        "variant": args.variant,
        "ladder": ladder,
        "primary_labels": {"continue": 0, "stop": 1},
        "diagnostic_labels": {"insufficient": 0, "partial": 1, "sufficient": 2},
        "by_split": by_split,
    }
    write_json_atomic(distribution_path, distribution)
    report_path.write_text(render_trajectory_report(distribution), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": 2,
        "phase1_frozen_commit": retrieval_config["phase1_frozen_commit"],
        "git_commit": git_commit(ROOT),
        "retrieval_config": portable_path(args.retrieval_config, ROOT),
        "controller_config": portable_path(args.controller_config, ROOT),
        "variant": args.variant,
        "splits": splits,
        "ranking_sources": ranking_sources,
        "outputs": {split: portable_path(path, ROOT) for split, path in output_paths.items()},
        "invalid_gold_questions_excluded": dict(excluded_invalid),
    }
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()

"""Build sentence-level three-class sufficiency trajectories from existing rankings."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from evidence_utils import (
    assert_disjoint_question_ids,
    supporting_fact_metrics,
    unique_document_titles,
    validate_tuning_splits,
)
from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "retrieval" / "unified_hotpotqa.json"
DEFAULT_OUTPUT = ROOT / "data" / "sufficiency"
DEFAULT_REPORT = ROOT / "docs" / "sufficiency_label_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--jsonl-only",
        action="store_true",
        help="Build only ignored train/dev/test JSONL files without replacing tracked reports",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_questions(all_questions, legacy_test_questions):
    test = all_questions[:1000]
    remaining = all_questions[1000:]
    split = {"train": remaining[:5405], "dev": remaining[5405:6405], "test": test}
    expected_test = {question["qid"] for question in legacy_test_questions}
    actual_test = {question["qid"] for question in split["test"]}
    if actual_test != expected_test:
        raise ValueError("The all-question ordering no longer reproduces the frozen legacy test split")
    assert_disjoint_question_ids(
        {name: [question["qid"] for question in questions] for name, questions in split.items()}
    )
    validate_tuning_splits(["train", "dev"])
    return split


def output_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / "train.jsonl",
        output_dir / "dev.jsonl",
        output_dir / "test.jsonl",
        output_dir / "label_distribution.json",
        output_dir / "dataset_manifest.json",
    ]


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    dataset_config = config["dataset"]
    all_data = load_json(ROOT / dataset_config["all_questions_data"])
    legacy_test = load_json(ROOT / dataset_config["legacy_test_data"])
    metadata = load_json(ROOT / dataset_config["metadata"])
    chunk_metadata = metadata["chunks"]
    question_metadata = metadata["questions"]
    retrieval = load_json(ROOT / "results" / "hotpotqa_all_retrieval.json")
    split = split_questions(all_data["questions"], legacy_test["questions"])

    jsonl_targets = [args.output_dir / f"{split_name}.jsonl" for split_name in split]
    targets = jsonl_targets if args.jsonl_only else output_paths(args.output_dir) + [args.report]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite outputs: {[str(path) for path in existing]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    stage_specs = config["baselines"]
    counts = defaultdict(Counter)
    overall = Counter()
    trajectories = {name: Counter() for name in split}
    non_monotonic = Counter()
    total_samples = 0

    for split_name, questions in split.items():
        target = args.output_dir / f"{split_name}.jsonl"
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for question in questions:
                qid = question["qid"]
                if qid not in question_metadata:
                    raise KeyError(f"Missing supporting-fact metadata for {qid}")
                gold = question_metadata[qid]["gold_supporting_facts"]
                trajectory = []
                for stage_spec in stage_specs:
                    stage_name = stage_spec["name"]
                    stage = stage_spec["stage"]
                    k = int(stage_spec["k"])
                    ranked = retrieval[stage][qid][:k]
                    metrics = supporting_fact_metrics(ranked, chunk_metadata, gold)
                    titles = unique_document_titles(ranked, chunk_metadata)
                    evidence_text = " [DOC] ".join(all_data["kb"][chunk_id] for chunk_id in ranked)
                    record = {
                        "question_id": qid,
                        "qid": qid,
                        "question": question["question"],
                        "answer": question["answer"],
                        "stage": stage_name,
                        "retrieval_stage": stage,
                        "k": k,
                        "evidence_text": evidence_text,
                        "retrieved_chunk_ids": ranked,
                        "retrieved_document_titles": titles,
                        "retrieved_scores": [],
                        "retrieved_scores_status": "unavailable_in_legacy_ranking",
                        "gold_supporting_facts": gold,
                        **metrics,
                        "gold_supporting_fact_ids": gold,
                        "covered_supporting_fact_ids": metrics["covered_supporting_facts"],
                        "label_name": metrics["evidence_state"],
                        "critic_score": None,
                        "critic_prediction": None,
                        "retrieval_action": (
                            "stop" if metrics["evidence_state"] == "sufficient" else "escalate"
                        ),
                        "final_answer": None,
                        "latency": None,
                        "token_count": None,
                        "reranker_calls": int(stage == "rerank"),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    state = metrics["evidence_state"]
                    counts[(split_name, stage_name)][state] += 1
                    overall[(split_name, state)] += 1
                    trajectory.append(state)
                    total_samples += 1
                trajectory_key = " -> ".join(trajectory)
                trajectories[split_name][trajectory_key] += 1
                numeric = [0 if state == "insufficient" else 1 if state == "partial" else 2 for state in trajectory]
                if any(right < left for left, right in zip(numeric, numeric[1:])):
                    non_monotonic[split_name] += 1
        temporary.replace(target)
        logger.info("saved %s", target)

    if args.jsonl_only:
        logger.info("JSONL-only build complete; tracked reports and manifests were not changed")
        return

    gold_count_distribution = Counter()
    for item in question_metadata.values():
        unique_gold = {
            (fact["title"], int(fact["sentence_id"])) for fact in item["gold_supporting_facts"]
        }
        gold_count_distribution[len(unique_gold)] += 1

    distribution = {
        "schema_version": 1,
        "label_order": ["insufficient", "partial", "sufficient"],
        "stage_order": [item["name"] for item in stage_specs],
        "gold_supporting_fact_count": dict(sorted(gold_count_distribution.items())),
        "by_split_stage": {
            split_name: {
                stage["name"]: dict(counts[(split_name, stage["name"])]) for stage in stage_specs
            }
            for split_name in split
        },
        "overall_by_split": {
            split_name: {
                state: overall[(split_name, state)]
                for state in ["insufficient", "partial", "sufficient"]
            }
            for split_name in split
        },
        "trajectories": {name: dict(counter.most_common()) for name, counter in trajectories.items()},
        "non_monotonic_trajectories": dict(non_monotonic),
    }
    write_json_atomic(args.output_dir / "label_distribution.json", distribution)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "config": portable_path(args.config, ROOT),
        "git_commit": git_commit(ROOT),
        "seed": config["seed"],
        "split_question_counts": {name: len(questions) for name, questions in split.items()},
        "total_samples": total_samples,
        "metadata_source": dataset_config["metadata"],
        "retrieval_source": "results/hotpotqa_all_retrieval.json",
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(args.output_dir / "dataset_manifest.json", manifest)

    lines = [
        "# Sufficiency 标签分布",
        "",
        "标签按严格 supporting fact `[title, sentence_id]` coverage 构造：0=Insufficient，1=Partial，2=Sufficient。",
        "",
        "## Supporting Fact 数量",
        "",
        "| Gold facts / question | Questions |",
        "|---:|---:|",
    ]
    for count, number in sorted(gold_count_distribution.items()):
        lines.append(f"| {count} | {number} |")
    for split_name in ["train", "dev", "test"]:
        lines.extend(
            [
                "",
                f"## {split_name}",
                "",
                "| Stage | Insufficient | Partial | Sufficient |",
                "|---|---:|---:|---:|",
            ]
        )
        for stage in stage_specs:
            current = counts[(split_name, stage["name"])]
            lines.append(
                f"| {stage['name']} | {current['insufficient']} | {current['partial']} | "
                f"{current['sufficient']} |"
            )
        lines.extend(
            [
                "",
                f"Non-monotonic trajectories: {non_monotonic[split_name]} / {len(split[split_name])}",
                "",
                "Most common trajectories:",
                "",
            ]
        )
        for trajectory, count in trajectories[split_name].most_common(10):
            lines.append(f"- `{trajectory}`: {count}")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("saved %s", args.report)


if __name__ == "__main__":
    main()

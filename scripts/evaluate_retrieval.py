"""Evaluate fixed HotpotQA retrieval baselines with sentence-level evidence metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from evidence_utils import supporting_fact_metrics, unique_document_titles
from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "retrieval" / "unified_hotpotqa.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    dataset_config = config["dataset"]
    test_data = load_json(ROOT / dataset_config["legacy_test_data"])
    metadata = load_json(ROOT / dataset_config["metadata"])
    chunk_metadata = metadata["chunks"]
    question_metadata = metadata["questions"]
    question_by_id = {question["qid"]: question for question in test_data["questions"]}
    output = config["output"]
    summary_path = ROOT / output["summary"]
    per_sample_dir = ROOT / output["per_sample_dir"]
    manifest_path = summary_path.parent / "run_manifest.json"
    targets = [summary_path, manifest_path] + [
        per_sample_dir / f"{baseline['name'].replace('@', '_at_')}.jsonl"
        for baseline in config["baselines"]
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite outputs: {[str(path) for path in existing]}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for baseline in config["baselines"]:
        result_data = load_json(ROOT / baseline["result"])
        result_by_id = {item["qid"]: item for item in result_data["results"]}
        missing = sorted(set(question_by_id) - set(result_by_id))
        if missing:
            raise ValueError(f"{baseline['name']} is missing {len(missing)} test questions")
        stage = baseline["stage"]
        k = int(baseline["k"])
        latency_ms = float(config["legacy_latency_ms"][stage])
        reranker_calls = int(stage == "rerank")
        sample_path = per_sample_dir / f"{baseline['name'].replace('@', '_at_')}.jsonl"
        temporary = sample_path.with_suffix(sample_path.suffix + ".tmp")
        total_recall = 0.0
        total_complete = 0
        total_chunks = 0
        total_unique_documents = 0
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for qid, question in question_by_id.items():
                ranked = result_by_id[qid]["ranked"][:k]
                gold = question_metadata[qid]["gold_supporting_facts"]
                metrics = supporting_fact_metrics(ranked, chunk_metadata, gold)
                titles = unique_document_titles(ranked, chunk_metadata)
                record = {
                    "question_id": qid,
                    "question": question["question"],
                    "answer": question["answer"],
                    "retrieved_chunk_ids": ranked,
                    "retrieved_document_titles": titles,
                    "retrieved_scores": [],
                    "retrieved_scores_status": "unavailable_in_legacy_ranking",
                    "retrieval_stage": baseline["name"],
                    "covered_supporting_fact_ids": metrics["covered_supporting_facts"],
                    "gold_supporting_fact_ids": gold,
                    "supporting_fact_recall": metrics["supporting_fact_recall"],
                    "complete_evidence_coverage": metrics["complete_evidence_coverage"],
                    "evidence_state": metrics["evidence_state"],
                    "critic_score": None,
                    "critic_prediction": None,
                    "retrieval_action": "fixed_stop",
                    "final_answer": None,
                    "latency": {
                        "milliseconds": latency_ms,
                        "source": "legacy_stage_mean_estimate",
                    },
                    "token_count": None,
                    "reranker_calls": reranker_calls,
                    "reranker_scored_chunks": k if stage == "rerank" else 0,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_recall += metrics["supporting_fact_recall"]
                total_complete += metrics["complete_evidence_coverage"]
                total_chunks += len(ranked)
                total_unique_documents += len(titles)
        temporary.replace(sample_path)
        count = len(question_by_id)
        summaries.append(
            {
                "baseline": baseline["name"],
                "questions": count,
                "supporting_fact_recall": total_recall / count,
                "complete_evidence_coverage": total_complete / count,
                "average_retrieved_chunks": total_chunks / count,
                "average_unique_documents": total_unique_documents / count,
                "average_latency_ms": latency_ms,
                "latency_source": "legacy_stage_mean_estimate",
                "average_reranker_calls": reranker_calls,
                "answer_em": "not_evaluated",
                "answer_f1": "not_evaluated",
            }
        )
        logger.info("evaluated %s -> %s", baseline["name"], sample_path)

    fieldnames = list(summaries[0])
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    temporary_summary.replace(summary_path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "config": portable_path(args.config, ROOT),
        "git_commit": git_commit(ROOT),
        "seed": config["seed"],
        "latency_note": config["legacy_latency_ms"]["source"],
        "environment": collect_environment(ROOT),
        "summaries": summaries,
    }
    write_json_atomic(manifest_path, manifest)
    logger.info("saved %s", summary_path)


if __name__ == "__main__":
    main()

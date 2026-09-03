"""Evaluate Phase 2 chunk retrieval with strict title-and-sentence evidence metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from evidence_utils import supporting_fact_metrics, unique_document_titles, unique_supporting_facts
from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase2" / "chunk_retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", help="Comma-separated variants; defaults to config selection set")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
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


def ranking_score(row: dict, method: str) -> float:
    if method == "hybrid":
        return float(row["rrf_score"])
    return float(row["score"])


def method_latency_ms(record: dict, method: str) -> float:
    latency = record["latency_ms"]
    if method == "dense":
        return latency["query_embedding_run_mean"] + latency["dense_search"]
    if method == "bm25":
        return latency["bm25_scoring"]
    hybrid = (
        latency["query_embedding_run_mean"]
        + latency["dense_search"]
        + latency["bm25_scoring"]
        + latency["rrf_fusion"]
    )
    if method == "hybrid":
        return hybrid
    if method == "rerank":
        return hybrid + latency["reranker_allocated"]
    raise ValueError(f"Unknown retrieval method: {method}")


def render_markdown(summaries: list[dict], split: str) -> str:
    lines = [
        "# Phase 2 Chunk Retrieval Comparison",
        "",
        f"Evaluation split: `{split}`. Evidence matches require exact `(document_title, sentence_id)` equality.",
        "",
        "| Variant | Method | K | SF Recall | Complete coverage | Any coverage | Gold-title recall | Avg docs | Avg latency ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['variant']} | {item['method']} | {item['k']} | "
            f"{item['supporting_fact_recall']:.4f} | {item['complete_evidence_coverage']:.4f} | "
            f"{item['any_evidence_coverage']:.4f} | {item['gold_title_recall']:.4f} | "
            f"{item['average_unique_documents']:.2f} | {item['average_latency_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Latency is measured during this retrieval run. Batched query-embedding and reranker time is allocated equally within the corresponding batch; it is not a single-query online benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluation = config["retrieval_evaluation"]
    variants = (
        [value.strip() for value in args.variants.split(",") if value.strip()]
        if args.variants
        else evaluation["selection_variants"]
    )
    known = {item["name"] for item in config["chunking"]["variants"]}
    if not variants or set(variants) - known:
        raise ValueError(f"Unknown or empty variant list: {variants}")
    if args.split == "test":
        selected = evaluation["selected_variant_after_dev"]
        if selected is None or variants != [selected]:
            raise ValueError("Test evaluation requires exactly the dev-selected chunk variant")

    questions = json.loads(
        (ROOT / "data" / "phase2" / "questions.json").read_text(encoding="utf-8")
    )
    question_by_id = {
        qid: question
        for qid, question in questions.items()
        if question["split"] == args.split and question.get("gold_annotation_valid", True)
    }
    methods = evaluation["methods"]
    ks = [int(k) for k in evaluation["ks"]]
    output_dir = ROOT / "results" / "phase2" / "retrieval_eval" / args.split
    summary_csv = output_dir / "retrieval_summary.csv"
    summary_md = output_dir / "retrieval_summary.md"
    manifest_path = output_dir / "evaluation_manifest.json"
    per_sample_paths = {
        (variant, method, k): output_dir
        / "per_sample"
        / f"{variant}__{method}_at_{k}.jsonl"
        for variant in variants
        for method in methods
        for k in ks
    }
    targets = [summary_csv, summary_md, manifest_path, *per_sample_paths.values()]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite evaluation outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    ranking_sources = {}
    for variant in variants:
        chunk_path = ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl"
        ranking_path = ROOT / "results" / "phase2" / "retrieval" / variant / f"{args.split}.jsonl"
        if not chunk_path.exists() or not ranking_path.exists():
            raise FileNotFoundError(f"Missing chunks or rankings for {variant} {args.split}")
        chunks = read_jsonl(chunk_path)
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        rankings = read_jsonl(ranking_path)
        ranking_by_id = {record["question_id"]: record for record in rankings}
        missing_questions = set(question_by_id) - set(ranking_by_id)
        if missing_questions:
            raise ValueError(f"{variant} rankings miss {len(missing_questions)} questions")
        ranking_sources[variant] = portable_path(ranking_path, ROOT)

        index_manifest_path = (
            ROOT / "data" / "phase2" / "indices" / variant / "index_manifest.json"
        )
        index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        index_bytes = sum(item["bytes"] for item in index_manifest["files"].values())

        for method in methods:
            for k in ks:
                sample_records = []
                totals = {
                    "supporting_fact_recall": 0.0,
                    "complete_evidence_coverage": 0,
                    "any_evidence_coverage": 0,
                    "gold_title_recall": 0.0,
                    "complete_gold_title_coverage": 0,
                    "retrieved_chunks": 0,
                    "unique_documents": 0,
                    "latency_ms": 0.0,
                }
                for qid, question in question_by_id.items():
                    ranking_record = ranking_by_id[qid]
                    rows = ranking_record["rankings"][method][:k]
                    chunk_ids = [row["chunk_id"] for row in rows]
                    metrics = supporting_fact_metrics(
                        chunk_ids, chunk_by_id, question["gold_supporting_facts"]
                    )
                    titles = unique_document_titles(chunk_ids, chunk_by_id)
                    gold_titles = {
                        title for title, _ in unique_supporting_facts(question["gold_supporting_facts"])
                    }
                    retrieved_titles = set(titles)
                    title_recall = len(gold_titles & retrieved_titles) / len(gold_titles)
                    latency_ms = method_latency_ms(ranking_record, method)
                    sample = {
                        "question_id": qid,
                        "split": args.split,
                        "question": question["question"],
                        "answer": question["answer"],
                        "chunk_variant": variant,
                        "method": method,
                        "k": k,
                        "retrieved_chunk_ids": chunk_ids,
                        "retrieved_document_titles": titles,
                        "retrieved_scores": [ranking_score(row, method) for row in rows],
                        "gold_supporting_facts": question["gold_supporting_facts"],
                        "covered_supporting_facts": metrics["covered_supporting_facts"],
                        "supporting_fact_recall": metrics["supporting_fact_recall"],
                        "complete_evidence_coverage": metrics["complete_evidence_coverage"],
                        "evidence_state": metrics["evidence_state"],
                        "gold_title_recall": title_recall,
                        "complete_gold_title_coverage": int(title_recall == 1.0),
                        "latency_ms": latency_ms,
                        "latency_note": "batched model time allocated per query",
                    }
                    sample_records.append(sample)
                    totals["supporting_fact_recall"] += metrics["supporting_fact_recall"]
                    totals["complete_evidence_coverage"] += metrics[
                        "complete_evidence_coverage"
                    ]
                    totals["any_evidence_coverage"] += int(
                        bool(metrics["covered_supporting_facts"])
                    )
                    totals["gold_title_recall"] += title_recall
                    totals["complete_gold_title_coverage"] += int(title_recall == 1.0)
                    totals["retrieved_chunks"] += len(chunk_ids)
                    totals["unique_documents"] += len(titles)
                    totals["latency_ms"] += latency_ms

                write_jsonl_atomic(per_sample_paths[(variant, method, k)], sample_records)
                count = len(sample_records)
                summaries.append(
                    {
                        "variant": variant,
                        "split": args.split,
                        "method": method,
                        "k": k,
                        "questions": count,
                        "supporting_fact_recall": totals["supporting_fact_recall"] / count,
                        "complete_evidence_coverage": totals[
                            "complete_evidence_coverage"
                        ]
                        / count,
                        "any_evidence_coverage": totals["any_evidence_coverage"] / count,
                        "gold_title_recall": totals["gold_title_recall"] / count,
                        "complete_gold_title_coverage": totals[
                            "complete_gold_title_coverage"
                        ]
                        / count,
                        "average_retrieved_chunks": totals["retrieved_chunks"] / count,
                        "average_unique_documents": totals["unique_documents"] / count,
                        "average_latency_ms": totals["latency_ms"] / count,
                        "index_storage_bytes": index_bytes,
                    }
                )
                logger.info("evaluated %s %s@%s", variant, method, k)

    temporary_csv = summary_csv.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    temporary_csv.replace(summary_csv)
    summary_md.write_text(render_markdown(summaries, args.split), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": 2,
        "phase1_frozen_commit": config["phase1_frozen_commit"],
        "git_commit": git_commit(ROOT),
        "config": portable_path(args.config, ROOT),
        "split": args.split,
        "variants": variants,
        "ranking_sources": ranking_sources,
        "question_count": len(question_by_id),
        "invalid_gold_questions_excluded": sum(
            question["split"] == args.split
            and not question.get("gold_annotation_valid", True)
            for question in questions.values()
        ),
        "metric_definition": "exact document_title and sentence_id supporting-fact coverage",
        "summaries": summaries,
    }
    write_json_atomic(manifest_path, manifest)
    logger.info("saved %s", summary_csv)


if __name__ == "__main__":
    main()

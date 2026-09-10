"""Run score-preserving Dense/BM25/RRF/Rerank retrieval for Phase 2 chunks."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import pickle
import time
from pathlib import Path

import numpy as np

from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
from phase2_retrieval import (
    bm25_tokenize,
    deterministic_top_indices,
    ranked_entries,
    retrieval_document_text,
    rrf_fuse_precomputed_bm25,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase2" / "chunk_retrieval.json"
VALID_SPLITS = ("train", "dev", "test")
_WORKER_BM25 = None
_WORKER_BM25_DEPTH = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--splits", default="dev", help="Comma-separated train,dev,test")
    parser.add_argument("--dense-model", help="Local dense model path or model name")
    parser.add_argument("--reranker-model", help="Local reranker path or model name")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--reranker-query-batch", type=int, default=32)
    parser.add_argument("--bm25-workers", type=int, default=1)
    parser.add_argument("--dense-search-batch-size", type=int, default=1)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Deterministic per-split question limit for non-formal throughput checks",
    )
    parser.add_argument(
        "--output-label",
        help="Required isolated output label when --limit-per-split is used",
    )
    parser.add_argument("--allow-download", action="store_true")
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


def _initialize_bm25_worker(bm25, depth: int) -> None:
    global _WORKER_BM25, _WORKER_BM25_DEPTH
    _WORKER_BM25 = bm25
    _WORKER_BM25_DEPTH = depth


def _score_bm25_worker(task: tuple[int, list[str]]) -> tuple[int, np.ndarray, np.ndarray, float]:
    position, query_tokens = task
    started = time.perf_counter()
    scores = _WORKER_BM25.get_scores(query_tokens)
    top = deterministic_top_indices(scores, _WORKER_BM25_DEPTH)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return position, top, np.asarray(scores[top]), elapsed_ms


def precompute_bm25_top(
    bm25, tokenized_queries: list[list[str]], depth: int, workers: int
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float]:
    started = time.perf_counter()
    results: list[tuple[np.ndarray, np.ndarray, float] | None] = [
        None
    ] * len(tokenized_queries)
    tasks = list(enumerate(tokenized_queries))
    if workers == 1:
        _initialize_bm25_worker(bm25, depth)
        iterator = map(_score_bm25_worker, tasks)
        pool = None
    else:
        if os.name != "posix":
            raise RuntimeError("Parallel BM25 requires POSIX fork semantics")
        context = multiprocessing.get_context("fork")
        pool = context.Pool(
            workers,
            initializer=_initialize_bm25_worker,
            initargs=(bm25, depth),
        )
        iterator = pool.imap_unordered(_score_bm25_worker, tasks, chunksize=1)
    try:
        for completed, (position, top, scores, elapsed_ms) in enumerate(iterator, start=1):
            results[position] = (top, scores, elapsed_ms)
            if completed % 500 == 0 or completed == len(tasks):
                logger.info("BM25 retrieval %s/%s", completed, len(tasks))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if any(result is None for result in results):
        raise RuntimeError("BM25 workers did not return every query")
    return results, time.perf_counter() - started


def validate_protocol(config: dict, variant: str, splits: list[str]) -> None:
    known_variants = {item["name"] for item in config["chunking"]["variants"]}
    if variant not in known_variants:
        raise ValueError(f"Unknown chunk variant {variant!r}; expected one of {sorted(known_variants)}")
    valid_splits = set(config.get("dataset", {}).get("valid_splits", VALID_SPLITS))
    unknown_splits = set(splits) - valid_splits
    if unknown_splits:
        raise ValueError(f"Unknown splits: {sorted(unknown_splits)}")
    if len(splits) != len(set(splits)):
        raise ValueError("Duplicate split names are not allowed")
    evaluation_split = config.get("dataset", {}).get("evaluation_split", "test")
    if evaluation_split in splits:
        selected = config["retrieval_evaluation"]["selected_variant_after_dev"]
        if selected is None:
            raise ValueError(
                "Held-out retrieval is locked until the transferred/selected variant is recorded"
            )
        if selected != variant:
            raise ValueError(f"Held-out evaluation is locked to variant {selected!r}")


def select_questions_for_run(
    questions: dict[str, dict], splits: list[str], limit_per_split: int | None
) -> tuple[list[dict], dict[str, int]]:
    eligible = {
        split: [question for question in questions.values() if question["split"] == split]
        for split in splits
    }
    eligible_counts = {split: len(rows) for split, rows in eligible.items()}
    if limit_per_split is None:
        selected = [
            question for question in questions.values() if question["split"] in set(splits)
        ]
    else:
        selected = []
        for split in splits:
            selected.extend(
                sorted(eligible[split], key=lambda row: row["question_id"])[
                    :limit_per_split
                ]
            )
    return selected, eligible_counts


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    validate_protocol(config, args.variant, splits)
    if args.limit_per_split is not None:
        if args.limit_per_split < 1:
            raise ValueError("--limit-per-split must be positive")
        if not args.output_label:
            raise ValueError("--output-label is required with --limit-per-split")
    elif args.output_label:
        raise ValueError("--output-label is only valid with --limit-per-split")
    if args.output_label and (
        not args.output_label.replace("-", "").replace("_", "").isalnum()
    ):
        raise ValueError("--output-label may contain only letters, numbers, '-' and '_'")
    if args.bm25_workers < 1:
        raise ValueError("--bm25-workers must be positive")
    if args.dense_search_batch_size < 1:
        raise ValueError("--dense-search-batch-size must be positive")

    data_dir = ROOT / config.get("outputs", {}).get("data_dir", "data/phase2")
    result_dir = ROOT / config.get("outputs", {}).get("result_dir", "results/phase2")
    chunk_path = data_dir / "chunks" / f"{args.variant}.jsonl"
    question_path = data_dir / "questions.json"
    index_dir = data_dir / "indices" / args.variant
    if args.limit_per_split is None:
        output_dir = result_dir / "retrieval" / args.variant
    else:
        output_dir = result_dir / "retrieval_smoke" / args.output_label / args.variant
    output_paths = {split: output_dir / f"{split}.jsonl" for split in splits}
    manifest_path = output_dir / f"run_manifest_{'_'.join(splits)}.json"
    existing = [path for path in [*output_paths.values(), manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite retrieval outputs: {existing}")

    required = [
        chunk_path,
        question_path,
        index_dir / "dense.faiss",
        index_dir / "doc_ids.npy",
        index_dir / "bm25.pkl",
        index_dir / "index_manifest.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing configured data/index files: {missing}")

    chunks = read_jsonl(chunk_path)
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    selected_questions, eligible_counts = select_questions_for_run(
        questions, splits, args.limit_per_split
    )
    if not selected_questions:
        raise ValueError("No questions selected")

    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    index = faiss.read_index(str(index_dir / "dense.faiss"))
    doc_ids = np.load(index_dir / "doc_ids.npy", allow_pickle=False).tolist()
    with (index_dir / "bm25.pkl").open("rb") as handle:
        bm25_payload = pickle.load(handle)
    if doc_ids != bm25_payload["doc_ids"]:
        raise ValueError("Dense and BM25 document orders differ")
    if len(doc_ids) != len(chunk_by_id) or index.ntotal != len(doc_ids):
        raise ValueError("Index, document IDs, and chunk file have different sizes")
    bm25 = bm25_payload["bm25"]

    retrieval_config = config["retrieval"]
    dense_model_name = args.dense_model or retrieval_config["dense_model"]
    reranker_model_name = args.reranker_model or retrieval_config["reranker_model"]
    queries = [question["question"] for question in selected_questions]
    instruction = retrieval_config.get("query_instruction")
    encoded_queries = [f"{instruction}{query}" for query in queries] if instruction else queries

    dense_depth = int(retrieval_config["dense_candidate_k"])
    bm25_depth = int(retrieval_config["bm25_candidate_k"])
    hybrid_depth = int(retrieval_config["hybrid_candidate_k"])
    saved_depth = int(retrieval_config["saved_ranking_depth"])
    if saved_depth > min(dense_depth, bm25_depth) or hybrid_depth < saved_depth:
        raise ValueError("Saved depth must fit dense, BM25, and hybrid candidate depths")

    tokenized_queries = [bm25_tokenize(question["question"]) for question in selected_questions]
    bm25_results, bm25_wall_seconds = precompute_bm25_top(
        bm25, tokenized_queries, bm25_depth, args.bm25_workers
    )

    dense_model = SentenceTransformer(
        dense_model_name, local_files_only=not args.allow_download
    )
    embed_started = time.perf_counter()
    query_embeddings = dense_model.encode(
        encoded_queries,
        normalize_embeddings=retrieval_config["dense_normalize_embeddings"],
        batch_size=args.query_batch_size,
        show_progress_bar=True,
    )
    embed_seconds = time.perf_counter() - embed_started
    embed_ms_per_query = embed_seconds * 1000.0 / len(selected_questions)

    dense_scores_by_query = []
    dense_indices_by_query = []
    dense_allocated_ms = []
    dense_wall_started = time.perf_counter()
    for batch_start in range(0, len(selected_questions), args.dense_search_batch_size):
        embeddings = np.asarray(
            query_embeddings[batch_start : batch_start + args.dense_search_batch_size],
            dtype="float32",
        )
        started = time.perf_counter()
        batch_scores, batch_indices = index.search(embeddings, dense_depth)
        allocated = (time.perf_counter() - started) * 1000.0 / len(embeddings)
        dense_scores_by_query.extend(batch_scores)
        dense_indices_by_query.extend(batch_indices)
        dense_allocated_ms.extend([allocated] * len(embeddings))
    dense_search_wall_seconds = time.perf_counter() - dense_wall_started

    records = []
    retrieval_started = time.perf_counter()
    for position, question in enumerate(selected_questions, start=1):
        dense_indices = dense_indices_by_query[position - 1]
        dense_scores = dense_scores_by_query[position - 1]
        bm25_top, bm25_top_scores, bm25_ms = bm25_results[position - 1]
        started = time.perf_counter()
        dense_candidate_bm25 = bm25.get_batch_scores(
            tokenized_queries[position - 1], [int(index) for index in dense_indices]
        )
        bm25_ms += (time.perf_counter() - started) * 1000.0
        bm25_score_by_index = {
            int(index): float(score) for index, score in zip(bm25_top, bm25_top_scores)
        }
        for index, score in zip(dense_indices, dense_candidate_bm25):
            bm25_score_by_index.setdefault(int(index), float(score))
        started = time.perf_counter()
        hybrid_rows = rrf_fuse_precomputed_bm25(
            dense_indices,
            dense_scores,
            bm25_top,
            bm25_score_by_index,
            dense_depth=dense_depth,
            bm25_depth=bm25_depth,
            rrf_k=int(retrieval_config["rrf_k"]),
            top=hybrid_depth,
        )
        rrf_ms = (time.perf_counter() - started) * 1000.0
        for rank, row in enumerate(hybrid_rows, start=1):
            row["chunk_id"] = doc_ids[row.pop("index")]
            row["rank"] = rank

        records.append(
            {
                "question_id": question["question_id"],
                "split": question["split"],
                "question": question["question"],
                "chunk_variant": args.variant,
                "rankings": {
                    "dense": ranked_entries(
                        dense_indices[:saved_depth], dense_scores[:saved_depth], doc_ids
                    ),
                    "bm25": ranked_entries(
                        bm25_top[:saved_depth], bm25_top_scores[:saved_depth], doc_ids
                    ),
                    "hybrid": hybrid_rows[:saved_depth],
                    "rerank": [],
                },
                "latency_ms": {
                    "query_embedding_run_mean": embed_ms_per_query,
                    "dense_search": dense_allocated_ms[position - 1],
                    "bm25_scoring": bm25_ms,
                    "rrf_fusion": rrf_ms,
                    "reranker_allocated": None,
                },
            }
        )
        if position % 500 == 0:
            logger.info("first-stage retrieval %s/%s", position, len(selected_questions))
    fusion_seconds = time.perf_counter() - retrieval_started
    first_stage_seconds = bm25_wall_seconds + dense_search_wall_seconds + fusion_seconds

    reranker = CrossEncoder(
        reranker_model_name,
        max_length=int(retrieval_config["reranker_max_length"]),
        local_files_only=not args.allow_download,
    )
    rerank_started = time.perf_counter()
    for batch_start in range(0, len(records), args.reranker_query_batch):
        batch = records[batch_start : batch_start + args.reranker_query_batch]
        pairs = []
        for record in batch:
            pairs.extend(
                (
                    record["question"],
                    retrieval_document_text(
                        chunk_by_id[row["chunk_id"]], retrieval_config["document_text"]
                    ),
                )
                for row in record["rankings"]["hybrid"]
            )
        started = time.perf_counter()
        predicted = reranker.predict(
            pairs,
            batch_size=args.reranker_batch_size,
            show_progress_bar=False,
        )
        predicted = np.asarray(predicted).reshape(-1)
        batch_ms_per_query = (time.perf_counter() - started) * 1000.0 / len(batch)
        offset = 0
        for record in batch:
            candidates = record["rankings"]["hybrid"]
            scores = predicted[offset : offset + len(candidates)]
            offset += len(candidates)
            order = deterministic_top_indices(scores, len(scores))
            reranked = []
            for rank, candidate_position in enumerate(order, start=1):
                row = dict(candidates[int(candidate_position)])
                row["rank"] = rank
                row["score"] = float(scores[int(candidate_position)])
                reranked.append(row)
            record["rankings"]["rerank"] = reranked
            record["latency_ms"]["reranker_allocated"] = batch_ms_per_query
        logger.info(
            "rerank %s/%s", min(batch_start + len(batch), len(records)), len(records)
        )
    rerank_seconds = time.perf_counter() - rerank_started

    for split, path in output_paths.items():
        split_records = [record for record in records if record["split"] == split]
        write_jsonl_atomic(path, split_records)
        logger.info("saved %s records -> %s", len(split_records), path)

    index_manifest = json.loads(
        (index_dir / "index_manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": int(config.get("phase", 2)),
        "phase1_frozen_commit": config.get("phase1_frozen_commit"),
        "phase3b_frozen_commit": config.get("phase3b_freeze", {}).get("commit"),
        "git_commit": git_commit(ROOT),
        "config": portable_path(args.config, ROOT),
        "variant": args.variant,
        "run_scope": "formal" if args.limit_per_split is None else "smoke_throughput_only",
        "limit_per_split": args.limit_per_split,
        "output_label": args.output_label,
        "selection_order": (
            "source_order" if args.limit_per_split is None else "question_id_ascending"
        ),
        "splits": splits,
        "eligible_question_counts": eligible_counts,
        "question_counts": {
            split: sum(record["split"] == split for record in records) for split in splits
        },
        "dense_model": dense_model_name,
        "reranker_model": reranker_model_name,
        "dense_candidate_k": dense_depth,
        "bm25_candidate_k": bm25_depth,
        "hybrid_candidate_k": hybrid_depth,
        "saved_ranking_depth": saved_depth,
        "rrf_k": retrieval_config["rrf_k"],
        "query_embedding_seconds": embed_seconds,
        "bm25_workers": args.bm25_workers,
        "bm25_wall_seconds": bm25_wall_seconds,
        "dense_search_batch_size": args.dense_search_batch_size,
        "dense_search_wall_seconds": dense_search_wall_seconds,
        "fusion_and_record_seconds": fusion_seconds,
        "first_stage_seconds": first_stage_seconds,
        "reranking_seconds": rerank_seconds,
        "index_chunk_sha256": index_manifest["chunk_file_sha256"],
        "outputs": {split: portable_path(path, ROOT) for split, path in output_paths.items()},
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()

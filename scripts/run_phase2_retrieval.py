"""Run score-preserving Dense/BM25/RRF/Rerank retrieval for Phase 2 chunks."""

from __future__ import annotations

import argparse
import json
import logging
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
    rrf_fuse_with_components,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase2" / "chunk_retrieval.json"
VALID_SPLITS = ("train", "dev", "test")


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


def validate_protocol(config: dict, variant: str, splits: list[str]) -> None:
    known_variants = {item["name"] for item in config["chunking"]["variants"]}
    if variant not in known_variants:
        raise ValueError(f"Unknown chunk variant {variant!r}; expected one of {sorted(known_variants)}")
    unknown_splits = set(splits) - set(VALID_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unknown splits: {sorted(unknown_splits)}")
    if len(splits) != len(set(splits)):
        raise ValueError("Duplicate split names are not allowed")
    if "test" in splits:
        selected = config["retrieval_evaluation"]["selected_variant_after_dev"]
        if selected is None:
            raise ValueError(
                "Test retrieval is locked until selected_variant_after_dev is recorded in the config"
            )
        if selected != variant:
            raise ValueError(f"Test is locked to the dev-selected variant {selected!r}")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    validate_protocol(config, args.variant, splits)

    chunk_path = ROOT / "data" / "phase2" / "chunks" / f"{args.variant}.jsonl"
    question_path = ROOT / "data" / "phase2" / "questions.json"
    index_dir = ROOT / "data" / "phase2" / "indices" / args.variant
    output_dir = ROOT / "results" / "phase2" / "retrieval" / args.variant
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
        raise FileNotFoundError(f"Missing Phase 2 data/index files: {missing}")

    chunks = read_jsonl(chunk_path)
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    selected_questions = [
        question for question in questions.values() if question["split"] in set(splits)
    ]
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

    dense_depth = int(retrieval_config["dense_candidate_k"])
    bm25_depth = int(retrieval_config["bm25_candidate_k"])
    hybrid_depth = int(retrieval_config["hybrid_candidate_k"])
    saved_depth = int(retrieval_config["saved_ranking_depth"])
    if saved_depth > min(dense_depth, bm25_depth) or hybrid_depth < saved_depth:
        raise ValueError("Saved depth must fit dense, BM25, and hybrid candidate depths")

    records = []
    retrieval_started = time.perf_counter()
    for position, (question, embedding) in enumerate(zip(selected_questions, query_embeddings), start=1):
        started = time.perf_counter()
        dense_scores, dense_indices = index.search(
            np.asarray([embedding], dtype="float32"), dense_depth
        )
        dense_ms = (time.perf_counter() - started) * 1000.0
        dense_indices = dense_indices[0]
        dense_scores = dense_scores[0]

        started = time.perf_counter()
        bm25_scores = bm25.get_scores(bm25_tokenize(question["question"]))
        bm25_top = deterministic_top_indices(bm25_scores, bm25_depth)
        bm25_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        hybrid_rows = rrf_fuse_with_components(
            dense_indices,
            dense_scores,
            bm25_scores,
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
                        bm25_top[:saved_depth], bm25_scores[bm25_top[:saved_depth]], doc_ids
                    ),
                    "hybrid": hybrid_rows[:saved_depth],
                    "rerank": [],
                },
                "latency_ms": {
                    "query_embedding_run_mean": embed_ms_per_query,
                    "dense_search": dense_ms,
                    "bm25_scoring": bm25_ms,
                    "rrf_fusion": rrf_ms,
                    "reranker_allocated": None,
                },
            }
        )
        if position % 500 == 0:
            logger.info("first-stage retrieval %s/%s", position, len(selected_questions))
    first_stage_seconds = time.perf_counter() - retrieval_started

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
        "phase": 2,
        "phase1_frozen_commit": config["phase1_frozen_commit"],
        "git_commit": git_commit(ROOT),
        "config": portable_path(args.config, ROOT),
        "variant": args.variant,
        "splits": splits,
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
        "first_stage_seconds": first_stage_seconds,
        "reranking_seconds": rerank_seconds,
        "index_chunk_sha256": index_manifest["chunk_file_sha256"],
        "outputs": {split: portable_path(path, ROOT) for split, path in output_paths.items()},
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()

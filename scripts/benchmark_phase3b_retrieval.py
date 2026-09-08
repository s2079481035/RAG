"""Fresh per-query Phase 3B retrieval-component latency benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np

from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
from generate_phase3b_stage_answers import validate_test_gate
from phase2_retrieval import bm25_tokenize, deterministic_top_indices, retrieval_document_text, rrf_fuse_with_components
from train_phase2_controller import read_jsonl, write_jsonl_atomic


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"
RETRIEVAL_CONFIG = ROOT / "configs" / "phase2" / "chunk_retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", required=True, choices=["dev_policy", "test"])
    parser.add_argument("--dense-model")
    parser.add_argument("--reranker-model")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def synchronize(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def timed_call(function, torch_module):
    synchronize(torch_module)
    started = time.perf_counter()
    value = function()
    synchronize(torch_module)
    return value, (time.perf_counter() - started) * 1000.0


def latency_summary(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.split == "test":
        validate_test_gate(config, args.config)
    retrieval_config = json.loads(RETRIEVAL_CONFIG.read_text(encoding="utf-8"))["retrieval"]
    variant = config["base_controller"]["variant"]
    index_dir = ROOT / "data" / "phase2" / "indices" / variant
    chunks_path = ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl"
    frozen_split = "dev" if args.split == "dev_policy" else "test"
    frozen_path = ROOT / "results" / "phase2" / "retrieval" / variant / f"{frozen_split}.jsonl"
    frozen_rows = read_jsonl(frozen_path)
    if args.split == "dev_policy":
        ids = read_ids(repo_path(config["dev_split"]["policy_ids"]))
        frozen_rows = [row for row in frozen_rows if row["question_id"] in ids]
    expected = int(config["dev_split"]["policy_questions"]) if args.split == "dev_policy" else 1000
    if len(frozen_rows) != expected:
        raise ValueError(f"Expected {expected} frozen retrieval rows, got {len(frozen_rows)}")

    output_dir = ROOT / "results" / "phase3b" / "latency"
    output_path = output_dir / f"{args.split}_retrieval.jsonl"
    manifest_path = output_dir / f"{args.split}_retrieval_manifest.json"
    existing = [path for path in [output_path, manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite retrieval benchmark: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import faiss
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    chunks = read_jsonl(chunks_path)
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    index = faiss.read_index(str(index_dir / "dense.faiss"))
    doc_ids = np.load(index_dir / "doc_ids.npy", allow_pickle=False).tolist()
    with (index_dir / "bm25.pkl").open("rb") as handle:
        bm25_payload = pickle.load(handle)
    if doc_ids != bm25_payload["doc_ids"]:
        raise ValueError("Dense and BM25 document orders differ")
    bm25 = bm25_payload["bm25"]
    dense_name = args.dense_model or os.environ.get("DENSE_MODEL_PATH") or retrieval_config["dense_model"]
    reranker_name = args.reranker_model or os.environ.get("RERANKER_MODEL_PATH") or retrieval_config["reranker_model"]
    dense_model = SentenceTransformer(dense_name, local_files_only=not args.allow_download)
    reranker = CrossEncoder(
        reranker_name,
        max_length=int(retrieval_config["reranker_max_length"]),
        local_files_only=not args.allow_download,
    )
    instruction = retrieval_config.get("query_instruction")
    dense_depth = int(retrieval_config["dense_candidate_k"])
    bm25_depth = int(retrieval_config["bm25_candidate_k"])
    hybrid_depth = int(retrieval_config["hybrid_candidate_k"])

    def execute(record: dict, measured: bool) -> dict | None:
        question = record["question"]
        encoded_question = f"{instruction}{question}" if instruction else question
        embedding, embedding_ms = timed_call(
            lambda: dense_model.encode(
                [encoded_question], normalize_embeddings=retrieval_config["dense_normalize_embeddings"], show_progress_bar=False
            ),
            torch,
        )
        (dense_scores, dense_indices), dense_search_ms = timed_call(
            lambda: index.search(np.asarray(embedding, dtype="float32"), dense_depth), torch
        )
        dense_scores, dense_indices = dense_scores[0], dense_indices[0]
        bm25_scores, bm25_ms = timed_call(
            lambda: bm25.get_scores(bm25_tokenize(question)), torch
        )
        _, bm25_top_ms = timed_call(
            lambda: deterministic_top_indices(bm25_scores, bm25_depth), torch
        )
        _, rrf_ms = timed_call(
            lambda: rrf_fuse_with_components(
                dense_indices,
                dense_scores,
                bm25_scores,
                dense_depth=dense_depth,
                bm25_depth=bm25_depth,
                rrf_k=int(retrieval_config["rrf_k"]),
                top=hybrid_depth,
            ),
            torch,
        )
        frozen_hybrid = record["rankings"]["hybrid"][:hybrid_depth]
        pairs = [
            (question, retrieval_document_text(chunk_by_id[row["chunk_id"]], retrieval_config["document_text"]))
            for row in frozen_hybrid
        ]
        _, reranker_ms = timed_call(
            lambda: reranker.predict(pairs, batch_size=len(pairs), show_progress_bar=False), torch
        )
        if not measured:
            return None
        dense_stage = embedding_ms + dense_search_ms
        hybrid_stage = dense_stage + bm25_ms + bm25_top_ms + rrf_ms
        rerank_stage = hybrid_stage + reranker_ms
        return {
            "question_id": record["question_id"],
            "phase3b_split": args.split,
            "query_embedding_ms": embedding_ms,
            "dense_search_ms": dense_search_ms,
            "bm25_scoring_ms": bm25_ms,
            "bm25_topk_ms": bm25_top_ms,
            "rrf_fusion_ms": rrf_ms,
            "reranker_ms": reranker_ms,
            "dense@5_ms": dense_stage,
            "hybrid@10_ms": hybrid_stage,
            "rerank@20_ms": rerank_stage,
            "reranker_candidate_count": len(pairs),
            "cuda_synchronized": bool(torch.cuda.is_available()),
            "frozen_rankings_reused_for_reranker_workload": True,
        }

    warmup = int(config["cost_measurement"]["warmup_questions"])
    for row in frozen_rows[:warmup]:
        execute(row, measured=False)
    measured_rows = []
    for index_value, row in enumerate(frozen_rows, start=1):
        measured_rows.append(execute(row, measured=True))
        if index_value % 50 == 0:
            logger.info("benchmarked retrieval %s/%s", index_value, len(frozen_rows))
    write_jsonl_atomic(output_path, measured_rows)
    summary = {
        key: latency_summary(measured_rows, key)
        for key in [
            "query_embedding_ms",
            "dense_search_ms",
            "bm25_scoring_ms",
            "bm25_topk_ms",
            "rrf_fusion_ms",
            "reranker_ms",
            "dense@5_ms",
            "hybrid@10_ms",
            "rerank@20_ms",
        ]
    }
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3B",
            "git_commit": git_commit(ROOT),
            "split": args.split,
            "questions": len(measured_rows),
            "warmup_questions": warmup,
            "cuda_synchronized": bool(torch.cuda.is_available()),
            "dense_model": dense_name,
            "reranker_model": reranker_name,
            "frozen_rankings": portable_path(frozen_path, ROOT),
            "frozen_rankings_sha256": file_sha256(frozen_path),
            "output": portable_path(output_path, ROOT),
            "output_sha256": file_sha256(output_path),
            "summary": summary,
            "environment": collect_environment(ROOT),
        },
    )
    logger.info("saved retrieval benchmark -> %s", output_path)


if __name__ == "__main__":
    main()

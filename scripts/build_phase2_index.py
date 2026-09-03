"""Build isolated dense and BM25 indices for one Phase 2 chunk variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
from phase2_retrieval import bm25_tokenize, retrieval_document_text


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase2" / "chunk_retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model", help="Local model path or Hugging Face model name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    variants = {item["name"] for item in config["chunking"]["variants"]}
    if args.variant not in variants:
        raise ValueError(f"Unknown chunk variant {args.variant!r}; expected one of {sorted(variants)}")

    chunk_path = ROOT / "data" / "phase2" / "chunks" / f"{args.variant}.jsonl"
    output_dir = ROOT / "data" / "phase2" / "indices" / args.variant
    paths = {
        "dense": output_dir / "dense.faiss",
        "doc_ids": output_dir / "doc_ids.npy",
        "bm25": output_dir / "bm25.pkl",
        "manifest": output_dir / "index_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite index files: {existing}")
    if not chunk_path.exists():
        raise FileNotFoundError(f"Build Phase 2 chunks first: {chunk_path}")

    chunks = load_chunks(chunk_path)
    doc_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Chunk IDs are not unique")
    document_text_mode = config["retrieval"]["document_text"]
    texts = [retrieval_document_text(chunk, document_text_mode) for chunk in chunks]
    model_name = args.model or config["retrieval"]["dense_model"]
    output_dir.mkdir(parents=True, exist_ok=True)

    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(model_name, local_files_only=not args.allow_download)
    embeddings = model.encode(
        texts,
        normalize_embeddings=config["retrieval"]["dense_normalize_embeddings"],
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(paths["dense"]))
    np.save(paths["doc_ids"], np.asarray(doc_ids))
    dense_seconds = time.perf_counter() - started

    started = time.perf_counter()
    bm25 = BM25Okapi([bm25_tokenize(text) for text in texts])
    with paths["bm25"].open("wb") as handle:
        pickle.dump({"doc_ids": doc_ids, "bm25": bm25}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bm25_seconds = time.perf_counter() - started

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": 2,
        "phase1_frozen_commit": config["phase1_frozen_commit"],
        "git_commit": git_commit(ROOT),
        "config": portable_path(args.config, ROOT),
        "variant": args.variant,
        "chunk_file": portable_path(chunk_path, ROOT),
        "chunk_file_sha256": file_sha256(chunk_path),
        "chunks": len(chunks),
        "document_text_mode": document_text_mode,
        "document_text_template": config["retrieval"]["document_text_template"],
        "dense_model": model_name,
        "dense_dimensions": int(embeddings.shape[1]),
        "dense_normalized": config["retrieval"]["dense_normalize_embeddings"],
        "dense_index": config["retrieval"]["faiss_index"],
        "dense_build_seconds": dense_seconds,
        "bm25_implementation": config["retrieval"]["bm25_implementation"],
        "bm25_parameters": config["retrieval"]["bm25_parameters"],
        "bm25_build_seconds": bm25_seconds,
        "files": {
            name: {"path": portable_path(path, ROOT), "bytes": path.stat().st_size}
            for name, path in paths.items()
            if name != "manifest"
        },
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(paths["manifest"], manifest)
    logger.info("indexed %s chunks for %s", len(chunks), args.variant)


if __name__ == "__main__":
    main()

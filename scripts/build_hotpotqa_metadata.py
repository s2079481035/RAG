"""Reconstruct sentence-aware HotpotQA metadata without changing legacy data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from experiment_utils import portable_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEGACY_DATA = ROOT / "data" / "hotpotqa_all.json"
DEFAULT_OUTPUT = ROOT / "data" / "sufficiency" / "hotpotqa_metadata.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-data", type=Path, default=DEFAULT_LEGACY_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-scan", type=int, default=7405)
    parser.add_argument(
        "--source-parquet",
        type=Path,
        help="Optional local HotpotQA distractor/validation Parquet file",
    )
    parser.add_argument("--force", action="store_true", help="Replace only the sidecar output")
    return parser.parse_args()


def iter_source_items(source_parquet: Path | None):
    if source_parquet is not None:
        import pyarrow.parquet as parquet

        logger.info("reading local source parquet: %s", source_parquet)
        parquet_file = parquet.ParquetFile(source_parquet)
        for batch in parquet_file.iter_batches(batch_size=256):
            yield from batch.to_pylist()
        return

    from datasets import load_dataset

    logger.info("reading HotpotQA through datasets streaming")
    yield from load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True
    )


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force to replace the sidecar")

    legacy = json.loads(args.legacy_data.read_text(encoding="utf-8"))
    legacy_kb = legacy["kb"]
    legacy_qids = {question["qid"] for question in legacy["questions"]}

    chunks = {}
    questions = {}
    all_doc_count = 0
    for scanned, item in enumerate(iter_source_items(args.source_parquet), start=1):
        if scanned > args.max_scan:
            break
        qid = f"hotpot_{scanned:06d}"
        titles = item["context"]["title"]
        sentence_groups = item["context"]["sentences"]
        for context_index, title in enumerate(titles):
            sentences = sentence_groups[context_index] if context_index < len(sentence_groups) else []
            if not isinstance(sentences, list):
                sentences = [str(sentences)]
            doc_id = f"hp_{all_doc_count:06d}"
            all_doc_count += 1
            if doc_id not in legacy_kb:
                continue
            text = " ".join(sentences)
            if text != legacy_kb[doc_id]:
                raise ValueError(f"Legacy KB text mismatch for {doc_id}; source revision may have changed")
            chunks[doc_id] = {
                "chunk_id": doc_id,
                "document_title": title,
                "sentence_ids": list(range(len(sentences))),
                "source_question_id": qid,
                "source_context_index": context_index,
            }

        supporting = item["supporting_facts"]
        gold = [
            {"title": title, "sentence_id": int(sentence_id)}
            for title, sentence_id in zip(supporting["title"], supporting["sent_id"])
        ]
        if qid in legacy_qids:
            questions[qid] = {
                "question_id": qid,
                "source_question_id": item.get("id"),
                "gold_supporting_facts": gold,
            }
        if scanned % 500 == 0:
            logger.info("scanned=%d chunks=%d questions=%d", scanned, len(chunks), len(questions))

    missing_chunks = sorted(set(legacy_kb) - set(chunks))
    missing_questions = sorted(legacy_qids - set(questions))
    if missing_chunks or missing_questions:
        raise ValueError(
            f"Metadata reconstruction incomplete: missing_chunks={len(missing_chunks)}, "
            f"missing_questions={len(missing_questions)}"
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "validation",
            "max_scan": args.max_scan,
            "source_parquet": portable_path(args.source_parquet, ROOT) if args.source_parquet else None,
            "source_parquet_sha256": file_sha256(args.source_parquet) if args.source_parquet else None,
            "legacy_data": portable_path(args.legacy_data, ROOT),
            "legacy_data_sha256": file_sha256(args.legacy_data),
        },
        "chunk_granularity": "whole_context_document",
        "chunks": chunks,
        "questions": questions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    logger.info("saved %s (%d chunks, %d questions)", args.output, len(chunks), len(questions))


if __name__ == "__main__":
    main()

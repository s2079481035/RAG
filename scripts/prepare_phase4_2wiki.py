"""Audit official 2WikiMultiHopQA and build its frozen shared chunk corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from evidence_utils import unique_supporting_facts
from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
from phase2_chunking import canonicalize_articles, sentence_aligned_chunks
from phase4_2wiki import (
    deterministic_train_split,
    require_official_record,
    source_statistics,
    split_digest,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--tokenizer", help="Local tokenizer path or frozen model name")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_official(path: Path, split: str) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Official {split} file must contain a JSON list")
    return [require_official_record(record, split) for record in value]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def mapping_audit(chunks: list[dict], questions: dict[str, dict]) -> dict:
    available = defaultdict(set)
    for chunk in chunks:
        available[chunk["document_title"]].update(chunk["sentence_ids"])
    by_split = defaultdict(lambda: {"gold": 0, "mapped": 0})
    missing = []
    for question in questions.values():
        for title, sentence_id in unique_supporting_facts(
            question["gold_supporting_facts"]
        ):
            by_split[question["split"]]["gold"] += 1
            if sentence_id in available[title]:
                by_split[question["split"]]["mapped"] += 1
            else:
                missing.append(
                    {
                        "question_id": question["question_id"],
                        "split": question["split"],
                        "title": title,
                        "sentence_id": sentence_id,
                    }
                )
    total = sum(item["gold"] for item in by_split.values())
    mapped = sum(item["mapped"] for item in by_split.values())
    return {
        "gold_supporting_facts": total,
        "mapped_supporting_facts": mapped,
        "mapping_rate": mapped / total,
        "by_split": {
            split: {
                **counts,
                "mapping_rate": counts["mapped"] / counts["gold"],
            }
            for split, counts in sorted(by_split.items())
        },
        "missing_supporting_facts": missing,
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def render_dataset_audit(
    train_stats: dict,
    dev_stats: dict,
    split_counts: Counter,
    mapping: dict,
    source_paths: dict[str, Path],
    source_hashes: dict[str, str],
    split_sha256: str,
) -> str:
    lines = [
        "# 2WikiMultiHopQA Dataset Audit",
        "",
        "The input is validated against the official JSON structure. Format differences cause an explicit failure; no field is silently remapped.",
        "",
        "## Sources and Roles",
        "",
        *markdown_table(
            ["Official split", "Experiment role", "Questions", "Source", "SHA-256"],
            [
                ["Train", "train_core/dev_calibration/dev_policy", train_stats["questions"], portable_path(source_paths["train"], ROOT), source_hashes["train"]],
                ["Dev", "heldout evaluation only", dev_stats["questions"], portable_path(source_paths["dev"], ROOT), source_hashes["dev"]],
            ],
        ),
        "",
        f"- Question-level split digest: `{split_sha256}`",
        "- Official unlabeled Test is not used.",
        "- Supporting facts were observed and required as exact `[title, sentence_id]` pairs.",
        "",
        "## Experiment Splits",
        "",
        *markdown_table(
            ["Split", "Questions"],
            [[split, split_counts[split]] for split in sorted(split_counts)],
        ),
        "",
    ]
    for name, stats in [("Official Train", train_stats), ("Official Dev", dev_stats)]:
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Raw context document instances: {stats['raw_context_documents']}",
                f"- Unique document titles: {stats['unique_titles']}",
                f"- Repeated-title groups: {stats['repeated_title_groups']}",
                f"- Repeated-title instances beyond first: {stats['repeated_title_instances_beyond_first']}",
                f"- Unique exact document texts: {stats['unique_texts']}",
                f"- Duplicate-text groups: {stats['duplicate_text_groups']}",
                f"- Duplicate-text instances beyond first: {stats['duplicate_text_instances_beyond_first']}",
                f"- Contexts per question: `{json.dumps(stats['contexts_per_question'])}`",
                f"- Supporting-fact counts: `{json.dumps(stats['supporting_fact_count_distribution'])}`",
                f"- Answer types: `{json.dumps(stats['answer_types'])}`",
                f"- Question types: `{json.dumps(stats['question_types'])}`",
                f"- Observed source key sets: `{json.dumps(stats['observed_source_key_sets'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Gold-to-Chunk Mapping",
            "",
            f"- Gold supporting facts: {mapping['gold_supporting_facts']}",
            f"- Mapped supporting facts: {mapping['mapped_supporting_facts']}",
            f"- Mapping rate: {mapping['mapping_rate']:.6f}",
            f"- Missing facts: {len(mapping['missing_supporting_facts'])}",
            "",
            "Mapping requires exact `chunk.document_title == gold.title` and membership of `gold.sentence_id` in `chunk.sentence_ids`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_corpus_audit(
    corpus: dict, dedup: dict, chunk_count: int, oversized: int
) -> str:
    return "\n".join(
        [
            "# 2Wiki Shared Benchmark-Context Corpus Audit",
            "",
            "This corpus pools context documents from official Train and labeled Dev, then deduplicates exact title-plus-text instances. It is not Full Wikipedia retrieval.",
            "",
            f"- Raw document instances: {corpus['raw_document_count']}",
            f"- Unique titles: {corpus['unique_title_count']}",
            f"- Unique exact texts: {corpus['unique_text_count']}",
            f"- Deduplicated title-plus-text documents: {dedup['canonical_articles']}",
            f"- Exact duplicate instances removed: {dedup['duplicate_title_identical_text_extra_instances']}",
            f"- Same-title/different-text titles: {dedup['same_title_different_text_titles']}",
            f"- Different-title/identical-text groups: {dedup['different_title_identical_text_groups']}",
            f"- Sentence-segmentation conflicts: {len(dedup['sentence_segmentation_conflicts'])}",
            f"- Frozen sentence-aligned 256-token chunks: {chunk_count}",
            f"- Oversized single-sentence chunks: {oversized}",
            "",
            "No 2Wiki chunk-size sweep is performed; the HotpotQA-selected approximately 256-token setting is transferred unchanged.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["phase3b_freeze"]["commit"] != "1ed762c":
        raise ValueError("Phase 4 must retain the frozen Phase 3B commit")
    dataset = config["dataset"]
    train_path = (args.train or ROOT / dataset["official_train"]).resolve()
    dev_path = (args.dev or ROOT / dataset["official_dev"]).resolve()
    missing_sources = [path for path in [train_path, dev_path] if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"Missing official 2Wiki source files: {missing_sources}")

    output_dir = ROOT / config["outputs"]["data_dir"]
    chunk_path = output_dir / "chunks" / "sentence_256.jsonl"
    question_path = output_dir / "questions.json"
    manifest_path = output_dir / "preparation_manifest.json"
    split_dir = output_dir / "splits"
    audit_path = ROOT / config["outputs"]["dataset_audit"]
    corpus_audit_path = ROOT / config["outputs"]["corpus_audit"]
    split_names = config["dataset"]["valid_splits"]
    split_paths = {split: split_dir / f"{split}_ids.txt" for split in split_names}
    targets = [chunk_path, question_path, manifest_path, audit_path, corpus_audit_path, *split_paths.values()]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 4 preparation outputs: {existing}")

    train_questions = read_official(train_path, "train")
    dev_questions = read_official(dev_path, "dev")
    train_ids = {question["question_id"] for question in train_questions}
    dev_ids = {question["question_id"] for question in dev_questions}
    overlap = train_ids & dev_ids
    if overlap:
        raise ValueError(f"Official Train/Dev question-ID overlap: {sorted(overlap)[:5]}")
    internal = dataset["internal_split"]
    train_split = deterministic_train_split(
        train_ids,
        seed=int(internal["seed"]),
        dev_calibration_questions=int(internal["dev_calibration_questions"]),
        dev_policy_questions=int(internal["dev_policy_questions"]),
    )

    questions = {}
    document_instances = []
    for source_questions in [train_questions, dev_questions]:
        for question in source_questions:
            qid = question["question_id"]
            split = train_split[qid] if question["official_split"] == "train" else "heldout"
            questions[qid] = {
                key: value
                for key, value in question.items()
                if key not in {"contexts", "source_keys"}
            }
            questions[qid]["split"] = split
            questions[qid]["gold_annotation_valid"] = True
            for index, context in enumerate(question["contexts"]):
                text = config["chunking"]["sentence_separator"].join(
                    context["sentence_texts"]
                )
                document_instances.append(
                    {
                        "legacy_doc_id": f"2wiki_{question['official_split']}_{qid}_{index:02d}",
                        "document_title": context["document_title"],
                        "sentence_texts": context["sentence_texts"],
                        "text": text,
                        "source_question_id": qid,
                        "source_context_index": index,
                    }
                )

    from transformers import AutoTokenizer

    tokenizer_name = args.tokenizer or config["chunking"]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=not args.allow_download
    )
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    corpus_stats = {
        "raw_document_count": len(document_instances),
        "unique_title_count": len(
            {instance["document_title"] for instance in document_instances}
        ),
        "unique_text_count": len({instance["text"] for instance in document_instances}),
    }
    articles, dedup = canonicalize_articles(document_instances)
    variant = config["chunking"]["variants"]
    if variant != [{"name": "sentence_256", "target_tokens": 256}]:
        raise ValueError("Phase 4 2Wiki chunking must remain frozen to sentence_256")
    chunks = []
    for article in articles:
        chunks.extend(
            sentence_aligned_chunks(
                article,
                256,
                token_count,
                config["chunking"]["sentence_separator"],
            )
        )
    mapping = mapping_audit(chunks, questions)
    if mapping["missing_supporting_facts"]:
        raise ValueError(
            "Gold supporting facts do not all map to frozen chunks; inspect source format "
            f"before proceeding: {mapping['missing_supporting_facts'][:10]}"
        )

    split_to_ids = {
        split: sorted(
            qid for qid, question in questions.items() if question["split"] == split
        )
        for split in split_names
    }
    split_sha256 = split_digest(split_to_ids)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, path in split_paths.items():
        path.write_text("\n".join(split_to_ids[split]) + "\n", encoding="utf-8")
    write_jsonl_atomic(chunk_path, chunks)
    write_json_atomic(question_path, questions)

    source_paths = {"train": train_path, "dev": dev_path}
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}
    train_stats = source_statistics(train_questions)
    dev_stats = source_statistics(dev_questions)
    split_counts = Counter(question["split"] for question in questions.values())
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        render_dataset_audit(
            train_stats,
            dev_stats,
            split_counts,
            mapping,
            source_paths,
            source_hashes,
            split_sha256,
        ),
        encoding="utf-8",
    )
    corpus_audit_path.write_text(
        render_corpus_audit(
            corpus_stats,
            dedup,
            len(chunks),
            sum(chunk["oversized_single_sentence"] for chunk in chunks),
        ),
        encoding="utf-8",
    )
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "phase3b_frozen_commit": config["phase3b_freeze"]["commit"],
            "git_commit": git_commit(ROOT),
            "config": portable_path(args.config, ROOT),
            "sources": {
                name: {"path": portable_path(path, ROOT), "sha256": source_hashes[name]}
                for name, path in source_paths.items()
            },
            "question_counts": dict(sorted(split_counts.items())),
            "split_unit": "question_id",
            "split_seed": internal["seed"],
            "split_sha256": split_sha256,
            "corpus_scope": config["corpus_scope"],
            "corpus_statistics": {
                **corpus_stats,
                "deduplicated_document_count": dedup["canonical_articles"],
            },
            "dedup": dedup,
            "chunks": len(chunks),
            "chunk_file": portable_path(chunk_path, ROOT),
            "chunk_file_sha256": file_sha256(chunk_path),
            "supporting_fact_mapping": mapping,
            "tokenizer": tokenizer_name,
            "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
            "environment": collect_environment(ROOT),
        },
    )
    logger.info("prepared %d questions and %d frozen chunks", len(questions), len(chunks))


if __name__ == "__main__":
    main()

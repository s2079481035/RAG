"""Build canonical sentence-aligned HotpotQA chunk KBs and Phase 2 audits."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evidence_utils import assert_disjoint_question_ids, unique_supporting_facts, validate_tuning_splits
from experiment_utils import collect_environment, git_commit, portable_path, utc_now, write_json_atomic
from phase2_chunking import canonicalize_articles, sentence_aligned_chunks


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase2" / "chunk_retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def iter_source(source_parquet: Path | None, dataset_config: dict):
    if source_parquet:
        import pyarrow.parquet as parquet

        parquet_file = parquet.ParquetFile(source_parquet)
        for batch in parquet_file.iter_batches(batch_size=256):
            yield from batch.to_pylist()
        return
    from datasets import load_dataset

    yield from load_dataset(
        dataset_config["source"],
        dataset_config["source_config"],
        split=dataset_config["source_split"],
        streaming=True,
    )


def load_token_counter(tokenizer_json: Path | None, tokenizer_name: str, allow_download: bool):
    if tokenizer_json:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids), {
            "name": tokenizer_name,
            "implementation": "tokenizers.Tokenizer",
            "tokenizer_json": portable_path(tokenizer_json, ROOT),
        }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, local_files_only=not allow_download
    )
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False)), {
        "name": tokenizer_name,
        "implementation": "transformers.AutoTokenizer",
        "resolved_revision": getattr(tokenizer, "_commit_hash", None),
    }


def reconstruct_instances(source_items, legacy_kb: dict) -> tuple[list[dict], dict, int]:
    document_instances = []
    source_questions = {}
    all_doc_count = 0
    for scanned, item in enumerate(source_items, start=1):
        qid = f"hotpot_{scanned:06d}"
        context = item["context"]
        for context_index, (title, sentences) in enumerate(
            zip(context["title"], context["sentences"])
        ):
            legacy_doc_id = f"hp_{all_doc_count:06d}"
            all_doc_count += 1
            if legacy_doc_id not in legacy_kb:
                continue
            sentence_texts = list(sentences)
            text = " ".join(sentence_texts)
            if text != legacy_kb[legacy_doc_id]:
                raise ValueError(f"Source text mismatch for {legacy_doc_id}")
            document_instances.append(
                {
                    "legacy_doc_id": legacy_doc_id,
                    "document_title": title,
                    "sentence_texts": sentence_texts,
                    "text": text,
                    "source_question_id": qid,
                    "source_context_index": context_index,
                }
            )
        supporting = item["supporting_facts"]
        source_questions[qid] = {
            "source_question_id": item["id"],
            "gold_supporting_facts": [
                {"title": title, "sentence_id": int(sentence_id)}
                for title, sentence_id in zip(supporting["title"], supporting["sent_id"])
            ],
        }
    return document_instances, source_questions, all_doc_count


def split_map(all_questions: list[dict], test_questions: list[dict]) -> dict[str, str]:
    test = all_questions[:1000]
    remaining = all_questions[1000:]
    groups = {"train": remaining[:5405], "dev": remaining[5405:6405], "test": test}
    if {q["qid"] for q in test} != {q["qid"] for q in test_questions}:
        raise ValueError("Frozen Phase 1 test split mismatch")
    assert_disjoint_question_ids(
        {name: [question["qid"] for question in questions] for name, questions in groups.items()}
    )
    validate_tuning_splits(["train", "dev"])
    return {
        question["qid"]: split_name
        for split_name, questions in groups.items()
        for question in questions
    }


def write_jsonl_atomic(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(path)


def chunk_statistics(chunks: list[dict], article_count: int, target_tokens: int) -> dict:
    token_counts = np.asarray([chunk["token_count"] for chunk in chunks], dtype=int)
    regular = token_counts[token_counts <= target_tokens]
    underfill = target_tokens - regular
    article_chunk_counts = Counter(chunk["article_id"] for chunk in chunks)
    single_chunk_articles = sum(count == 1 for count in article_chunk_counts.values())
    return {
        "chunks": len(chunks),
        "mean_tokens": float(token_counts.mean()),
        "p50_tokens": float(np.percentile(token_counts, 50)),
        "p95_tokens": float(np.percentile(token_counts, 95)),
        "max_tokens": int(token_counts.max()),
        "mean_chunks_per_article": len(chunks) / article_count,
        "max_chunks_per_article": max(article_chunk_counts.values()),
        "single_chunk_articles": single_chunk_articles,
        "single_chunk_article_ratio": single_chunk_articles / article_count,
        "multi_chunk_articles": article_count - single_chunk_articles,
        "oversized_single_sentences": int(np.sum(token_counts > target_tokens)),
        "mean_sentence_boundary_underfill_tokens": float(underfill.mean()),
        "p95_sentence_boundary_underfill_tokens": float(np.percentile(underfill, 95)),
        "mean_budget_utilization": float(np.mean(regular / target_tokens)),
    }


def supporting_fact_mapping_audit(chunks: list[dict], questions: dict) -> dict:
    available = defaultdict(set)
    for chunk in chunks:
        available[chunk["document_title"]].update(chunk["sentence_ids"])
    total = mapped = 0
    by_split = defaultdict(lambda: {"gold_supporting_facts": 0, "mapped_supporting_facts": 0})
    missing = []
    for qid, question in questions.items():
        split = question["split"]
        for title, sentence_id in unique_supporting_facts(question["gold_supporting_facts"]):
            total += 1
            by_split[split]["gold_supporting_facts"] += 1
            if sentence_id in available[title]:
                mapped += 1
                by_split[split]["mapped_supporting_facts"] += 1
            else:
                missing.append(
                    {
                        "question_id": qid,
                        "split": split,
                        "title": title,
                        "sentence_id": sentence_id,
                    }
                )
    return {
        "gold_supporting_facts": total,
        "mapped_supporting_facts": mapped,
        "mapping_rate": mapped / total,
        "by_split": {
            split: {
                **counts,
                "mapping_rate": counts["mapped_supporting_facts"]
                / counts["gold_supporting_facts"],
            }
            for split, counts in sorted(by_split.items())
        },
        "missing_supporting_facts": missing,
    }


def render_dedup_audit(audit: dict) -> str:
    return "\n".join(
        [
            "# Phase 2 KB Dedup Audit",
            "",
            "Canonicalization only merges exact `document_title + joined article text` duplicates.",
            "The corpus membership is held fixed to the Phase 1 shared gold-context pool; this is not the full HotpotQA distractor-context pool.",
            "",
            "| Category | Groups/Titles | Instances/Variants |",
            "|---|---:|---:|",
            f"| duplicate title + identical text | {audit['duplicate_title_identical_text_groups']} | {audit['duplicate_title_identical_text_instances']} |",
            f"| same title + different text | {audit['same_title_different_text_titles']} | {audit['same_title_different_text_variants']} |",
            f"| different title + identical text | {audit['different_title_identical_text_groups']} | {audit['different_title_identical_text_titles']} |",
            "",
            f"- Legacy document instances: {audit['document_instances']}",
            f"- Official source context instances scanned: {audit['source_context_instances_scanned']}",
            f"- Source contexts outside frozen Phase 1 KB: {audit['source_context_instances_excluded']}",
            f"- Every frozen KB instance is a gold document for at least one question: {audit['all_frozen_instances_are_gold_contexts']}",
            f"- Canonical articles: {audit['canonical_articles']}",
            f"- Removed duplicate instances: {audit['duplicate_title_identical_text_extra_instances']}",
            f"- Maximum duplicate group size: {audit['max_duplicate_group_size']}",
            f"- Sentence-segmentation conflicts among exact duplicates: {len(audit['sentence_segmentation_conflicts'])}",
            "",
            "Different titles are never merged, even if their text matches. Same-title text variants would also remain distinct. Supporting facts are deduplicated as `(title, sentence_id)` before coverage is computed.",
            "",
        ]
    )


def render_chunking_audit(stats_by_variant: dict, mapping_by_variant: dict, tokenizer_info: dict) -> str:
    lines = [
        "# Phase 2 Chunking Audit",
        "",
        f"Tokenizer: `{tokenizer_info['name']}` without special tokens in the chunk budget.",
        "Sentence-aligned greedy packing, overlap=0. A sentence is never split; a sentence exceeding the budget forms one oversized chunk.",
        "",
        "Natural paragraph boundaries are unavailable: HotpotQA exposes each context article as a flat sentence list and does not preserve paragraph markers. The legacy whole-article unit remains the paragraph/article-scale reference and is not mislabeled as a recovered natural paragraph.",
        "",
        "| Variant | Chunks | Mean tokens | P50 | P95 | Max | Mean chunks/article | Oversized sentences | Mean underfill | Mapping rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in stats_by_variant.items():
        mapping = mapping_by_variant[name]
        lines.append(
            f"| {name} | {stats['chunks']} | {stats['mean_tokens']:.2f} | "
            f"{stats['p50_tokens']:.0f} | {stats['p95_tokens']:.0f} | {stats['max_tokens']} | "
            f"{stats['mean_chunks_per_article']:.2f} | {stats['oversized_single_sentences']} | "
            f"{stats['mean_sentence_boundary_underfill_tokens']:.2f} | {mapping['mapping_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Sentence-boundary deviation is reported as unused budget for non-oversized chunks. Dev/test supporting-fact mapping must be 1.0 before retrieval experiments are allowed to run.",
            "",
            "| Variant | Train mapping | Dev mapping | Test mapping |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, mapping in mapping_by_variant.items():
        by_split = mapping["by_split"]
        lines.append(
            f"| {name} | {by_split['train']['mapping_rate']:.6f} | "
            f"{by_split['dev']['mapping_rate']:.6f} | {by_split['test']['mapping_rate']:.6f} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Granularity check",
            "",
            "| Variant | Single-chunk articles | Multi-chunk articles | Single-chunk ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, stats in stats_by_variant.items():
        lines.append(
            f"| {name} | {stats['single_chunk_articles']} | {stats['multi_chunk_articles']} | "
            f"{stats['single_chunk_article_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "A high single-chunk ratio means the target budget often leaves the original short HotpotQA context article intact. This limits how strongly that variant repairs the Phase 1 article-granularity problem and must be considered together with dev retrieval results.",
            "",
        ]
    )
    missing = next(iter(mapping_by_variant.values()))["missing_supporting_facts"]
    if missing:
        lines.extend(
            [
                "## Invalid source annotations",
                "",
                "Original gold annotations are preserved and never auto-corrected. Questions with impossible train annotations are retained with `gold_annotation_valid=false` and excluded from supervised Controller training.",
                "",
                "| Question | Split | Title | Sentence ID |",
                "|---|---|---|---:|",
            ]
        )
        for item in missing:
            lines.append(
                f"| {item['question_id']} | {item['split']} | {item['title']} | {item['sentence_id']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    dataset_config = config["dataset"]
    output_config = config["outputs"]
    output_dir = ROOT / output_config["data_dir"]
    chunk_dir = output_dir / "chunks"
    audit_paths = [ROOT / output_config["chunking_audit"], ROOT / output_config["dedup_audit"]]
    variants = config["chunking"]["variants"]
    targets = [chunk_dir / f"{variant['name']}.jsonl" for variant in variants]
    targets += [output_dir / "questions.json", output_dir / "chunk_manifest.json", *audit_paths]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 2 outputs: {[str(path) for path in existing]}")

    all_data = load_json(ROOT / dataset_config["legacy_all_questions"])
    test_data = load_json(ROOT / dataset_config["legacy_test"])
    split_by_qid = split_map(all_data["questions"], test_data["questions"])
    instances, source_questions, source_context_count = reconstruct_instances(
        iter_source(args.source_parquet, dataset_config), all_data["kb"]
    )
    articles, dedup_audit = canonicalize_articles(instances)
    legacy_gold_ids = {
        doc_id for question in all_data["questions"] for doc_id in question["gold_docs"]
    }
    if not legacy_gold_ids.issubset(all_data["kb"]):
        raise ValueError("Frozen Phase 1 gold document IDs are missing from its KB")
    dedup_audit.update(
        {
            "source_context_instances_scanned": source_context_count,
            "source_context_instances_excluded": source_context_count - len(instances),
            "all_frozen_instances_are_gold_contexts": set(all_data["kb"]) == legacy_gold_ids,
            "corpus_scope": config["corpus_scope"],
        }
    )
    logger.info(
        "canonicalized %d instances -> %d articles", len(instances), len(articles)
    )

    token_count, tokenizer_info = load_token_counter(
        args.tokenizer_json, config["chunking"]["tokenizer"], args.allow_download
    )
    questions = {
        question["qid"]: {
            "question_id": question["qid"],
            "question": question["question"],
            "answer": question["answer"],
            "type": question.get("type"),
            "split": split_by_qid[question["qid"]],
            "gold_supporting_facts": source_questions[question["qid"]][
                "gold_supporting_facts"
            ],
        }
        for question in all_data["questions"]
    }
    stats_by_variant = {}
    mapping_by_variant = {}
    for variant in variants:
        chunks = []
        for article in articles:
            chunks.extend(
                sentence_aligned_chunks(
                    article,
                    int(variant["target_tokens"]),
                    token_count,
                    config["chunking"]["sentence_separator"],
                )
            )
        if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
            raise ValueError(f"Duplicate canonical chunk IDs in {variant['name']}")
        mapping = supporting_fact_mapping_audit(chunks, questions)
        invalid_eval = [
            item for item in mapping["missing_supporting_facts"] if item["split"] in {"dev", "test"}
        ]
        if invalid_eval:
            raise ValueError(
                f"Invalid dev/test supporting facts for {variant['name']}: {invalid_eval}"
            )
        path = chunk_dir / f"{variant['name']}.jsonl"
        write_jsonl_atomic(path, chunks)
        stats_by_variant[variant["name"]] = chunk_statistics(
            chunks, len(articles), int(variant["target_tokens"])
        )
        mapping_by_variant[variant["name"]] = mapping
        logger.info("saved %s (%d chunks)", path, len(chunks))

    invalid_by_qid = defaultdict(list)
    for item in next(iter(mapping_by_variant.values()))["missing_supporting_facts"]:
        invalid_by_qid[item["question_id"]].append(item)
    for qid, question in questions.items():
        question["gold_annotation_valid"] = qid not in invalid_by_qid
        question["invalid_gold_supporting_facts"] = invalid_by_qid[qid]
    write_json_atomic(output_dir / "questions.json", questions)

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": 2,
        "phase1_frozen_commit": config["phase1_frozen_commit"],
        "git_commit": git_commit(ROOT),
        "config": portable_path(args.config, ROOT),
        "source_parquet": portable_path(args.source_parquet, ROOT) if args.source_parquet else None,
        "tokenizer": tokenizer_info,
        "dedup": dedup_audit,
        "chunking": stats_by_variant,
        "supporting_fact_mapping": mapping_by_variant,
        "environment": collect_environment(ROOT),
    }
    write_json_atomic(output_dir / "chunk_manifest.json", manifest)
    audit_paths[0].parent.mkdir(parents=True, exist_ok=True)
    audit_paths[0].write_text(
        render_chunking_audit(stats_by_variant, mapping_by_variant, tokenizer_info),
        encoding="utf-8",
    )
    audit_paths[1].write_text(render_dedup_audit(dedup_audit), encoding="utf-8")
    logger.info("Phase 2 chunk build complete")


if __name__ == "__main__":
    main()

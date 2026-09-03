"""Pure sentence-aligned chunking and canonicalization helpers for Phase 2."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any


def stable_article_id(title: str, text: str) -> str:
    digest = hashlib.sha256(f"{title}\0{text}".encode("utf-8")).hexdigest()[:20]
    return f"article_{digest}"


def canonicalize_articles(document_instances: Iterable[dict[str, Any]]) -> tuple[list[dict], dict]:
    """Collapse only exact title+text duplicates and audit other duplicate categories."""
    by_title_text: dict[tuple[str, str], list[dict]] = defaultdict(list)
    texts_by_title: dict[str, set[str]] = defaultdict(set)
    titles_by_text: dict[str, set[str]] = defaultdict(set)
    for instance in document_instances:
        key = (instance["document_title"], instance["text"])
        by_title_text[key].append(instance)
        texts_by_title[key[0]].add(key[1])
        titles_by_text[key[1]].add(key[0])

    articles = []
    segmentation_conflicts = []
    for (title, text), instances in sorted(by_title_text.items()):
        segmentations = {tuple(instance["sentence_texts"]) for instance in instances}
        if len(segmentations) > 1:
            segmentation_conflicts.append(
                {"document_title": title, "instances": len(instances), "segmentations": len(segmentations)}
            )
        for segmentation_index, sentences in enumerate(sorted(segmentations)):
            identity_text = text if len(segmentations) == 1 else f"{text}\0segmentation={segmentation_index}"
            article_id = stable_article_id(title, identity_text)
            source_instances = [
                instance for instance in instances if tuple(instance["sentence_texts"]) == sentences
            ]
            articles.append(
                {
                    "article_id": article_id,
                    "document_title": title,
                    "sentence_texts": list(sentences),
                    "text": text,
                    "legacy_doc_ids": sorted(instance["legacy_doc_id"] for instance in source_instances),
                    "source_question_ids": sorted(
                        {instance["source_question_id"] for instance in source_instances}
                    ),
                    "duplicate_instance_count": len(source_instances),
                }
            )

    duplicate_groups = [instances for instances in by_title_text.values() if len(instances) > 1]
    audit = {
        "document_instances": sum(len(instances) for instances in by_title_text.values()),
        "canonical_articles": len(articles),
        "duplicate_title_identical_text_groups": len(duplicate_groups),
        "duplicate_title_identical_text_instances": sum(len(group) for group in duplicate_groups),
        "duplicate_title_identical_text_extra_instances": sum(len(group) - 1 for group in duplicate_groups),
        "max_duplicate_group_size": max((len(group) for group in duplicate_groups), default=1),
        "same_title_different_text_titles": sum(len(texts) > 1 for texts in texts_by_title.values()),
        "same_title_different_text_variants": sum(
            len(texts) for texts in texts_by_title.values() if len(texts) > 1
        ),
        "different_title_identical_text_groups": sum(
            len(titles) > 1 for titles in titles_by_text.values()
        ),
        "different_title_identical_text_titles": sum(
            len(titles) for titles in titles_by_text.values() if len(titles) > 1
        ),
        "sentence_segmentation_conflicts": segmentation_conflicts,
    }
    return articles, audit


def sentence_aligned_chunks(
    article: dict[str, Any],
    target_tokens: int,
    token_count: Callable[[str], int],
    sentence_separator: str = " ",
) -> list[dict[str, Any]]:
    """Greedily pack complete consecutive sentences without overlap."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    sentences = article["sentence_texts"]
    if not sentences:
        return []

    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for sentence_id, sentence_text in enumerate(sentences):
        candidate = current + [(sentence_id, sentence_text)]
        candidate_text = sentence_separator.join(text for _, text in candidate)
        if current and token_count(candidate_text) > target_tokens:
            groups.append(current)
            current = [(sentence_id, sentence_text)]
        else:
            current = candidate
    if current:
        groups.append(current)

    chunks = []
    for group in groups:
        sentence_ids = [sentence_id for sentence_id, _ in group]
        sentence_texts = [text for _, text in group]
        chunk_text = sentence_separator.join(sentence_texts)
        chunk_tokens = token_count(chunk_text)
        first_sentence = sentence_ids[0]
        last_sentence = sentence_ids[-1]
        chunk_id = (
            f"{article['article_id']}_b{target_tokens}_s{first_sentence:04d}-{last_sentence:04d}"
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "article_id": article["article_id"],
                "document_title": article["document_title"],
                "sentence_ids": sentence_ids,
                "sentence_texts": sentence_texts,
                "chunk_text": chunk_text,
                "token_count": chunk_tokens,
                "target_tokens": target_tokens,
                "oversized_single_sentence": len(group) == 1 and chunk_tokens > target_tokens,
                "legacy_doc_ids": article["legacy_doc_ids"],
            }
        )
    return chunks

import unittest

import numpy as np

from scripts.phase2_retrieval import (
    cumulative_unique,
    deterministic_top_indices,
    parse_stage,
    retrieval_document_text,
    rrf_fuse_precomputed_bm25,
    rrf_fuse_with_components,
    score_statistics,
)


class Phase2RetrievalTests(unittest.TestCase):
    def test_top_indices_break_ties_by_corpus_position(self):
        ranked = deterministic_top_indices([0.2, 0.8, 0.8, 0.1], 3)
        np.testing.assert_array_equal(ranked, [1, 2, 0])

    def test_rrf_retains_single_retriever_candidates(self):
        rows = rrf_fuse_with_components(
            [0, 1],
            [0.9, 0.8],
            [0.0, 0.0, 3.0],
            dense_depth=2,
            bm25_depth=1,
            rrf_k=60,
            top=3,
        )
        self.assertEqual({row["index"] for row in rows}, {0, 1, 2})
        self.assertIsNone(next(row for row in rows if row["index"] == 2)["dense_rank"])

    def test_precomputed_bm25_rrf_matches_full_score_vector(self):
        dense_indices = [0, 1]
        dense_scores = [0.9, 0.8]
        bm25_scores = np.asarray([0.1, 0.2, 3.0, 2.0])
        expected = rrf_fuse_with_components(
            dense_indices,
            dense_scores,
            bm25_scores,
            dense_depth=2,
            bm25_depth=2,
            rrf_k=60,
            top=4,
        )
        bm25_top = deterministic_top_indices(bm25_scores, 2)
        score_lookup = {
            int(index): float(bm25_scores[index])
            for index in set(dense_indices) | set(bm25_top)
        }
        actual = rrf_fuse_precomputed_bm25(
            dense_indices,
            dense_scores,
            bm25_top,
            score_lookup,
            dense_depth=2,
            bm25_depth=2,
            rrf_k=60,
            top=4,
        )
        self.assertEqual(actual, expected)

    def test_cumulative_evidence_preserves_first_seen_order(self):
        self.assertEqual(
            cumulative_unique([["a", "b"], ["b", "c"], ["a", "d"]]),
            [["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"]],
        )

    def test_score_statistics_are_finite(self):
        stats = score_statistics([2.0, 1.0, -1.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["top1_top2_margin"], 1.0)
        self.assertGreaterEqual(stats["softmax_entropy"], 0.0)
        self.assertLessEqual(stats["softmax_entropy"], 1.0)

    def test_controller_stage_parser(self):
        self.assertEqual(parse_stage("hybrid@10"), ("hybrid", 10))
        with self.assertRaises(ValueError):
            parse_stage("hybrid")

    def test_retrieval_text_includes_title_and_chunk(self):
        text = retrieval_document_text(
            {"document_title": "Entity", "chunk_text": "Some evidence."},
            "title_and_chunk_text",
        )
        self.assertEqual(text, "[TITLE] Entity [TEXT] Some evidence.")

import unittest

import numpy as np

from scripts.evidence_utils import (
    assert_disjoint_question_ids,
    supporting_fact_metrics,
    validate_tuning_splits,
)
from scripts.retrieval_utils import rrf_fuse_indices
from scripts.critic_metrics import classification_metrics


class SupportingFactMappingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = {
            "c1": {"document_title": "Scott Derrickson", "sentence_ids": [0, 1, 2]},
            "c2": {"document_title": "Doctor Strange", "sentence_ids": [0, 1]},
            "c3": {"document_title": "Scott Derrickson", "sentence_ids": [0]},
            "c4": {"document_title": "Doctor Strange", "sentence_ids": [0]},
        }
        self.gold = [
            {"title": "Scott Derrickson", "sentence_id": 0},
            {"title": "Doctor Strange", "sentence_id": 1},
        ]

    def test_title_and_sentence_id_are_both_required(self):
        metrics = supporting_fact_metrics(["c1"], self.chunks, self.gold)
        self.assertEqual(metrics["covered_supporting_facts"], [self.gold[0]])
        self.assertEqual(metrics["supporting_fact_recall"], 0.5)
        wrong_sentence = supporting_fact_metrics(["c4"], self.chunks, [self.gold[1]])
        self.assertEqual(wrong_sentence["covered_supporting_facts"], [])

    def test_duplicate_chunks_do_not_duplicate_fact_counts(self):
        metrics = supporting_fact_metrics(["c1", "c3", "c1"], self.chunks, self.gold)
        self.assertEqual(len(metrics["covered_supporting_facts"]), 1)
        self.assertEqual(metrics["supporting_fact_recall"], 0.5)

    def test_supporting_fact_recall_never_exceeds_one(self):
        duplicated_gold = self.gold + [self.gold[0], self.gold[1]]
        metrics = supporting_fact_metrics(["c1", "c2", "c3"], self.chunks, duplicated_gold)
        self.assertLessEqual(metrics["supporting_fact_recall"], 1.0)
        self.assertEqual(metrics["supporting_fact_recall"], 1.0)

    def test_complete_coverage_requires_every_gold_fact(self):
        partial = supporting_fact_metrics(["c1"], self.chunks, self.gold)
        complete = supporting_fact_metrics(["c1", "c2"], self.chunks, self.gold)
        self.assertEqual(partial["complete_evidence_coverage"], 0)
        self.assertEqual(complete["complete_evidence_coverage"], 1)
        self.assertEqual(partial["evidence_state"], "partial")
        self.assertEqual(complete["evidence_state"], "sufficient")


class RetrievalAndSplitSafetyTests(unittest.TestCase):
    def test_rrf_keeps_dense_only_and_bm25_only_documents(self):
        top, _ = rrf_fuse_indices(
            np.array([0, 1, 2]),
            np.array([0.0, 0.1, 10.0]),
            n_docs=3,
            dense_depth=1,
            bm25_depth=1,
            top=2,
        )
        self.assertEqual(set(top.tolist()), {0, 2})

    def test_question_ids_must_be_disjoint(self):
        assert_disjoint_question_ids({"train": ["q1"], "dev": ["q2"], "test": ["q3"]})
        with self.assertRaises(ValueError):
            assert_disjoint_question_ids({"train": ["q1"], "test": ["q1"]})

    def test_test_split_cannot_be_used_for_tuning(self):
        validate_tuning_splits(["dev"])
        with self.assertRaises(ValueError):
            validate_tuning_splits(["dev", "test"])

    def test_false_stop_rate_uses_actual_non_sufficient_denominator(self):
        labels = np.array([0, 1, 1, 2])
        probabilities = np.array(
            [
                [0.1, 0.1, 0.8],
                [0.1, 0.2, 0.7],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        metrics, _ = classification_metrics(
            labels,
            probabilities,
            label_names=["insufficient", "partial", "sufficient"],
        )
        self.assertEqual(metrics["false_stop_count"], 2)
        self.assertEqual(metrics["actual_non_sufficient_count"], 3)
        self.assertAlmostEqual(metrics["false_stop_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from phase4_2wiki import (  # noqa: E402
    answer_type,
    deterministic_train_split,
    require_official_record,
    source_statistics,
    split_digest,
)
from phase4_score_threshold import (  # noqa: E402
    attach_normalized_scores,
    fit_stage_minmax,
    minmax_score,
)


def record(qid="q1"):
    return {
        "_id": qid,
        "question": "Who was born first?",
        "answer": "Alice",
        "supporting_facts": [["Alice", 0], ["Bob", 1]],
        "context": [
            ["Alice", ["Alice was born in 1900."]],
            ["Bob", ["Bob was a person.", "Bob was born in 1910."]],
        ],
        "type": "comparison",
        "evidences": [["Alice", "birth year", "1900"]],
    }


class Phase42WikiTests(unittest.TestCase):
    def test_requires_official_pair_formats(self):
        parsed = require_official_record(record(), "train")
        self.assertEqual(parsed["gold_supporting_facts"][1], {"title": "Bob", "sentence_id": 1})
        broken = record()
        broken["context"] = {"title": ["Alice"], "sentences": [["text"]]}
        with self.assertRaisesRegex(ValueError, "no context documents"):
            require_official_record(broken, "train")

    def test_question_split_is_deterministic_disjoint_and_exhaustive(self):
        ids = [f"q{i}" for i in range(20)]
        first = deterministic_train_split(
            ids, seed=42, dev_calibration_questions=4, dev_policy_questions=5
        )
        second = deterministic_train_split(
            reversed(ids), seed=42, dev_calibration_questions=4, dev_policy_questions=5
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(value == "dev_calibration" for value in first.values()), 4)
        self.assertEqual(sum(value == "dev_policy" for value in first.values()), 5)
        self.assertEqual(sum(value == "train_core" for value in first.values()), 11)

    def test_split_digest_is_order_independent(self):
        left = {"train_core": ["b", "a"], "heldout": ["c"]}
        right = {"heldout": ["c"], "train_core": ["a", "b"]}
        self.assertEqual(split_digest(left), split_digest(right))

    def test_answer_types_are_declared_heuristics(self):
        self.assertEqual(answer_type("yes"), "yes_no")
        self.assertEqual(answer_type("1999"), "year")
        self.assertEqual(answer_type("12.5%"), "number")
        self.assertEqual(answer_type("June 1, 1956"), "date")
        self.assertEqual(answer_type("Gregg Harper"), "span")

    def test_source_statistics_reports_duplicates(self):
        first = require_official_record(record("q1"), "train")
        second = require_official_record(record("q2"), "train")
        stats = source_statistics([first, second])
        self.assertEqual(stats["questions"], 2)
        self.assertEqual(stats["raw_context_documents"], 4)
        self.assertEqual(stats["unique_titles"], 2)
        self.assertEqual(stats["duplicate_text_instances_beyond_first"], 2)

    def test_score_normalization_is_fit_per_stage_and_clipped(self):
        def stage_row(qid, stage, method, score):
            return {
                "question_id": qid,
                "stage": stage,
                "retrieval_statistics": {
                    "current_method": method,
                    method: {"max": score},
                },
            }

        rows = [
            stage_row("q1", "dense@5", "dense", 0.2),
            stage_row("q2", "dense@5", "dense", 0.6),
            stage_row("q1", "hybrid@10", "hybrid", 0.01),
            stage_row("q2", "hybrid@10", "hybrid", 0.03),
        ]
        parameters = fit_stage_minmax(rows, ["dense@5", "hybrid@10"])
        self.assertAlmostEqual(minmax_score(0.4, parameters["dense@5"]), 0.5)
        heldout = [stage_row("q3", "dense@5", "dense", 0.8)]
        scored = attach_normalized_scores(heldout, parameters)
        self.assertEqual(scored[0]["normalized_score"], 1.0)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from scripts.compare_phase4_retrieval_runs import compare_value, read_by_question


class Phase4RetrievalParityTests(unittest.TestCase):
    def test_float_tolerance_and_exact_ranking_identity(self):
        differences = []
        compare_value(
            {"chunk_id": "a", "score": 0.5},
            {"chunk_id": "a", "score": 0.5000001},
            "row",
            1e-6,
            differences,
        )
        self.assertEqual(differences, [])
        compare_value(["a", "b"], ["b", "a"], "ranking", 1e-6, differences)
        self.assertTrue(differences)

    def test_duplicate_question_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                '{"question_id":"q1"}\n{"question_id":"q1"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate question_id"):
                read_by_question(path)


if __name__ == "__main__":
    unittest.main()

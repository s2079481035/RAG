import tempfile
import unittest
from pathlib import Path

from scripts.compare_phase4_retrieval_runs import compare_value, read_by_question, sha256


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

    def test_sha256_identifies_compared_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"question_id":"q1"}\n', encoding="utf-8")
            self.assertEqual(
                sha256(path),
                "0fb68cdcb574a7f8ab59e78f454b3c721c51a02926c188bd51572a3eaa8b5a59",
            )


if __name__ == "__main__":
    unittest.main()

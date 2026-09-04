import unittest

from scripts.phase3a_generation_utils import (
    canonical_rows_sha256,
    exact_match_score,
    extract_short_answer,
    normalize_answer,
    token_f1_score,
)


class Phase3AGenerationTests(unittest.TestCase):
    def test_extracts_only_first_nonempty_answer_line(self):
        raw = "\nFinal answer: Paris\nBecause the context says so."

        self.assertEqual(extract_short_answer(raw), "Paris")

    def test_hotpot_normalization_removes_articles_and_punctuation(self):
        self.assertEqual(normalize_answer("The Foo, Bar!"), "foo bar")
        self.assertEqual(exact_match_score("The Eiffel Tower", "Eiffel Tower"), 1)

    def test_special_yes_no_answers_do_not_receive_partial_credit(self):
        self.assertEqual(token_f1_score("yes", "no"), 0.0)
        self.assertEqual(token_f1_score("yes because", "yes"), 0.0)

    def test_audit_hash_ignores_human_fields_but_detects_automatic_edits(self):
        fields = ["question_id", "raw_output"]
        original = [{"question_id": "q1", "raw_output": "Paris", "reviewer": ""}]
        reviewed = [{"question_id": "q1", "raw_output": "Paris", "reviewer": "R1"}]
        changed = [{"question_id": "q1", "raw_output": "London", "reviewer": "R1"}]

        self.assertEqual(
            canonical_rows_sha256(original, fields),
            canonical_rows_sha256(reviewed, fields),
        )
        self.assertNotEqual(
            canonical_rows_sha256(original, fields),
            canonical_rows_sha256(changed, fields),
        )


if __name__ == "__main__":
    unittest.main()

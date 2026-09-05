import csv
import tempfile
import unittest
from pathlib import Path

from scripts.review_manual_sufficiency_audit import read_audit, save_annotation


FIELDS = [
    "audit_id",
    "question",
    "automatic_three_class_label",
    "human_sufficient_to_answer",
    "human_label",
    "human_notes",
    "reviewer",
]


class Phase3AManualReviewTests(unittest.TestCase):
    def make_audit(self, directory: str) -> Path:
        path = Path(directory) / "audit.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "audit_id": "1",
                    "question": "Who?",
                    "automatic_three_class_label": "partial",
                    "human_sufficient_to_answer": "",
                    "human_label": "",
                    "human_notes": "",
                    "reviewer": "",
                }
            )
        return path

    def test_save_derives_binary_label_and_preserves_automatic_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_audit(directory)
            saved = save_annotation(
                path,
                "1",
                {
                    "human_label": "sufficient",
                    "human_notes": "Alternative evidence completes the answer.",
                    "reviewer": "R1",
                },
            )
            _, rows = read_audit(path)

        self.assertEqual(saved["human_sufficient_to_answer"], "yes")
        self.assertEqual(rows[0]["human_label"], "sufficient")
        self.assertEqual(rows[0]["automatic_three_class_label"], "partial")
        self.assertEqual(rows[0]["question"], "Who?")

    def test_disagreement_requires_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_audit(directory)

            with self.assertRaisesRegex(ValueError, "human_notes"):
                save_annotation(
                    path,
                    "1",
                    {"human_label": "insufficient", "human_notes": "", "reviewer": "R1"},
                )

    def test_matching_partial_label_maps_to_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_audit(directory)
            save_annotation(
                path,
                "1",
                {"human_label": "partial", "human_notes": "", "reviewer": "R1"},
            )
            _, rows = read_audit(path)

        self.assertEqual(rows[0]["human_sufficient_to_answer"], "no")


if __name__ == "__main__":
    unittest.main()

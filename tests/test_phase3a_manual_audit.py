import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_manual_sufficiency_audit import (
    THREE_CLASS,
    classification_summary,
    read_completed_audit,
)


FIELDS = [
    "audit_id",
    "automatic_three_class_label",
    "automatic_stop_label",
    "human_sufficient_to_answer",
    "human_label",
    "human_notes",
    "reviewer",
]


class Phase3AManualAuditTests(unittest.TestCase):
    def write_audit(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_utf8_bom_is_removed_and_labels_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            self.write_audit(
                path,
                [
                    {
                        "audit_id": "1",
                        "automatic_three_class_label": "sufficient",
                        "automatic_stop_label": "1",
                        "human_sufficient_to_answer": "yes",
                        "human_label": "sufficient",
                        "human_notes": "agrees",
                        "reviewer": "R1",
                    }
                ],
            )

            rows = read_completed_audit(path)

            self.assertIn("audit_id", rows[0])
            self.assertEqual(rows[0]["human_stop_label"], 1)

    def test_incomplete_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            self.write_audit(
                path,
                [
                    {
                        "audit_id": "1",
                        "automatic_three_class_label": "partial",
                        "automatic_stop_label": "0",
                        "human_sufficient_to_answer": "",
                        "human_label": "",
                        "human_notes": "",
                        "reviewer": "",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_completed_audit(path)

    def test_three_class_sufficient_metrics_use_human_as_reference(self):
        summary = classification_summary(
            ["insufficient", "partial", "sufficient"],
            ["partial", "partial", "sufficient"],
            THREE_CLASS,
        )

        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["per_class"]["sufficient"]["precision"], 1.0)
        self.assertEqual(summary["per_class"]["sufficient"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

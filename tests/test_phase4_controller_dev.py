import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_phase4_controller_dev import (  # noqa: E402
    aggregate_seed_rows,
    enrich_selection,
    select_primary,
    validate_split_rows,
)


class Phase4ControllerDevTests(unittest.TestCase):
    def test_protocol_excludes_terminal_from_controller_risk(self):
        config = json.loads(
            (ROOT / "configs" / "phase4" / "protocol.json").read_text()
        )
        policy = config["in_domain_policy"]
        self.assertEqual(policy["decision_stages"], ["dense@5", "hybrid@10"])
        self.assertEqual(policy["forced_final_stage"], "rerank@20")
        self.assertEqual(policy["operational_seed"], 42)
        self.assertTrue(policy["heldout_tuning_forbidden"])

    def test_heldout_predictions_are_rejected(self):
        rows = [
            {"question_id": "q1", "stage": stage, "split": "heldout"}
            for stage in ["dense@5", "hybrid@10", "rerank@20"]
        ]
        with self.assertRaisesRegex(ValueError, "cannot read split"):
            validate_split_rows(
                rows, "heldout", ["dense@5", "hybrid@10", "rerank@20"]
            )

    def test_incomplete_trajectory_is_rejected(self):
        rows = [
            {"question_id": "q1", "stage": "dense@5", "split": "dev_policy"},
            {"question_id": "q1", "stage": "rerank@20", "split": "dev_policy"},
        ]
        with self.assertRaisesRegex(ValueError, "Invalid stage order"):
            validate_split_rows(
                rows, "dev_policy", ["dense@5", "hybrid@10", "rerank@20"]
            )

    def test_seed_summary_uses_sample_standard_deviation(self):
        rows = []
        for seed, value in [(42, 0.2), (123, 0.3), (2026, 0.4)]:
            row = {
                "group": "final_evidence_aware",
                "seed": seed,
                "evaluation_scope": "reached_actionable_sequential_decisions",
            }
            for metric in [
                "macro_f1",
                "stop_f1",
                "auroc",
                "false_stop_rate",
                "hard_partial_false_stop_rate",
                "unnecessary_escalation_rate",
            ]:
                row[metric] = value
            rows.append(row)
        summary = aggregate_seed_rows(rows)[0]
        self.assertAlmostEqual(summary["mean_macro_f1"], 0.3)
        self.assertAlmostEqual(summary["std_macro_f1"], 0.1)

    def test_primary_policy_selection_is_exact(self):
        selections = [
            {
                "calibration_method": "temperature",
                "selection": "conservative",
                "risk_target": 0.1,
                "status": "selected",
                "threshold": 0.93,
            },
            {
                "calibration_method": "raw",
                "selection": "conservative",
                "risk_target": 0.1,
                "status": "selected",
                "threshold": 0.91,
            },
        ]
        primary = select_primary(
            selections,
            {"calibration": "temperature", "selection": "conservative", "risk_target": 0.1},
        )
        self.assertEqual(primary["threshold"], 0.93)

    def test_selected_policy_is_enriched_from_exact_sweep_row(self):
        selection = {
            "calibration_method": "temperature",
            "selection": "conservative",
            "risk_target": 0.1,
            "status": "selected",
            "threshold": 0.5,
        }
        sweep = [
            {
                "calibration_method": "temperature",
                "threshold": 0.5,
                "macro_f1": 0.8,
                "auroc": 0.9,
                "hard_partial_false_stop_rate": 0.1,
                "unnecessary_escalation_rate": 0.2,
                "final_supporting_fact_recall": 0.7,
                "final_complete_evidence_coverage": 0.6,
                "average_unique_titles": 8.0,
                "average_controller_calls": 1.5,
            }
        ]
        enriched = enrich_selection(selection, sweep)
        self.assertEqual(enriched["macro_f1"], 0.8)
        self.assertEqual(enriched["final_complete_evidence_coverage"], 0.6)


if __name__ == "__main__":
    unittest.main()

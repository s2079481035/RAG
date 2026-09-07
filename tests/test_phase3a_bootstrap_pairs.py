import unittest

from scripts.summarize_phase3a_bootstrap_pairs import (
    paired_comparison_rows,
    parse_comparisons,
)


class Phase3ABootstrapPairTests(unittest.TestCase):
    def test_parses_distinct_comparisons(self):
        self.assertEqual(
            parse_comparisons(["A,B", "B,C"]),
            [("A", "B"), ("B", "C")],
        )

    def test_paired_delta_uses_same_replicate_for_both_models(self):
        replicates = [
            {
                "models": {
                    "A": {"macro_f1": 0.4, "auroc": 0.6, "false_stop_rate": 0.7},
                    "B": {"macro_f1": 0.5, "auroc": 0.8, "false_stop_rate": 0.3},
                }
            },
            {
                "models": {
                    "A": {"macro_f1": 0.3, "auroc": 0.5, "false_stop_rate": 0.8},
                    "B": {"macro_f1": 0.6, "auroc": 0.7, "false_stop_rate": 0.4},
                }
            },
        ]
        observed = {
            "A": {"macro_f1": 0.35, "auroc": 0.55, "false_stop_rate": 0.75},
            "B": {"macro_f1": 0.55, "auroc": 0.75, "false_stop_rate": 0.35},
        }

        rows = paired_comparison_rows(replicates, observed, [("A", "B")])
        by_metric = {row["metric"]: row for row in rows}

        self.assertAlmostEqual(by_metric["macro_f1"]["observed_delta"], 0.2)
        self.assertAlmostEqual(by_metric["macro_f1"]["bootstrap_mean_delta"], 0.2)
        self.assertAlmostEqual(by_metric["false_stop_rate"]["bootstrap_mean_delta"], -0.4)
        self.assertTrue(by_metric["false_stop_rate"]["ci_excludes_zero"])


if __name__ == "__main__":
    unittest.main()

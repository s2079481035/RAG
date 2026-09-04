import unittest

from scripts.phase3a_metrics import (
    coverage_bucket,
    coverage_regression_metrics,
    metrics_from_decisions,
    sampling_group,
    sampling_weights,
    spearman_correlation,
)


class Phase3AMetricsTests(unittest.TestCase):
    def test_coverage_bucket_boundaries(self):
        self.assertEqual(coverage_bucket(0.0), "easy_continue")
        self.assertEqual(coverage_bucket(0.25), "medium_partial")
        self.assertEqual(coverage_bucket(0.5), "hard_partial")
        self.assertEqual(coverage_bucket(0.999), "hard_partial")
        self.assertEqual(coverage_bucket(1.0), "sufficient")

    def test_continue_only_metrics_have_no_auroc(self):
        metrics = metrics_from_decisions([0, 0], [0, 1], [0.1, 0.9])

        self.assertIsNone(metrics["auroc"])
        self.assertEqual(metrics["false_stop_rate"], 0.5)

    def test_spearman_and_coverage_mae(self):
        self.assertAlmostEqual(spearman_correlation([0.0, 0.5, 1.0], [0.1, 0.6, 0.9]), 1.0)
        metrics = coverage_regression_metrics([0.0, 0.5, 1.0], [0.1, 0.5, 0.8])
        self.assertAlmostEqual(metrics["mae"], 0.1)
        self.assertAlmostEqual(metrics["spearman"], 1.0)

    def test_balanced_sampling_assigns_equal_total_class_mass(self):
        items = [
            {"label": 1, "coverage_target": 1.0},
            {"label": 1, "coverage_target": 1.0},
            {"label": 1, "coverage_target": 1.0},
            {"label": 0, "coverage_target": 0.0},
        ]
        weights, audit = sampling_weights(
            items,
            "balanced_stop_continue",
            {"stop": 0.5, "continue": 0.5},
        )

        masses = {"stop": 0.0, "continue": 0.0}
        for item, weight in zip(items, weights):
            masses[sampling_group(item, "balanced_stop_continue")] += weight
        self.assertAlmostEqual(masses["stop"], 0.5)
        self.assertAlmostEqual(masses["continue"], 0.5)
        self.assertEqual(audit["num_samples"], len(items))

    def test_hard_partial_sampling_uses_preregistered_group_masses(self):
        items = [
            {"label": 1, "coverage_target": 1.0},
            {"label": 1, "coverage_target": 1.0},
            {"label": 0, "coverage_target": 0.0},
            {"label": 0, "coverage_target": 0.25},
            {"label": 0, "coverage_target": 0.5},
        ]
        targets = {
            "stop": 0.5,
            "continue_non_hard": 0.25,
            "continue_hard_partial": 0.25,
        }
        weights, _ = sampling_weights(items, "hard_partial_aware", targets)

        masses = {key: 0.0 for key in targets}
        for item, weight in zip(items, weights):
            masses[sampling_group(item, "hard_partial_aware")] += weight
        self.assertEqual(masses, targets)


if __name__ == "__main__":
    unittest.main()

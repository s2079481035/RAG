import unittest

import numpy as np

from scripts.phase3b_metrics import (
    binary_nll,
    deterministic_question_split,
    expected_calibration_error,
    fit_temperature,
    operational_metrics,
    rows_by_question,
    simulate_trajectory,
    temperature_scale,
    threshold_values,
)


def stage(question_id, name, label, probability, coverage, index):
    return {
        "question_id": question_id,
        "stage": name,
        "stage_index": index,
        "actual_stop_label": label,
        "actual_three_class_label": "sufficient" if label else "partial",
        "true_coverage": coverage,
        "raw_stop_probability": probability,
        "evidence_chunk_ids": [f"{question_id}-{name}"],
        "evidence_document_titles": [question_id],
    }


class Phase3BMetricsTests(unittest.TestCase):
    def test_question_split_is_deterministic_and_disjoint(self):
        ids = [f"q{index:03d}" for index in range(20)]
        first = deterministic_question_split(ids, 8, 42)
        second = deterministic_question_split(list(reversed(ids)), 8, 42)
        self.assertEqual(first, second)
        self.assertEqual(set(first[0]) & set(first[1]), set())
        self.assertEqual(set(first[0]) | set(first[1]), set(ids))

    def test_temperature_one_is_identity_and_preserves_order(self):
        probabilities = np.asarray([0.01, 0.2, 0.5, 0.8, 0.99])
        scaled = temperature_scale(probabilities, 1.0)
        np.testing.assert_allclose(scaled, probabilities)
        self.assertEqual(np.argsort(scaled).tolist(), np.argsort(probabilities).tolist())

    def test_temperature_fit_can_reduce_overconfident_nll(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.01, 0.99, 0.01, 0.99])
        before = binary_nll(labels, probabilities)
        temperature, after = fit_temperature(
            labels,
            probabilities,
            log_bounds=(-5.0, 5.0),
            iterations=100,
            epsilon=1e-7,
        )
        self.assertGreater(temperature, 1.0)
        self.assertLess(after, before)

    def test_ece_is_zero_for_matching_bin_frequency(self):
        labels = [0, 1, 0, 1]
        probabilities = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities, bins=10), 0.0)

    def test_threshold_grid_includes_both_ends(self):
        values = threshold_values(0.0, 1.0, 0.01)
        self.assertEqual(len(values), 101)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)

    def test_operational_metrics_exclude_unreached_second_stage(self):
        grouped = {
            "q1": [
                stage("q1", "dense@5", 1, 0.9, 1.0, 1),
                stage("q1", "hybrid@10", 0, 0.9, 0.5, 2),
                stage("q1", "rerank@20", 1, 0.9, 1.0, 3),
            ],
            "q2": [
                stage("q2", "dense@5", 0, 0.1, 0.5, 1),
                stage("q2", "hybrid@10", 1, 0.9, 1.0, 2),
                stage("q2", "rerank@20", 1, 0.9, 1.0, 3),
            ],
        }
        metrics = operational_metrics(grouped, "raw_stop_probability", 0.5)
        self.assertEqual(metrics["reached_actionable_decisions"], 3)
        self.assertEqual(metrics["false_stop_count"], 0)
        self.assertEqual(metrics["false_stop_rate"], 0.0)

    def test_trajectory_preserves_reached_evidence(self):
        rows = [
            stage("q1", "dense@5", 0, 0.1, 0.5, 1),
            stage("q1", "hybrid@10", 1, 0.8, 1.0, 2),
            stage("q1", "rerank@20", 1, 0.9, 1.0, 3),
        ]
        trajectory = simulate_trajectory(rows, "raw_stop_probability", 0.5)
        self.assertEqual(trajectory["final_stage"], "hybrid@10")
        self.assertEqual(len(trajectory["decisions"]), 2)
        self.assertEqual(
            trajectory["decisions"][0]["evidence_chunk_ids"], ["q1-dense@5"]
        )

    def test_rows_by_question_rejects_incomplete_trajectory(self):
        with self.assertRaises(ValueError):
            rows_by_question(
                [stage("q1", "dense@5", 0, 0.1, 0.0, 1)],
                ["dense@5", "hybrid@10", "rerank@20"],
            )


if __name__ == "__main__":
    unittest.main()

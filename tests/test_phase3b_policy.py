import unittest

import numpy as np

from scripts.select_phase3b_risk_policy import bootstrap_fsr, select_threshold


def row(question_id, stage, label, probability):
    return {
        "question_id": question_id,
        "stage": stage,
        "actual_stop_label": label,
        "raw_stop_probability": probability,
    }


class Phase3BPolicyTests(unittest.TestCase):
    def test_bootstrap_fsr_keeps_question_trajectories_and_reachability(self):
        grouped = {
            "q1": [
                row("q1", "dense@5", 1, 0.9),
                row("q1", "hybrid@10", 0, 0.9),
                row("q1", "rerank@20", 1, 0.9),
            ],
            "q2": [
                row("q2", "dense@5", 0, 0.1),
                row("q2", "hybrid@10", 1, 0.9),
                row("q2", "rerank@20", 1, 0.9),
            ],
        }
        values, draws = bootstrap_fsr(
            grouped,
            "raw_stop_probability",
            [0.5],
            replicates=20,
            seed=42,
        )
        self.assertEqual(draws.shape, (20, 2))
        finite = values[0.5][np.isfinite(values[0.5])]
        self.assertGreater(len(finite), 0)
        np.testing.assert_allclose(finite, 0.0)

    def test_threshold_selection_minimizes_registered_cost(self):
        rows = [
            {
                "threshold": 0.4,
                "false_stop_rate": 0.1,
                "fsr_ci95_low": 0.05,
                "fsr_ci95_high": 0.15,
                "average_retrieved_chunks": 12.0,
                "estimated_average_retrieval_latency_ms": 30.0,
                "average_reranker_calls": 0.4,
                "average_final_stage": 1.8,
            },
            {
                "threshold": 0.6,
                "false_stop_rate": 0.05,
                "fsr_ci95_low": 0.02,
                "fsr_ci95_high": 0.09,
                "average_retrieved_chunks": 16.0,
                "estimated_average_retrieval_latency_ms": 40.0,
                "average_reranker_calls": 0.6,
                "average_final_stage": 2.2,
            },
        ]
        point = select_threshold(rows, 0.1, "false_stop_rate")
        conservative = select_threshold(rows, 0.1, "fsr_ci95_high")
        self.assertEqual(point["threshold"], 0.4)
        self.assertEqual(conservative["threshold"], 0.6)

    def test_infeasible_threshold_is_recorded_not_fabricated(self):
        result = select_threshold(
            [
                {
                    "threshold": 1.0,
                    "false_stop_rate": 0.2,
                    "average_retrieved_chunks": 20.0,
                    "estimated_average_retrieval_latency_ms": 100.0,
                }
            ],
            0.05,
            "false_stop_rate",
        )
        self.assertEqual(result, {"status": "infeasible", "threshold": None})


if __name__ == "__main__":
    unittest.main()

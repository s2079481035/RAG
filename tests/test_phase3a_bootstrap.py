import unittest

from scripts.bootstrap_phase3a import group_by_question, sampled_metrics


class Phase3ABootstrapTests(unittest.TestCase):
    def test_question_resampling_keeps_all_stage_decisions(self):
        rows = [
            {"question_id": "q1", "stage": "s1", "actual_stop_label": 0, "predicted_stop_label": 0, "stop_probability": 0.1},
            {"question_id": "q1", "stage": "s2", "actual_stop_label": 1, "predicted_stop_label": 1, "stop_probability": 0.9},
            {"question_id": "q2", "stage": "s1", "actual_stop_label": 0, "predicted_stop_label": 1, "stop_probability": 0.8},
            {"question_id": "q2", "stage": "s2", "actual_stop_label": 1, "predicted_stop_label": 0, "stop_probability": 0.2},
        ]
        grouped = group_by_question(rows)

        metrics = sampled_metrics(grouped, ["q1", "q1"])

        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["false_stop_count"], 0)
        self.assertEqual(metrics["unnecessary_escalation_count"], 0)


if __name__ == "__main__":
    unittest.main()

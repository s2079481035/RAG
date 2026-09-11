import io
import unittest

from scripts.analyze_phase4_retrieval_ceiling import (
    analyze_split,
    controller_recoverability_metrics,
)


def row(qid, stage, sufficient, coverage, chunks, facts=2):
    evidence_state = "sufficient" if sufficient else "partial"
    return {
        "question_id": qid,
        "split": "dev_policy",
        "stage": stage,
        "gold_supporting_fact_count": facts,
        "cumulative_evidence_memory": {
            "stop_label": int(sufficient),
            "evidence_state": evidence_state,
            "supporting_fact_recall": coverage,
            "items": [{"chunk_id": f"{qid}-{index}"} for index in range(chunks)],
        },
        "latency_ms": {
            "query_embedding_run_mean": 2.0,
            "dense_search": 8.0,
            "bm25_scoring": 7.0,
            "rrf_fusion": 3.0,
            "reranker_allocated": 10.0,
        },
    }


def trajectory(qid, labels, facts=2):
    return [
        row(qid, "dense@5", labels[0], 1.0 if labels[0] else 0.5, 1, facts),
        row(qid, "hybrid@10", labels[1], 1.0 if labels[1] else 0.5, 2, facts),
        row(qid, "rerank@20", labels[2], 1.0 if labels[2] else 0.5, 3, facts),
    ]


class Phase4RetrievalCeilingTests(unittest.TestCase):
    def test_rescue_ceiling_oracle_and_fact_buckets(self):
        source = [
            trajectory("q1", [0, 1, 1], 2),
            trajectory("q2", [0, 0, 1], 3),
            trajectory("q3", [0, 0, 0], 4),
            trajectory("q4", [1, 1, 1], 2),
        ]
        diagnostic = io.StringIO()
        ceiling, rescue, buckets = analyze_split(source, "dev_policy", diagnostic)
        by_name = {row["analysis"]: row for row in rescue}

        self.assertEqual(ceiling["final_complete_evidence_coverage"], 0.75)
        self.assertEqual(ceiling["terminal_retrieval_failure_rate"], 0.25)
        self.assertEqual(ceiling["oracle_average_chunks"], 2.25)
        self.assertEqual(ceiling["oracle_reranker_calls"], 0.5)
        self.assertEqual(ceiling["oracle_average_retrieval_latency_ms"], 22.5)
        self.assertEqual(by_name["Dense_to_Hybrid_Rescue"]["rescued"], 1)
        self.assertEqual(by_name["Dense_to_Final_Rescue"]["rescued"], 2)
        self.assertEqual(by_name["Hybrid_to_Final_Rescue"]["rescued"], 1)
        self.assertEqual([row["questions"] for row in buckets], [2, 1, 1])

    def test_recoverable_fsr_uses_reached_actionable_stages_only(self):
        source = [
            trajectory("q1", [0, 1, 1]),
            trajectory("q2", [0, 0, 1]),
            trajectory("q3", [0, 0, 0]),
            trajectory("q4", [1, 1, 1]),
        ]
        diagnostic = io.StringIO()
        analyze_split(source, "dev_policy", diagnostic)

        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            path.write_text(diagnostic.getvalue(), encoding="utf-8")
            decisions = {
                "q1": [1, 0, 1],
                "q2": [0, 1, 1],
                "q3": [1, 1, 1],
                "q4": [1, 1, 1],
            }
            predictions = {}
            for qid, values in decisions.items():
                for stage, value in zip(
                    ["dense@5", "hybrid@10", "rerank@20"], values
                ):
                    predictions[(qid, stage)] = {
                        "question_id": qid,
                        "stage": stage,
                        "predicted_stop_label": value,
                        "stop_probability": float(value),
                    }
            metrics = controller_recoverability_metrics(
                path, predictions, "model", "dev_policy", "saved"
            )

        self.assertEqual(metrics["false_stop_rate"], 0.75)
        self.assertAlmostEqual(metrics["recoverable_false_stop_rate"], 2 / 3)
        self.assertEqual(metrics["unrecoverable_early_stop_rate"], 1.0)
        self.assertAlmostEqual(metrics["recoverable_share_of_false_stops"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

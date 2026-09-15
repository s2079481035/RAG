import sys
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_phase4_heldout import (  # noqa: E402
    bootstrap,
    decision_metrics,
    report_table,
    retrieval_tables,
    risk_interval_statement,
    select_threshold_policy,
)
from build_phase2_controller_data import resolve_controller_ladder  # noqa: E402
from generate_phase4_heldout_stage_answers import validated_resume_prefix  # noqa: E402


def source_trajectory(question_id: str, labels: list[int], coverages: list[float]):
    rows = []
    for index, (stage, label, coverage) in enumerate(
        zip(["dense@5", "hybrid@10", "rerank@20"], labels, coverages), start=1
    ):
        rows.append(
            {
                "question_id": question_id,
                "stage": stage,
                "stage_index": index,
                "actual_stop_label": label,
                "actual_three_class_label": "sufficient" if label else "partial",
                "true_coverage": coverage,
                "raw_stop_probability": 0.9 if index == 1 else 0.1,
                "temperature_stop_probability": 0.9 if index == 1 else 0.1,
                "retrieved_chunks": [5, 12, 20][index - 1],
                "unique_titles": [5, 10, 18][index - 1],
                "reranker_calls": int(index == 3),
                "retrieval_latency_ms": [10.0, 20.0, 40.0][index - 1],
                "question_type": "compositional",
                "gold_supporting_fact_count": 2,
            }
        )
    return rows


def outcome(question_id: str, f1: float, chunks: int, false_stop: int = 0):
    return {
        "question_id": question_id,
        "answer_f1": f1,
        "answer_exact_match": int(f1 == 1.0),
        "retrieved_chunks": chunks,
        "reranker_calls": int(chunks == 20),
        "total_latency_ms": float(chunks),
        "complete_evidence_coverage": int(chunks == 20),
        "decisions": [
            {
                "actual_stop_label": 0,
                "predicted_stop_label": false_stop,
            }
        ],
    }


class Phase4HeldoutTests(unittest.TestCase):
    def test_risk_interval_relation_distinguishes_crossing_and_above(self):
        self.assertIn("wholly below", risk_interval_statement(0.05, 0.09, 0.1))
        self.assertIn("crosses", risk_interval_statement(0.08, 0.12, 0.1))
        self.assertIn("wholly above", risk_interval_statement(0.11, 0.13, 0.1))

    def test_report_table_supports_multiple_text_columns(self):
        table = report_table(
            [{"dataset": "2Wiki", "system": "FixedHeavy", "f1": 0.5}],
            [("dataset", "Dataset"), ("system", "System"), ("f1", "F1")],
        )
        self.assertEqual(table[-1], "| 2Wiki | FixedHeavy | 0.5000 |")

    def test_generation_resume_keeps_only_valid_frozen_prefix(self):
        records = [
            {"question_id": "q1", "stage": "dense@5", "stage_index": 0},
            {"question_id": "q1", "stage": "hybrid@10", "stage_index": 1},
            {"question_id": "q1", "stage": "rerank@20", "stage_index": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_generations.jsonl.tmp"
            with path.open("wb") as handle:
                for record in records[:2]:
                    row = {
                        "question_id": record["question_id"],
                        "split": "heldout",
                        "stage": record["stage"],
                        "stage_index": record["stage_index"] + 1,
                    }
                    handle.write((json.dumps(row) + "\n").encode())
                handle.write(b'{"question_id":"incomplete"')

            self.assertEqual(validated_resume_prefix(path, records, "heldout"), 2)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_generation_resume_rejects_nonmatching_prefix(self):
        records = [{"question_id": "q1", "stage": "dense@5", "stage_index": 0}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_generations.jsonl.tmp"
            path.write_text(
                json.dumps(
                    {
                        "question_id": "wrong",
                        "split": "heldout",
                        "stage": "dense@5",
                        "stage_index": 1,
                    }
                )
                + "\n"
            )
            with self.assertRaises(ValueError):
                validated_resume_prefix(path, records, "heldout")

    def test_phase4_controller_ladder_falls_back_to_protocol(self):
        protocol = {"controller_ladder": ["dense@5", "hybrid@10", "rerank@20"]}
        self.assertEqual(resolve_controller_ladder({}, protocol), protocol["controller_ladder"])

    def test_controller_ladder_prefers_controller_config(self):
        controller = {"controller_ladder": ["dense@5"]}
        protocol = {"controller_ladder": ["hybrid@10"]}
        self.assertEqual(resolve_controller_ladder(controller, protocol), ["dense@5"])

    def test_terminal_stage_is_not_a_controller_decision(self):
        rows = source_trajectory("q1", [0, 0, 0], [0.5, 0.5, 0.5])
        trajectory = select_threshold_policy(rows, "raw_stop_probability", 0.5, "x")
        self.assertEqual(len(trajectory["decisions"]), 1)
        self.assertEqual(trajectory["final_stage"], "dense@5")
        metrics = decision_metrics([trajectory], probability_available=True)
        self.assertEqual(metrics["false_stop_rate"], 1.0)
        self.assertEqual(metrics["unrecoverable_early_stop_rate"], 1.0)

    def test_recoverable_false_stop_uses_later_sufficiency(self):
        rows = source_trajectory("q1", [0, 0, 1], [0.5, 0.5, 1.0])
        trajectory = select_threshold_policy(rows, "raw_stop_probability", 0.5, "x")
        self.assertEqual(trajectory["false_stop_recoverable"], 1)
        metrics = decision_metrics([trajectory], probability_available=True)
        self.assertEqual(metrics["recoverable_false_stop_rate"], 1.0)
        self.assertIsNone(metrics["unrecoverable_early_stop_rate"])

    def test_retrieval_ceiling_is_exact_complement(self):
        source = {
            "q1": source_trajectory("q1", [0, 0, 1], [0.5, 0.5, 1.0]),
            "q2": source_trajectory("q2", [0, 0, 0], [0.0, 0.5, 0.5]),
        }
        _, rescue, ceiling = retrieval_tables(source)
        self.assertEqual(ceiling["final_complete_evidence_coverage"], 0.5)
        self.assertEqual(ceiling["terminal_retrieval_failure_rate"], 0.5)
        self.assertEqual(rescue[1]["rescued"], 1)

    def test_bootstrap_delta_is_model_minus_baseline(self):
        baseline = [outcome("q1", 0.0, 20), outcome("q2", 0.0, 20)]
        model = [outcome("q1", 1.0, 10), outcome("q2", 1.0, 10)]
        summary, replicates = bootstrap(
            {"baseline": baseline, "model": model},
            [("baseline", "model")],
            replicates=3,
            seed=42,
        )
        f1 = next(row for row in summary if row["metric"] == "answer_f1")
        chunks = next(row for row in summary if row["metric"] == "chunks")
        self.assertEqual(f1["observed_delta"], 1.0)
        self.assertEqual(chunks["observed_delta"], -10.0)
        self.assertEqual(len(replicates), 3 * 6)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from phase4_2wiki import (  # noqa: E402
    answer_type,
    deterministic_train_split,
    require_official_record,
    source_statistics,
    split_digest,
)
from phase4_score_threshold import (  # noqa: E402
    attach_normalized_scores,
    fit_stage_minmax,
    minmax_score,
)
from evaluate_phase4_score_threshold import score_threshold_values  # noqa: E402
from phase4_adaptive_router import (  # noqa: E402
    route_target,
    router_record,
    summarize_routes,
)
from run_phase2_retrieval import select_questions_for_run  # noqa: E402


def record(qid="q1"):
    return {
        "_id": qid,
        "question": "Who was born first?",
        "answer": "Alice",
        "supporting_facts": [["Alice", 0], ["Bob", 1]],
        "context": [
            ["Alice", ["Alice was born in 1900."]],
            ["Bob", ["Bob was a person.", "Bob was born in 1910."]],
        ],
        "type": "comparison",
        "evidences": [["Alice", "birth year", "1900"]],
    }


def router_stage(qid, stage, coverage, stop, chunks, latency):
    return {
        "question_id": qid,
        "split": "dev_policy",
        "question": "Which route is needed?",
        "stage": stage,
        "cumulative_evidence_memory": {
            "supporting_fact_recall": coverage,
            "stop_label": stop,
            "items": [
                {"chunk_id": f"{qid}-{index}"} for index in range(chunks)
            ],
            "document_titles": [f"title-{index}" for index in range(chunks)],
        },
        "latency_ms": {
            "query_embedding_run_mean": 1.0,
            "dense_search": 2.0,
            "bm25_scoring": 3.0,
            "rrf_fusion": 4.0,
            "reranker_allocated": latency - 10.0,
        },
    }


def router_trajectory(qid, complete_stage=None):
    stops = [False, False, False]
    if complete_stage is not None:
        for index in range(complete_stage, 3):
            stops[index] = True
    coverage = [0.0, 0.5, 1.0] if complete_stage is not None else [0.0, 0.5, 0.5]
    if complete_stage == 0:
        coverage = [1.0, 1.0, 1.0]
    elif complete_stage == 1:
        coverage = [0.5, 1.0, 1.0]
    return [
        router_stage(qid, "dense@5", coverage[0], stops[0], 5, 3.0),
        router_stage(qid, "hybrid@10", coverage[1], stops[1], 10, 10.0),
        router_stage(qid, "rerank@20", coverage[2], stops[2], 20, 15.0),
    ]


class Phase42WikiTests(unittest.TestCase):
    def test_requires_official_pair_formats(self):
        parsed = require_official_record(record(), "train")
        self.assertEqual(parsed["gold_supporting_facts"][1], {"title": "Bob", "sentence_id": 1})
        broken = record()
        broken["context"] = {"title": ["Alice"], "sentences": [["text"]]}
        with self.assertRaisesRegex(ValueError, "no context documents"):
            require_official_record(broken, "train")

    def test_question_split_is_deterministic_disjoint_and_exhaustive(self):
        ids = [f"q{i}" for i in range(20)]
        first = deterministic_train_split(
            ids, seed=42, dev_calibration_questions=4, dev_policy_questions=5
        )
        second = deterministic_train_split(
            reversed(ids), seed=42, dev_calibration_questions=4, dev_policy_questions=5
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(value == "dev_calibration" for value in first.values()), 4)
        self.assertEqual(sum(value == "dev_policy" for value in first.values()), 5)
        self.assertEqual(sum(value == "train_core" for value in first.values()), 11)

    def test_split_digest_is_order_independent(self):
        left = {"train_core": ["b", "a"], "heldout": ["c"]}
        right = {"heldout": ["c"], "train_core": ["a", "b"]}
        self.assertEqual(split_digest(left), split_digest(right))

    def test_answer_types_are_declared_heuristics(self):
        self.assertEqual(answer_type("yes"), "yes_no")
        self.assertEqual(answer_type("1999"), "year")
        self.assertEqual(answer_type("12.5%"), "number")
        self.assertEqual(answer_type("June 1, 1956"), "date")
        self.assertEqual(answer_type("Gregg Harper"), "span")

    def test_source_statistics_reports_duplicates(self):
        first = require_official_record(record("q1"), "train")
        second = require_official_record(record("q2"), "train")
        stats = source_statistics([first, second])
        self.assertEqual(stats["questions"], 2)
        self.assertEqual(stats["raw_context_documents"], 4)
        self.assertEqual(stats["unique_titles"], 2)
        self.assertEqual(stats["duplicate_text_instances_beyond_first"], 2)

    def test_score_normalization_is_fit_per_stage_and_clipped(self):
        def stage_row(qid, stage, method, score):
            return {
                "question_id": qid,
                "stage": stage,
                "retrieval_statistics": {
                    "current_method": method,
                    method: {"max": score},
                },
            }

        rows = [
            stage_row("q1", "dense@5", "dense", 0.2),
            stage_row("q2", "dense@5", "dense", 0.6),
            stage_row("q1", "hybrid@10", "hybrid", 0.01),
            stage_row("q2", "hybrid@10", "hybrid", 0.03),
        ]
        parameters = fit_stage_minmax(rows, ["dense@5", "hybrid@10"])
        self.assertAlmostEqual(minmax_score(0.4, parameters["dense@5"]), 0.5)
        heldout = [stage_row("q3", "dense@5", "dense", 0.8)]
        scored = attach_normalized_scores(heldout, parameters)
        self.assertEqual(scored[0]["normalized_score"], 1.0)

    def test_score_threshold_grid_has_always_final_endpoint(self):
        thresholds = score_threshold_values({"start": 0.0, "stop": 1.0, "step": 0.01})
        self.assertEqual(thresholds[-2], 1.0)
        self.assertGreater(thresholds[-1], 1.0)

    def test_retrieval_smoke_limit_is_deterministic_per_split(self):
        questions = {
            "z": {"question_id": "z", "split": "train_core"},
            "b": {"question_id": "b", "split": "dev_policy"},
            "a": {"question_id": "a", "split": "train_core"},
            "c": {"question_id": "c", "split": "dev_policy"},
        }
        selected, eligible = select_questions_for_run(
            questions, ["train_core", "dev_policy"], 1
        )
        self.assertEqual([row["question_id"] for row in selected], ["a", "b"])
        self.assertEqual(eligible, {"train_core": 2, "dev_policy": 2})

    def test_adaptive_router_target_is_earliest_complete_else_heavy(self):
        medium = router_trajectory("q1", complete_stage=1)
        unresolved = router_trajectory("q2")
        self.assertEqual(route_target(medium)["route"], "medium")
        self.assertTrue(route_target(medium)["recoverable"])
        self.assertEqual(route_target(unresolved)["route"], "heavy")
        self.assertFalse(route_target(unresolved)["recoverable"])
        self.assertEqual(router_record(medium, "dev_policy")["label"], 1)

    def test_adaptive_router_under_retrieval_excludes_unrecoverable(self):
        trajectories = {
            "q1": router_trajectory("q1", complete_stage=1),
            "q2": router_trajectory("q2"),
        }
        metrics = summarize_routes(trajectories, {"q1": 0, "q2": 0})
        self.assertEqual(metrics["recoverable_questions"], 1)
        self.assertEqual(metrics["unrecoverable_questions"], 1)
        self.assertEqual(metrics["recoverable_under_retrieval_rate"], 1.0)
        self.assertEqual(metrics["terminal_retrieval_failure_rate"], 0.5)
        self.assertEqual(metrics["light_route_rate"], 1.0)

    def test_adaptive_router_protocol_requires_dev_gate(self):
        protocol = json.loads(
            (ROOT / "configs" / "phase4" / "protocol.json").read_text(
                encoding="utf-8"
            )
        )
        adaptive = protocol["external_baselines"]["adaptive_rag"]
        self.assertEqual(adaptive["execution_status"], "required_before_heldout")
        self.assertEqual(adaptive["router_input"], "question_only")
        self.assertFalse(adaptive["strict_official_reproduction"])


if __name__ == "__main__":
    unittest.main()

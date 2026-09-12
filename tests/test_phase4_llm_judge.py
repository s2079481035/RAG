import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_phase4_llm_judge_prompt import validate_prompt_check  # noqa: E402
from analyze_phase4_llm_judge import hard_sequential_metrics  # noqa: E402
from run_phase4_llm_judge import (  # noqa: E402
    ALLOWED_SPLITS,
    fit_full_evidence,
    parse_label,
    render_judge_prompt,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"] + ("\nA:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [ord(character) for character in text]}


class Phase4LLMJudgeTests(unittest.TestCase):
    def test_only_train_derived_dev_splits_are_callable(self):
        self.assertEqual(ALLOWED_SPLITS, {"dev_calibration", "dev_policy"})
        self.assertNotIn("heldout", ALLOWED_SPLITS)

    def test_strict_label_parser_rejects_explanations(self):
        labels = ["SUFFICIENT", "INSUFFICIENT"]
        self.assertEqual(parse_label(" sufficient\n", labels), "SUFFICIENT")
        self.assertEqual(parse_label("INSUFFICIENT", labels), "INSUFFICIENT")
        self.assertIsNone(parse_label("SUFFICIENT because evidence is complete", labels))
        self.assertIsNone(parse_label("yes", labels))

    def test_full_evidence_never_exceeds_prompt_budget(self):
        tokenizer = CharacterTokenizer()
        template = "Question:\n{question}\nEvidence:\n{packed_evidence}"
        chunks = {
            "c1": {"document_title": "A", "chunk_text": "x" * 80},
            "c2": {"document_title": "B", "chunk_text": "y" * 80},
        }
        items = [
            {"chunk_id": "c1", "document_title": "A"},
            {"chunk_id": "c2", "document_title": "B"},
        ]
        evidence, audit = fit_full_evidence(
            tokenizer, template, "Q?", items, chunks, max_input_tokens=120
        )
        prompt = render_judge_prompt(template, "Q?", evidence)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertLessEqual(len(tokenizer(rendered)["input_ids"]), 120)
        self.assertTrue(audit["truncated"])

    def test_prompt_gate_uses_parseability_not_accuracy(self):
        judge = {
            "prompt_check_questions": 100,
            "actionable_stages": ["dense@5", "hybrid@10"],
            "minimum_parse_rate": 0.99,
        }
        manifest = {
            "split": "dev_calibration",
            "heldout_consulted": False,
            "purpose": "prompt_check",
            "representation": "judge_512",
            "questions": 100,
            "actionable_states": 200,
            "actionable_stages": judge["actionable_stages"],
            "llm_judge_config_sha256": "placeholder",
            "strict_parse_rate": 0.995,
            "invalid_outputs": 1,
        }
        import freeze_phase4_llm_judge_prompt as freezer

        original = freezer.object_sha256
        freezer.object_sha256 = lambda value: "placeholder"
        try:
            result = validate_prompt_check(manifest, "judge_512", judge)
        finally:
            freezer.object_sha256 = original
        self.assertEqual(result["strict_parse_rate"], 0.995)
        self.assertNotIn("accuracy", result)

    def test_phase4_config_forbids_judge_probability_claim(self):
        config = json.loads((ROOT / "configs" / "phase4" / "protocol.json").read_text())
        judge = config["llm_judge"]
        self.assertFalse(judge["label_probability_available"])
        self.assertFalse(judge["auroc_reported"])
        self.assertEqual(judge["actionable_stages"], config["controller_ladder"][:-1])
        frozen = config["in_domain_policy"]["frozen_dev_gate"]
        self.assertAlmostEqual(
            frozen["classification_operating_point"]["threshold"], 0.79
        )
        self.assertEqual(frozen["risk_controlled_operating_point"]["threshold"], 0.01)

    def test_prompt_text_is_the_frozen_format_revision(self):
        expected = """You are given a question and the evidence currently retrieved by a retrieval-augmented generation system.

Determine whether the evidence is already sufficient to answer the question correctly.

Your entire response must be exactly one label from this list:

SUFFICIENT
INSUFFICIENT

Do not output punctuation, reasoning, an explanation, or a second line.

Question:
{question}

Evidence:
{packed_evidence}
"""
        actual = (ROOT / "configs" / "phase4" / "llm_judge_prompt.txt").read_text()
        self.assertEqual(actual, expected)

        config = json.loads((ROOT / "configs" / "phase4" / "protocol.json").read_text())
        self.assertEqual(config["llm_judge"]["prompt_version"], 2)
        self.assertEqual(config["llm_judge"]["prompt_debugging_rounds"], 1)

    def test_sequential_metrics_skip_unreached_hybrid_state(self):
        rows = [
            {
                "question_id": "q1",
                "stage": "dense@5",
                "actual_stop_label": 1,
                "predicted_stop_label": 1,
                "true_coverage": 1.0,
            },
            {
                "question_id": "q1",
                "stage": "hybrid@10",
                "actual_stop_label": 1,
                "predicted_stop_label": 0,
                "true_coverage": 1.0,
            },
            {
                "question_id": "q2",
                "stage": "dense@5",
                "actual_stop_label": 0,
                "predicted_stop_label": 0,
                "true_coverage": 0.5,
            },
            {
                "question_id": "q2",
                "stage": "hybrid@10",
                "actual_stop_label": 1,
                "predicted_stop_label": 1,
                "true_coverage": 1.0,
            },
        ]
        metrics, reached = hard_sequential_metrics(
            rows, ["dense@5", "hybrid@10", "rerank@20"]
        )
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual([(r["question_id"], r["stage"]) for r in reached], [
            ("q1", "dense@5"),
            ("q2", "dense@5"),
            ("q2", "hybrid@10"),
        ])


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.phase2_controller_inputs import prepare_nonhierarchical_input
from tests.test_phase2_packing import WhitespaceTokenizer


class Phase2ControllerInputTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WhitespaceTokenizer()
        self.chunk = {
            "chunk_id": "c1",
            "document_title": "Entity",
            "sentence_ids": [0],
            "sentence_texts": ["A supporting sentence."],
            "chunk_text": "A supporting sentence.",
        }
        evidence = {
            "chunk_ids": ["c1"],
            "items": [
                {
                    "chunk_id": "c1",
                    "document_title": "Entity",
                    "retrieval_score": 0.8,
                    "retrieval_rank": 1,
                    "score_source": "dense",
                }
            ],
            "stop_label": 1,
            "evidence_state": "sufficient",
        }
        self.record = {
            "question_id": "q1",
            "question": "Who is Entity?",
            "answer": "A person",
            "stage": "dense@5",
            "retrieval_method": "dense",
            "k": 5,
            "gold_supporting_facts": [{"title": "Entity", "sentence_id": 0}],
            "gold_supporting_fact_count": 1,
            "retrieval_statistics": {
                "dense": {
                    "max": 0.8,
                    "mean": 0.5,
                    "std": 0.2,
                    "top1_top2_margin": 0.1,
                    "softmax_entropy": 0.9,
                }
            },
            "raw_stage_evidence": evidence,
            "cumulative_evidence_memory": evidence,
        }

    def prepare(self, baseline):
        return prepare_nonhierarchical_input(
            self.record,
            baseline=baseline,
            representation="uniform_packing",
            evidence_mode="cumulative",
            tokenizer=self.tokenizer,
            chunk_by_id={"c1": self.chunk},
            max_length=64,
        )

    def test_query_stage_has_metadata_but_no_evidence(self):
        item = self.prepare("query_stage")
        self.assertIn("[STAGE] dense@5", item["text_a"])
        self.assertNotIn("supporting sentence", item["text_a"])
        self.assertIsNone(item["text_b"])

    def test_query_stage_stats_uses_only_inference_statistics(self):
        item = self.prepare("query_stage_stats")
        self.assertIn("dense_max=0.8", item["text_a"])
        self.assertIsNone(item["text_b"])

    def test_query_evidence_is_a_pair_and_keeps_stop_label(self):
        item = self.prepare("query_evidence")
        self.assertIn("[STAGE] dense@5", item["text_a"])
        self.assertIn("Entity", item["text_b"])
        self.assertEqual(item["label"], 1)


if __name__ == "__main__":
    unittest.main()

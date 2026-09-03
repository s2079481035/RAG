import unittest

from scripts.phase2_packing import allocate_tokens, gold_fact_bucket, length_bucket, pack_evidence


class WhitespaceTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        output = []
        for token in text.split():
            if token not in self.vocab:
                index = len(self.vocab) + 1
                self.vocab[token] = index
                self.reverse[index] = token
            output.append(self.vocab[token])
        return output

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(self.reverse[index] for index in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2


class Phase2PackingTests(unittest.TestCase):
    def test_uniform_packing_does_not_let_first_chunk_consume_budget(self):
        chunks = [
            {"token_ids": list(range(100)), "header_tokens": 2},
            {"token_ids": list(range(100)), "header_tokens": 2},
        ]
        allocations = allocate_tokens(chunks, 40, "uniform_packing", 5)
        self.assertEqual(allocations, [20, 20])

    def test_concat_truncate_is_prefix_only(self):
        chunks = [
            {"token_ids": list(range(30)), "header_tokens": 2},
            {"token_ids": list(range(30)), "header_tokens": 2},
        ]
        self.assertEqual(allocate_tokens(chunks, 40, "concat_truncate", 5), [30, 10])

    def test_uniform_packing_round_robins_when_headers_exceed_budget(self):
        chunks = [
            {"token_ids": list(range(20)), "header_tokens": 10},
            {"token_ids": list(range(20)), "header_tokens": 10},
            {"token_ids": list(range(20)), "header_tokens": 10},
        ]
        allocations = allocate_tokens(chunks, 8, "uniform_packing", 12)
        self.assertEqual(sum(allocations), 8)
        self.assertLessEqual(max(allocations) - min(allocations), 1)

    def test_pack_audit_tracks_fully_visible_sentences(self):
        tokenizer = WhitespaceTokenizer()
        chunk = {
            "chunk_id": "c1",
            "document_title": "Short title",
            "sentence_ids": [0, 1],
            "sentence_texts": ["first fact", "second fact"],
        }
        _, audit = pack_evidence(
            tokenizer,
            [{"chunk_id": "c1", "document_title": "Short title", "retrieval_rank": 1}],
            {"c1": chunk},
            question="which fact",
            max_length=30,
            strategy="uniform_packing",
        )
        self.assertFalse(audit["truncated"])
        self.assertEqual(
            audit["allocations"][0]["fully_visible_sentence_ids"], [0, 1]
        )

    def test_analysis_buckets(self):
        self.assertEqual(length_bucket(513), "513-1024")
        self.assertEqual(gold_fact_bucket(4), "4+")


if __name__ == "__main__":
    unittest.main()

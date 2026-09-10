import unittest

from scripts.select_phase4_retrieval_execution import choose_execution


def manifest(workers=None, batch=None, seconds=10.0):
    row = {
        "run_scope": "smoke_throughput_only",
        "question_counts": {"dev_policy": 100},
        "query_embedding_seconds": 1.0,
        "first_stage_seconds": seconds,
        "reranking_seconds": 2.0,
    }
    if workers is not None:
        row["bm25_workers"] = workers
    if batch is not None:
        row["dense_search_batch_size"] = batch
    return row


class Phase4RetrievalExecutionTests(unittest.TestCase):
    def test_accepts_parallel_bm25_and_rejects_failed_batched_faiss(self):
        decision = choose_execution(
            manifest(seconds=50.0),
            manifest(workers=8, batch=1, seconds=10.0),
            manifest(workers=8, batch=512, seconds=8.0),
            {"matches": True},
            {"matches": False},
        )
        self.assertEqual(decision["selected"], {"bm25_workers": 8, "dense_search_batch_size": 1})
        self.assertEqual(decision["first_stage_speedup"], 5.0)

    def test_rejects_parallel_candidate_without_parity(self):
        with self.assertRaisesRegex(ValueError, "failed parity"):
            choose_execution(
                manifest(seconds=50.0),
                manifest(workers=8, batch=1, seconds=10.0),
                manifest(workers=8, batch=512, seconds=8.0),
                {"matches": False},
                {"matches": False},
            )


if __name__ == "__main__":
    unittest.main()

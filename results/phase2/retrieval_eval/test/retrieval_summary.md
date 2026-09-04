# Phase 2 Chunk Retrieval Comparison

Evaluation split: `test`. Evidence matches require exact `(document_title, sentence_id)` equality.

| Variant | Method | K | SF Recall | Complete coverage | Any coverage | Gold-title recall | Avg docs | Avg latency ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sentence_256 | dense | 5 | 0.8959 | 0.8000 | 0.9910 | 0.8960 | 4.98 | 5.43 |
| sentence_256 | dense | 10 | 0.9238 | 0.8520 | 0.9960 | 0.9245 | 9.97 | 5.43 |
| sentence_256 | dense | 20 | 0.9458 | 0.8920 | 0.9990 | 0.9460 | 19.95 | 5.43 |
| sentence_256 | bm25 | 5 | 0.8503 | 0.7130 | 0.9860 | 0.8500 | 4.99 | 30.86 |
| sentence_256 | bm25 | 10 | 0.8796 | 0.7670 | 0.9910 | 0.8795 | 9.98 | 30.86 |
| sentence_256 | bm25 | 20 | 0.9047 | 0.8130 | 0.9960 | 0.9050 | 19.97 | 30.86 |
| sentence_256 | hybrid | 5 | 0.9090 | 0.8190 | 0.9970 | 0.9090 | 4.99 | 37.53 |
| sentence_256 | hybrid | 10 | 0.9384 | 0.8770 | 0.9980 | 0.9380 | 9.98 | 37.53 |
| sentence_256 | hybrid | 20 | 0.9564 | 0.9120 | 1.0000 | 0.9565 | 19.96 | 37.53 |
| sentence_256 | rerank | 5 | 0.9352 | 0.8710 | 0.9980 | 0.9350 | 4.99 | 175.55 |
| sentence_256 | rerank | 10 | 0.9511 | 0.9010 | 1.0000 | 0.9510 | 9.98 | 175.55 |
| sentence_256 | rerank | 20 | 0.9564 | 0.9120 | 1.0000 | 0.9565 | 19.96 | 175.55 |

Latency is measured during this retrieval run. Batched query-embedding and reranker time is allocated equally within the corresponding batch; it is not a single-query online benchmark.

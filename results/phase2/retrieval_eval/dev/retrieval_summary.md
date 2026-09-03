# Phase 2 Chunk Retrieval Comparison

Evaluation split: `dev`. Evidence matches require exact `(document_title, sentence_id)` equality.

| Variant | Method | K | SF Recall | Complete coverage | Any coverage | Gold-title recall | Avg docs | Avg latency ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sentence_128 | dense | 5 | 0.8932 | 0.7900 | 0.9890 | 0.8995 | 4.64 | 6.22 |
| sentence_128 | dense | 10 | 0.9265 | 0.8540 | 0.9930 | 0.9310 | 9.38 | 6.22 |
| sentence_128 | dense | 20 | 0.9473 | 0.8960 | 0.9940 | 0.9505 | 18.83 | 6.22 |
| sentence_128 | bm25 | 5 | 0.8347 | 0.6720 | 0.9890 | 0.8450 | 4.76 | 35.79 |
| sentence_128 | bm25 | 10 | 0.8798 | 0.7590 | 0.9940 | 0.8890 | 9.63 | 35.79 |
| sentence_128 | bm25 | 20 | 0.9052 | 0.8100 | 0.9950 | 0.9120 | 19.41 | 35.79 |
| sentence_128 | hybrid | 5 | 0.9092 | 0.8150 | 0.9970 | 0.9140 | 4.70 | 43.33 |
| sentence_128 | hybrid | 10 | 0.9385 | 0.8730 | 0.9980 | 0.9425 | 9.51 | 43.33 |
| sentence_128 | hybrid | 20 | 0.9573 | 0.9130 | 0.9980 | 0.9605 | 19.13 | 43.33 |
| sentence_128 | rerank | 5 | 0.9403 | 0.8790 | 0.9950 | 0.9430 | 4.71 | 153.75 |
| sentence_128 | rerank | 10 | 0.9528 | 0.9050 | 0.9970 | 0.9555 | 9.56 | 153.75 |
| sentence_128 | rerank | 20 | 0.9573 | 0.9130 | 0.9980 | 0.9605 | 19.13 | 153.75 |
| sentence_256 | dense | 5 | 0.9071 | 0.8190 | 0.9920 | 0.9055 | 4.99 | 5.54 |
| sentence_256 | dense | 10 | 0.9330 | 0.8700 | 0.9930 | 0.9315 | 9.98 | 5.54 |
| sentence_256 | dense | 20 | 0.9505 | 0.9040 | 0.9950 | 0.9495 | 19.96 | 5.54 |
| sentence_256 | bm25 | 5 | 0.8516 | 0.7120 | 0.9890 | 0.8505 | 4.99 | 33.88 |
| sentence_256 | bm25 | 10 | 0.8908 | 0.7850 | 0.9930 | 0.8890 | 9.99 | 33.88 |
| sentence_256 | bm25 | 20 | 0.9140 | 0.8300 | 0.9970 | 0.9135 | 19.98 | 33.88 |
| sentence_256 | hybrid | 5 | 0.9165 | 0.8340 | 0.9970 | 0.9155 | 4.99 | 40.76 |
| sentence_256 | hybrid | 10 | 0.9434 | 0.8870 | 0.9980 | 0.9425 | 9.99 | 40.76 |
| sentence_256 | hybrid | 20 | 0.9619 | 0.9250 | 0.9980 | 0.9615 | 19.97 | 40.76 |
| sentence_256 | rerank | 5 | 0.9452 | 0.8940 | 0.9950 | 0.9445 | 5.00 | 169.33 |
| sentence_256 | rerank | 10 | 0.9581 | 0.9180 | 0.9970 | 0.9575 | 9.99 | 169.33 |
| sentence_256 | rerank | 20 | 0.9619 | 0.9250 | 0.9980 | 0.9615 | 19.97 | 169.33 |

Latency is measured during this retrieval run. Batched query-embedding and reranker time is allocated equally within the corresponding batch; it is not a single-query online benchmark.

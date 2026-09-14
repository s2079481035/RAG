# 2Wiki Adaptive-RAG-style Router Dev Analysis

This is an adapted, question-only, pre-retrieval Router and not an official Adaptive-RAG reproduction. Training uses `train_core`, epoch selection uses `dev_calibration`, and this report evaluates `dev_policy` once. Heldout is not read.

## Three-seed Router

| Metric | Mean | Std |
|---|---:|---:|
| accuracy | 0.942333 | 0.002466 |
| macro_f1 | 0.631643 | 0.001822 |
| light_route_rate | 0.338333 | 0.002843 |
| medium_route_rate | 0.000000 | 0.000000 |
| heavy_route_rate | 0.661667 | 0.002843 |
| average_final_stage | 2.323333 | 0.005686 |
| average_retrieved_chunks | 15.069000 | 0.042875 |
| average_unique_titles | 12.500333 | 0.039119 |
| average_reranker_calls | 0.661667 | 0.002843 |
| average_retrieval_latency_ms | 1004.797403 | 2.580965 |
| supporting_fact_recall | 0.676300 | 0.000250 |
| complete_evidence_coverage | 0.340000 | 0.000500 |
| terminal_retrieval_failure_rate | 0.634000 | 0.000000 |
| recoverable_route_accuracy | 0.864299 | 0.006160 |
| recoverable_under_retrieval_rate | 0.071038 | 0.001366 |
| unrecoverable_average_chunks | 20.027340 | 0.031450 |
| unrecoverable_average_reranker_calls | 0.987382 | 0.002087 |
| unrecoverable_average_retrieval_latency_ms | 1482.183466 | 2.313631 |

## Frozen Quality-Cost Comparison

| System | Final depth | Chunks | Rerank calls | SF recall | Complete coverage | Retrieval latency ms | Recoverable under-retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|
| FixedHeavy | 3.000 | 20.254 | 1.000 | 0.6893 | 0.3660 | 1467.91 | 0.00% |
| Adaptive-RAG-style Query-Complexity Router | 2.323 | 15.069 | 0.662 | 0.6763 | 0.3400 | 1004.80 | 7.10% |
| EvidenceAwareClassification | 2.328 | 15.060 | 0.649 | 0.6885 | 0.3645 | 1027.33 | 0.41% |
| EvidenceAwareRisk10 | 2.238 | 14.338 | 0.589 | 0.6803 | 0.3515 | 986.61 | 3.96% |

## Retrieval-limited Diagnostics

- Frozen terminal retrieval failure rate: 63.40%.
- The Dev target contains 57 Medium questions, but the three Router seeds produced 0 Medium predictions in total.
- Router route accuracy on recoverable questions: 86.43%.
- Router recoverable under-retrieval rate: 7.10%.
- Operational Router inference latency is 12.80 ms/question at synchronized batch size one; Router plus retrieval latency is 1017.59 ms/question.
- On terminal-unrecoverable questions, the Router averages 20.027 chunks and 0.987 reranker calls.

## Research Questions

- RQ-A: Yes, query-only routing reduces cost relative to FixedHeavy by 5.185 chunks, 0.338 reranker calls, and 463.11 retrieval ms per question.
- RQ-B: This saving costs 2.60 percentage points of complete evidence coverage relative to FixedHeavy.
- RQ-C: Recoverable Under-Retrieval Rate is 7.10%. The zero Medium prediction rate is a material three-class failure and explains why accuracy alone overstates Router quality.
- RQ-D: On the frozen retrieval metrics, EvidenceAwareRisk10 uses 0.731 fewer chunks than the Router while gaining 1.15 percentage points of complete coverage and lowering recoverable under-retrieval. This supports a better descriptive quality-cost trade-off for evidence-conditioned sequential stopping.

No model, label construction, route mapping, threshold, or retrieval setting is changed from this Dev result.

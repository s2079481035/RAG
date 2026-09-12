# Adaptive-RAG Protocol Audit for Phase 4

## Audit Decision

The current 2Wiki experiment cannot be described as an official Adaptive-RAG reproduction. The only defensible unified-environment baseline is named **Adaptive-RAG-style Query Complexity Router**. It transfers the query-only routing principle, not the original paper's numerical setting.

## Original Method

The [Adaptive-RAG paper](https://aclanthology.org/2024.naacl-long.389/) predicts query complexity from the question and selects among three strategies: no retrieval, one retrieval-and-generation step, or iterative multi-step retrieval and generation. The [official repository](https://github.com/starsuzi/Adaptive-RAG) exposes these as `nor_qa`, `oner_qa`, and `ircot_qa`.

| Audit item | Original protocol |
|---|---|
| Router input | Query text |
| Complexity output | Three strategy labels |
| Strategy A | No retrieval |
| Strategy B | Single-step retrieval |
| Strategy C | Iterative multi-step retrieval |
| Supervision | Silver strategy-success labels, with simpler successful strategies preferred; dataset-level inductive labels supplement unresolved examples |
| Classifier | T5-Large in the reported implementation |
| Training mixture | SQuAD, Natural Questions, TriviaQA, MuSiQue, HotpotQA, and 2WikiMultiHopQA |
| Retriever | BM25 through Elasticsearch in the official code path |
| Multi-step method | IRCoT-style iterative retrieval and generation |
| Generators | Reported FLAN-T5 and GPT-3.5 configurations |
| Main quality metrics | Exact Match, token-level F1, and accuracy where applicable |
| Efficiency measures | Retrieval/generation steps and execution time relative to one-step retrieval |

The original repository samples a limited training mixture rather than training on the full 163,454-question 2Wiki `train_core` split used here. Its retriever, generator, corpus, and strategy semantics also differ from the frozen Phase 4 environment.

## Why Strict Mapping Is Invalid

The frozen Phase 4 ladder is:

1. `dense@5`
2. `hybrid@10`
3. `rerank@20`

All three stages retrieve evidence. They differ by candidate fusion, retrieval depth, and reranking, not by no-retrieval versus one-step versus iterative retrieval. In particular, `rerank@20` is a terminal reranking stage, not iterative query reformulation or IRCoT. Therefore:

- `dense@5` is not the original no-retrieval strategy.
- `hybrid@10` is not a faithful synonym for the original one-step BM25 strategy.
- `rerank@20` is not the original iterative multi-step strategy.

Renaming this mapping “Adaptive-RAG” would overstate fidelity.

## Frozen Adapted Protocol

If the optional Dev baseline is executed, its formal name is **Adaptive-RAG-style Query Complexity Router**.

| Component | Frozen adapted choice |
|---|---|
| Input at inference | Question text only |
| Forbidden inference inputs | Retrieved evidence, supporting facts, future retrieval outcomes, gold labels |
| Output | Direct three-class route, selected once before retrieval |
| Light | `dense@5` |
| Medium | `hybrid@10` |
| Heavy | `rerank@20` |
| Training split | `train_core` only |
| Dev model selection | `dev_calibration` only |
| Formal Dev evaluation | `dev_policy` only |
| Heldout | Locked; unavailable for fitting or protocol changes |
| Training target | Earliest frozen cumulative stage with complete gold supporting-fact coverage; Heavy if no stage is complete |
| Backbone | Same `BAAI/bge-reranker-base` family as the lightweight Controller |
| Input length | 512 tokens, question only |
| Prediction | Three-way argmax; no threshold sweep |
| Seeds if learned | 42, 123, 2026 |

Gold supporting facts are permitted only to construct `train_core` supervision and Dev evaluation labels. They are never router features. The target is an adapted retrieval-depth oracle, not the original answer-correctness silver label, and this difference must remain visible in the paper.

## Fair Evaluation

The adapted router and the evidence-aware sequential Controller must share:

- the benchmark-context corpus and sentence-256 chunks;
- dense retriever, BM25 fusion, reranker, and frozen stage outputs;
- Qwen2.5-7B generator, prompt, decoding, and answer extraction;
- answer evaluator and latency accounting.

Report Answer EM/F1, supporting-fact recall, complete evidence coverage, average chunks, reranker calls, LLM tokens, and latency. Because this router selects a complete strategy once rather than making evidence-conditioned Stop/Continue decisions, its error metric is **recoverable under-retrieval rate**, not Controller FSR. Terminal Retrieval Failure Rate must be reported separately.

## Execution Decision

The protocol is frozen, but execution is deferred. A learned three-seed router over 163,454 training questions is not a negligible-cost add-on, and the user instruction makes the Dev run optional only when cost is very low. Deferral does not block the primary LLM Judge comparison or invalidate the heldout protocol; it means no `adaptive_rag_dev_summary.csv` may be claimed unless the complete three-seed adapted baseline is later run exactly as specified above.

No heldout example or metric was consulted for this audit.

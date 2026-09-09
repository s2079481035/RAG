# Phase 4 Credibility Protocol

## Scope

Phase 4 adds credibility experiments only. It does not introduce a new retriever, reranker, Controller architecture, loss, packing method, calibration method, RL system, agent, or graph retriever.

The immutable Phase 3B reference is Git tag `phase3b_final` at commit `1ed762c`. Phase 4 runs on branch `research/sufficiency-phase4` and must not overwrite any formal Phase 1-3B artifact.

## 2WikiMultiHopQA

The official source is <https://github.com/Alab-NII/2wikimultihop>. Its documented JSON format uses `_id`, `question`, `answer`, `supporting_facts`, `context`, `evidences`, `type`, and `entity_ids`. Supporting facts must be exact `[title, sentence_id]` pairs and context documents exact `[title, sentences]` pairs. The parser fails on incompatible structures rather than silently remapping them.

Official Train is split by question ID with seed 42 into:

| Split | Role |
|---|---|
| `train_core` | model training only |
| `dev_calibration` | checkpoint selection and probability calibration |
| `dev_policy` | threshold/risk-policy selection |

Official labeled Dev is the final `heldout` evaluation set. Official unlabeled Test is unused. All stage decisions for one question remain in the same split.

## Corpus and Retrieval

All context documents in official Train and labeled Dev are pooled and exact title-plus-text duplicates are removed. The resulting resource is called the **shared benchmark-context corpus**, never Full Wikipedia.

The HotpotQA-selected sentence-aligned approximately 256-token chunking is transferred without a 2Wiki sweep. Dense (`BAAI/bge-large-en-v1.5`), BM25 defaults/tokenization, RRF, reranker (`BAAI/bge-reranker-v2-m3`), and the `Dense@5 -> Hybrid@10 -> Rerank@20` ladder remain frozen.

Evidence coverage requires exact title equality and exact sentence-ID membership. Binary Stop means complete coverage; Continue means incomplete coverage. Insufficient, Partial, Sufficient, and Hard Partial definitions are unchanged from Phase 3A.

## Controller Experiments

Cross-dataset zero-shot uses the HotpotQA-trained checkpoint, score-aware cumulative packing, temperature `1.0732333511122047`, and primary threshold `0.93` without 2Wiki fitting.

The in-domain Controller uses the frozen architecture and hyperparameters. Only Query+Stage and the final evidence-aware Controller are repeated for seeds 42, 123, and 2026. Calibration and risk selection use only internal Train splits. Held-out results cannot change a threshold, prompt, sampling ratio, or training setting.

## Baselines

- Score Threshold uses only the top-1 retrieval score, with stage-wise normalization fitted on `dev_policy`.
- LLM Judge uses the frozen Qwen generator, greedy decoding, and an exact binary-label prompt frozen before held-out evaluation. AUROC is reported only if genuine label probabilities are available.
- Official Adaptive-RAG reproduction is kept separate from an Adaptive-RAG-style Query Complexity Router adapted to this pipeline.
- S2G-RAG has a two-working-day reproduction limit. Partial or failed reproduction remains documented and is not relabeled as a successful fair baseline.

## Statistical Reporting

Core trained models use seeds 42, 123, and 2026. Final HotpotQA and 2Wiki comparisons use at least 2,000 paired question-level bootstrap replicates. Stage samples are never treated as independent bootstrap units.

All formal runs preserve per-sample outputs outside Git and commit compact manifests, summaries, tables, and figures. Phase 4 ends with an explicit GO/NO-GO and no further method expansion.

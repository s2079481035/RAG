# S2G-RAG Protocol Audit

## Audit Identity

- Designation: External Reference Experiment protocol audit
- Official paper: <https://aclanthology.org/2026.acl-long.1185/>
- Official repository: <https://github.com/nianaaa/S2G-RAG>
- Audited commit: `5d842a67a0a99a7b545bbad0dc402ceaae0e5eff`
- Repository license: no `LICENSE` or `COPYING` file was present at the audited commit. The paper's publication license must not be treated as a code license.
- Trained checkpoint: no official trained S2G-Judge adapter, release asset, model ID, or download URI was identified.

| Repository asset | Audit result |
|---|---|
| README | Present; documents environment variables, dataset/index preparation, training, BM25/E5 inference, and final-turn evaluation. |
| Dependency specification | `requirements.txt` is present, but it does not fully match the packages listed in the README (including retrieval/training extras). A clean environment would need manual reconciliation. |
| Dataset code | HotpotQA, 2WikiMultiHopQA, and TriviaQA preparation/retrieval paths are present; source datasets and indexes are not bundled. |
| Checkpoint | Base-model names are documented, but the task-trained S2G-Judge LoRA is not distributed. |
| Evaluation | `evaluate_S2G-RAG.py` computes final-turn EM/F1 after retaining the first 1,000 unique IDs. |
| Release/tag | No Git tag or GitHub release carrying reproducibility artifacts was found at audit time. |

## Method and Execution Protocol

| Question | Audited answer |
|---|---|
| 1. S2G-Judge input | The original question and accumulated, sentence-selected evidence context. It does not read the gold answer. |
| 2. Sufficiency label | A binary judgment of whether the current context contains enough information to answer reliably. An insufficient output also contains structured missing-information items. Teacher-generated process labels are filtered and used to LoRA-fine-tune the judge. |
| 3. Query + evidence? | Yes. The README and inference prompts explicitly provide both. |
| 4. Evidence memory | Retrieved documents are passed through a sentence-level Evidence Extractor. Selected sentences are appended to the accumulated evidence context; past document IDs/titles are also tracked. |
| 5. If insufficient | Gap items are converted into the next retrieval query, another retrieval turn runs, and new selected evidence is added to memory. |
| 6. Gap generation | The judge emits JSON with `sufficient` and `gap_items`; each gap item describes a missing link needed to answer the question. |
| 7. Query regeneration | The implementation converts the structured missing facts into a retrieval query rather than merely repeating the original question. |
| 8. Retrieval backend | Dataset-specific Pyserini/Lucene BM25 or E5 dense retrieval (`intfloat/e5-base-v2`), with FAISS when available and NumPy fallback. |
| 9. Per-turn top-k | `top_docs=6` by default. The extractor may select up to six sentences from retrieved documents. |
| 10. Maximum rounds | `max_turns=4` by default. |
| 11. Generator | The documented answer reasoner is Meta-Llama-3-8B-Instruct; the judge backbone is Llama-3.2-3B-Instruct plus a trained LoRA adapter. |
| 12. HotpotQA setting | Dataset-specific corpus/index, iterative retrieval, the same judge/extractor/reasoner loop, and final-turn answer evaluation. Training instructions use HotpotQA as the worked example. |
| 13. 2Wiki setting | The repository supplies 2Wiki corpus/index initialization and dataset handling. The README instructs repeating supervision preparation for 2Wiki when training a multi-dataset judge. |
| 14. Official metrics | Final-turn Exact Match and token-level F1. Trajectories also store turns, evidence, retrieved titles, and retrieval correctness, but the evaluator reports EM/F1. |
| 15. Trained Judge checkpoint? | No distributable trained adapter was found. The README explains how to train and save one locally. |
| 16. Direct inference without training? | No, not from the audited official artifacts. Both BM25 and E5 inference require `SUFF_LORA_PATH`; missing it raises an error, and loading an untrained/base judge would change the method. |

## Reproducibility Findings

The code exposes `LLAMA_PATH`, `SUFF_BASE_PATH`, and mandatory `SUFF_LORA_PATH`. The first two identify base models; the third is the task-trained controller. The official repository has no Git tag or release carrying that adapter. Consequently, a faithful smoke test cannot begin without first recreating teacher labels and training LoRA, which violates the no-retraining smoke protocol.

Additional reproducibility cautions:

- `requirements.txt` and the README installation list are not identical.
- The top-level `run_S2G-RAG.py` references entry points that are not present at the audited commit, while the README invokes current scripts directly.
- The evaluator takes the first 1,000 unique IDs rather than the pre-registered SHA-256 subset. A wrapper would therefore be required for the planned reference subset, but only after a successful smoke test.
- The inference implementation writes trajectory checkpoints by batch, but does not provide the required completed-ID resume manifest or 100-question progress manifest without a wrapper.

## Boundary with Ours

Shared ground:

- Both represent the decision state using the question and current evidence.
- Both ask whether the current evidence is sufficient before spending more retrieval cost.
- Both implement adaptive rather than uniformly maximal retrieval.

Material differences:

| Dimension | S2G-RAG | Ours |
|---|---|---|
| Insufficient-state output | Sufficiency plus structured gaps | Stop probability and auxiliary coverage estimate |
| Continuation | Generate a new query from missing information | Escalate through a frozen Dense@5 -> Hybrid@10 -> Rerank@20 ladder |
| Evidence memory | LLM sentence extraction and iterative memory construction | Frozen score-aware packing of cumulative retrieved evidence |
| Primary objective | Acquire missing evidence and improve final QA | Distinguish partial from sufficient evidence while controlling premature stopping risk and cost |
| Risk control | No matching pre-registered empirical FSR constraint in the audited pipeline | Dev-selected conservative threshold and frozen heldout risk/cost evaluation |

These differences prevent a unified fair numerical comparison unless retriever, corpus, generator, and evaluation protocol are aligned without altering either method.

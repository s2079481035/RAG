# Stop-RAG Protocol Audit

## Audit Identity

- Official paper: <https://arxiv.org/abs/2510.14337>
- Official repository: <https://github.com/chosolbee/Stop-RAG>
- Audited commit: `fcbcb781fd95d56b1fde72ada15357a6a6f03941`
- Official resource statement: all reported experiments used four H100 GPUs.

## Protocol Questions

| Question | Audited answer |
|---|---|
| 1. State | Conceptually, the interaction history: original question plus retrieval/query/document/intermediate-answer history. The released Q-network serializes the question and accumulated documents with separator tokens, omitting some conceptual trajectory fields from its encoder input. |
| 2. Actions | `STOP` and `CONTINUE`. |
| 3. Stop/Continue model | A DeBERTa-v3-large value model with two value heads estimates action values; the higher-valued action determines stopping, subject to horizon termination. |
| 4. Reward | Terminal answer quality, implemented from repeated answer generations and answer F1. Intermediate continuation has no independent immediate QA reward. |
| 5. Future utility | `CONTINUE` targets incorporate future prefix values/rewards through TD(lambda); lambda is scheduled during training. Thus continuation is valued by expected future answer gain, not current sufficiency alone. |
| 6. Value-model training | Required. The README instructs training, checkpoint scoring, dev checkpoint/threshold selection, then final evaluation. |
| 7. Trajectories | The base RAG system is rolled out to the maximum horizon. Every prefix becomes a state. The paper uses eight independent answer generations per state to estimate answer quality before Q-target construction. |
| 8. Retriever | The reported setup uses Contriever-MSMARCO; the repository also exposes BM25-related options. Each iteration retrieves ten candidates and retains the top reranked passage. |
| 9. Generator | The paper reports Llama-3.1-8B-Instruct for query and answer generation, with BGE reranking. |
| 10. Dataset use | HotpotQA, 2WikiMultiHopQA, and MuSiQue are downloaded and converted into method-specific corpora and trajectories. Validation/test subsets are sampled from development data. The 2Wiki/MuSiQue corpus construction pools contexts across official splits, unlike this study's frozen protocol. |
| 11. Compute | Four H100 GPUs for the official experiments. |
| 12. Pretrained checkpoint | No official trained Stop-RAG value-model checkpoint or release artifact was identified at the audited commit. |
| 13. Direct inference | No. A trained value model plus a selected checkpoint and threshold are required. |
| 14. Alignment with our ladder | Not strict. Stop-RAG uses a repeated, query-generating retrieval process with a longer horizon; Ours uses the frozen Dense@5 -> Hybrid@10 -> Rerank@20 ladder and a terminal forced stage. |

The reported maximum retrieval horizon is ten iterations. `STOP` is terminal, and reaching the horizon also terminates the trajectory. These semantics differ from treating Rerank@20 as the forced terminal stage in Ours.

## Central Theoretical Difference

Ours estimates:

`P(current evidence is sufficient | question, current evidence)`

Stop-RAG estimates an action value related to:

`expected future answer utility if retrieval continues from this trajectory state`.

These targets coincide only when insufficient evidence is reliably recoverable by the available next retrieval action. The frozen 2Wiki analysis shows that this assumption often fails.

## Experiment Decision

`STOP_RAG_EXPERIMENT_SKIPPED = true`

Reasons:

1. No official pretrained value-model checkpoint was identified.
2. Faithful execution requires full trajectory generation, repeated answer sampling, Q-target construction, and value-model training.
3. The official resource requirement exceeds the pre-registered smoke budget.
4. Mapping Stop-RAG onto our three stages would materially rewrite the method and would not be a valid Stop-RAG baseline.

Accordingly, Stop-RAG is retained as a protocol and theory comparison. A complete reproduction is not justified for this paper's frozen empirical scope.

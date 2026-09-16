# Retrieval-Limited Regimes and Future Utility

## Frozen Evidence

On the frozen 2Wiki heldout evaluation, final complete-evidence coverage is `0.397821`; terminal retrieval failure rate is therefore approximately `0.6022`. These are heldout descriptive results from the already completed Phase 4 run. No external-baseline work changed or reran them.

The Train-derived `dev_policy` rescue audit provides mechanism-level context:

- Among 1,356 questions incomplete after Dense@5, only 88 (`6.49%`) become sufficient by a later stage.
- Among 1,299 questions incomplete after Hybrid@10, only 31 (`2.39%`) are rescued by Rerank@20.
- Equivalently, `93.51%` of Dense incomplete states and `97.61%` of Hybrid incomplete states are unrecoverable within the frozen ladder.

The heldout ceiling and the dev-only rescue rates play different roles: the former describes final generalization, while the latter diagnoses where the fixed ladder gains evidence. Dev rescue percentages are not heldout estimates and were not used to change policy.

## Why Sufficiency and Continuation Utility Separate

`current evidence is insufficient` identifies a missing-evidence state. It does not imply that the available next retrieval action can find the missing evidence. In a retrieval-limited regime, the following can both be true:

1. Stopping now would be evidentially premature.
2. Continuing with the available retriever has low expected benefit.

This separates two estimands:

- Evidence sufficiency: whether the current evidence supports answering.
- Future retrieval utility: whether a specified continuation action is expected to improve answer quality enough to justify its cost.

Ours models the first. Stop-RAG is closer to the second. S2G-RAG attacks the separation by identifying missing information and generating a new query, potentially changing the retrieval action rather than merely valuing a fixed next stage.

## Consequences for Interpretation

- A Controller false stop at an actionable state remains an error under the sufficiency objective, even if the frozen ladder could not later recover the evidence.
- A forced terminal stop with incomplete evidence is a Terminal Retrieval Failure, not a Controller false stop.
- Low rescue rates explain why always continuing can spend substantially more retrieval cost without proportionate evidence gains.
- The approximately 60% 2Wiki terminal failure rate is a property of the evaluated corpus/retrieval ladder interaction, not proof that adaptive retrieval in general cannot solve those questions.
- Neither S2G-RAG nor Stop-RAG can be claimed to solve this ceiling without running their own official protocols; conversely, their different continuation mechanisms prevent a clean controller-only numerical comparison.

This boundary strengthens the paper's scope: it distinguishes reliable evidence-state recognition from the separate problem of acquiring missing evidence under a capable retrieval policy.

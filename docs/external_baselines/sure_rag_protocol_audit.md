# SURE-RAG Protocol Audit

## Audit Identity

- Paper: <https://arxiv.org/abs/2605.03534>
- Audit scope: method only; no experiment was pre-registered.
- Repository status: no author-linked official repository was identified from the paper record as of 2026-09-16.

## Method Questions

| Question | Audited answer |
|---|---|
| 1. Candidate answer input? | Yes. The state contains question, candidate answer, and retrieved evidence. |
| 2. Evidence representation | Candidate answers are decomposed into claims (a short answer is normally one claim). A shared DeBERTa-v3-base cross-encoder scores claim-passage pairs; aggregate coverage, relation-strength, uncertainty/conflict, and optional retrieval features represent the evidence support state. |
| 3. Output label | `Supported`, `Refuted`, or `Insufficient`. |
| 4. Label meaning | Supported evidence entails the central factual content; Refuted evidence contradicts it; Insufficient covers missing hops, partial/topical-only evidence, weak relations, or unresolved conflict. |
| 5. Answer verification? | Yes. It verifies a proposed answer against evidence. |
| 6. Retrieval stopping? | Not directly. Its operational choice is answer versus abstain, not continue retrieval versus stop retrieval. |
| 7. Calibration | A three-way logistic aggregation layer and post-hoc selection parameters are tuned on development data; threshold and risk/uncertainty weighting determine selective prediction. |
| 8. Risk | Unsafe answers among answered examples, evaluated across a risk-coverage curve. This is not Controller False Stop Rate. |
| 9. Selective answering | Answer only when the verifier predicts Supported and its safety score exceeds the threshold; otherwise abstain. |
| 10. Difference from Ours | SURE-RAG decides whether a candidate answer is evidentially safe to emit. Ours decides whether current evidence is sufficient to stop retrieval before a candidate answer is required. |

## Task Boundary

SURE-RAG:

`Question + Candidate Answer + Evidence -> support/refute/insufficient -> answer/abstain`

Ours:

`Question + Current Evidence -> stop/continue retrieval`

The common concern is evidence reliability, but the decision timing, action space, error denominator, and downstream intervention differ. A numerical experiment would therefore answer a different question and is unnecessary for the frozen paper.

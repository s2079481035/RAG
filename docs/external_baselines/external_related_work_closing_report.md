# External Related-Work Closing Report

## Scope and Integrity

This branch audited S2G-RAG, Stop-RAG, and SURE-RAG after the Phase 1-4 method and heldout results were frozen at `cc6951529fe77811bb287ed885b19ddbe9f7059c`. Ours was not retrained, retuned, recalibrated, or rerun on heldout. No external result was used to change the proposed method.

## Final Answers

### 1. Was S2G-RAG reproduced successfully?

No full reproduction was possible. Status is `PARTIAL`: the official method and code were audited, but inference could not faithfully start because the mandatory trained S2G-Judge LoRA is not distributed in the audited official artifacts. Recreating it would require a new teacher-labeling and LoRA-training pipeline, triggering the pre-registered stop condition.

### 2. Is S2G-RAG suitable as a numerical baseline in the main table?

No. There are no faithfully reproduced numbers, and even a future official-pipeline run would use a different retriever, corpus construction, iterative query mechanism, generator, and evidence memory. It should not be placed in a unified fair-comparison table or support an `Ours beats S2G-RAG` claim.

### 3. Can S2G-RAG be used as an external reference?

Yes. It is the closest method-level neighbor: both judge question-plus-current-evidence sufficiency, while S2G-RAG additionally produces structured gaps and new retrieval queries. The correct use is related-work positioning and, if an official checkpoint later becomes available, a clearly labeled different-stack descriptive reference.

### 4. Is a complete Stop-RAG reproduction warranted?

Not for this frozen paper. No trained value-model checkpoint was identified, faithful reproduction requires expensive trajectory generation and value training, the authors report four H100 GPUs, and its repeated retrieval horizon cannot be strictly aligned with our three-stage ladder. Rewriting it into our classifier would invalidate the baseline.

### 5. Does SURE-RAG require an experiment?

No. SURE-RAG verifies a candidate answer and chooses answer versus abstain. Ours evaluates evidence before answer generation and chooses stop versus continue retrieval. Their risk denominators and interventions differ, so a method-level distinction is more informative than a forced numerical comparison.

### 6. What is the real novelty boundary?

- S2G-RAG: missing-information diagnosis and query reformulation for evidence acquisition.
- Stop-RAG: expected future retrieval utility learned as a value function.
- SURE-RAG: calibrated candidate-answer verification and selective answering.
- Ours: hard-partial sufficiency discrimination using a lightweight evidence-aware critic, coverage auxiliary supervision, conservative threshold selection, empirical false-stop evaluation, and an explicit retrieval cost/latency boundary across retrieval-sufficient and retrieval-limited regimes.

The central limitation is also explicit: present evidence insufficiency is not equivalent to future retrieval usefulness. Ours measures the former and empirically exposes the latter through rescue and terminal-failure analyses; it does not claim to solve missing-evidence acquisition.

### 7. Are more experiments needed?

No for the current paper. The external audit found no unaddressed novelty collision that invalidates the contribution. Additional training-heavy reproductions would introduce protocol mismatch and delay writing without creating a defensible fair comparison. A future update may run S2G-RAG only if the authors publish the exact trained adapter and dependencies needed for direct inference.

## Reporting Rules

- Call this work `External Reference Experiments`.
- Do not report empty S2G rows as zero performance; `N=0` means not executed.
- Do not perform paired significance tests across different retrieval/generation stacks.
- Do not interpret latency or token differences as controller-only effects.
- Keep Stop-RAG and SURE-RAG in the method-level comparison unless protocol-compatible official artifacts become available.

## Decision

`MORE_EXPERIMENTS_NEEDED = NO`

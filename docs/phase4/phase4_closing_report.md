# Phase 4 Closing Report

## GO_FOR_PAPER_WRITING = YES

The frozen external heldout evaluation is complete and can now be used for paper writing. This decision means the evidence package is complete; it does not turn unsupported statistical claims into supported ones.

## Supported Claims

- The frozen Risk10 operating point did not meet the 10% observed heldout FSR target.
- Answer F1 noninferiority at the preregistered -0.02 margin is supported.
- Risk10 changed average retrieved chunks by -7.025 and total latency by -551.91 ms relative to FixedHeavy.
- Evidence-conditioned stopping, score routing, hard-label LLM Judges, and a three-seed query-only Router are compared under the same frozen retrieval ladder.

## Unsupported Claims

- No distribution-free or guaranteed risk-control claim is made from a point estimate alone.
- The Adaptive-RAG-style Router is not described as a strict official reproduction.
- No claim is made that terminal retrieval failures are Controller false stops.

## Limitations

- Final retrieval remains incomplete for 60.22% of heldout questions within the shared benchmark-context corpus.
- Latency is hardware- and corpus-dependent; Controller/Router latency uses the frozen synchronized batch-one Dev benchmark while retrieval and generation use heldout measurements.
- The same generator supplies QA answers and LLM-Judge labels, so the Judge is a compute-heavy baseline rather than an independent human oracle.

## Unexpected Results

- Risk10 FSR 95% CI is [0.1181, 0.1284]. The interval is wholly above 10%, so the frozen policy fails the heldout risk target.
- The query-only Router Medium-route mean is 0.0000.
- Alpha=5% behavior is retained exactly as frozen, including an always-final outcome if that is what the Dev gate selected.

No Phase 5 is started. No heldout threshold, prompt, model, retrieval setting, or routing label is changed.

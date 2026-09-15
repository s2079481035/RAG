# Phase 4 2Wiki Heldout Analysis

This is the single frozen evaluation on the official labeled 2WikiMultiHopQA Dev split (12,576 questions), used only as heldout. No heldout choice or threshold fitting is performed.

## Main Results

| System | EM | F1 | Complete coverage | FSR | Chunks | Reranker | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_heavy | 0.1678 | 0.1922 | 0.3978 | N/A | 20.2933 | 1.0000 | 1676.3053 |
| temperature_conservative_risk_10 | 0.1769 | 0.2026 | 0.3552 | 0.1231 | 13.2687 | 0.5010 | 1124.3924 |
| AdaptiveRouter_seed42 | 0.1707 | 0.1958 | 0.3663 | N/A | 15.4173 | 0.6813 | 1234.2085 |

## Frozen Risk Boundary

| Policy | FSR | RFSR | UER | Answer F1 | Chunks | Reranker |
|---|---:|---:|---:|---:|---:|---:|
| temperature_conservative_risk_20 | 0.1231 | 0.4298 | 0.0028 | 0.2026 | 13.2687 | 0.5010 |
| temperature_conservative_risk_10 | 0.1231 | 0.4298 | 0.0028 | 0.2026 | 13.2687 | 0.5010 |
| temperature_conservative_risk_05 | 0.1064 | 0.3661 | 0.0039 | 0.2036 | 13.5172 | 0.5178 |

The primary alpha=10% policy has observed FSR 0.1231 with question-level 95% CI [0.1181, 0.1284]. The interval is wholly above 10%, so the frozen policy fails the heldout risk target.

## Retrieval Ceiling

Final complete-evidence coverage is 0.3978; TRFR is 0.6022. The two values are exact complements. Terminal forced stop is excluded from every Controller FSR.
The retrieval oracle averages 14.994 chunks, 0.635 reranker calls, and 916.29 ms.

## Frozen Baselines

Judge-512 and Judge-Full strict parse rates are 1.0000 and 1.0000; AUROC is not reported for their hard labels. The three query-only Router seeds are reported without seed reselection.
The Router mean Medium-route rate is 0.0000; a zero rate, if present, is retained as a failure mode.

## Bootstrap

Risk10 minus FixedHeavy Answer F1 delta is +0.0104 (95% CI [+0.0057, +0.0153]); EM delta is +0.0091. Chunks change by -7.025 and total latency by -551.91 ms.
Using the preregistered -0.02 Answer F1 margin, noninferiority is supported.

## Answer Decomposition

Risk10 answer outcomes are partitioned into complete/correct, complete/wrong, incomplete/correct, and incomplete/wrong groups. Recoverable and unrecoverable false stops are reported separately in `results/phase4/2wiki/heldout/answer_failure_groups.csv`.

## Research Questions

1. RQ1: The final frozen ladder reaches complete evidence for 0.3978 of heldout questions; TRFR is 0.6022.
2. RQ2: Dense-to-final and Hybrid-to-final rescue rates are 0.1176 and 0.0516.
3. RQ3: EvidenceAwareClassification achieves Macro F1 0.9611, AUROC 0.9950, and FSR 0.0170 on reached actionable states.
4. RQ4: Frozen Risk10 observed FSR is 0.1231; its question-level interval is [0.1181, 0.1284].
5. RQ5: Risk10 Answer F1 delta versus FixedHeavy is +0.0104, with noninferiority supported at margin -0.02.
6. RQ6: Risk10 changes chunks by -7.025, reranker calls by -0.499, and latency by -551.91 ms.
7. RQ7: Judge-512 and Judge-Full FSRs are 0.0315 and 0.0197; their extra token and latency costs are reported in the external-baseline table.
8. RQ8: The query-only Router three-seed route Macro F1 is 0.5936 +/- 0.0024; Medium routing is not repaired post hoc.
9. RQ9: Cross-dataset transfer is judged from the frozen HotpotQA/2Wiki table; conclusions are limited to the observed risk-efficiency boundary rather than identical calibration.

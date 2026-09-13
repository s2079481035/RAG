# Phase 4 2Wiki LLM-as-a-Judge Analysis

All formal comparisons use the same 2,000 `dev_policy` questions, cumulative evidence, and actionable stages (`dense@5`, `hybrid@10`). `rerank@20` is terminal and is excluded from Judge and Controller FSR.

## Sequential Results

| System | Accuracy | Macro F1 | Stop P | Stop R | FSR | Hard Partial FSR | RFSR | UER | Latency/question ms | Input tokens/question | Output tokens/question | Parse rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Judge-512 | 0.7269 | 0.5642 | 0.7276 | 0.1873 | 3.14% | 3.12% | 1.68% | 81.27% | 265.77 | 1220.4 | 5.7 | 100.00% |
| Judge-Full | 0.7177 | 0.5336 | 0.7556 | 0.1430 | 2.10% | 2.05% | 3.36% | 85.70% | 332.30 | 1876.1 | 5.7 | 100.00% |
| EvidenceAwareClassification | 0.9920 | 0.9878 | 0.9829 | 0.9787 | 0.45% | 0.47% | 2.52% | 2.13% | 22.00 | 804.9 | 0.0 | N/A |
| EvidenceAwareRisk10 | 0.9588 | 0.9415 | 0.8345 | 1.0000 | 5.21% | 4.90% | 26.13% | 0.00% | 21.62 | 790.5 | 0.0 | N/A |

`EvidenceAwareClassification` uses the frozen seed-42 classification threshold; `EvidenceAwareRisk10` uses the separately frozen temperature-scaled risk-policy threshold. These are different operating concepts.

Judge outputs are hard labels. No Judge AUROC or artificial threshold sweep is reported. Invalid output, if any, follows the preregistered safe action `Continue`.

## Context Effect

Relative to Judge-512, Judge-Full changes sequential Macro F1 by -0.0307, FSR by -1.04 percentage points, and mean input tokens per question by +655.6.

## Retrieval-Limited Boundary

The frozen final-stage terminal retrieval failure rate is 63.40%. It is a retriever ceiling diagnostic, not a Controller false stop. RFSR measures only incorrect early stops whose evidence could actually become sufficient later in the frozen ladder.

## Interpretation

- Judge accuracy and FSR must be interpreted jointly with latency and token use; a larger model is not automatically a better adaptive policy.
- Judge-512 is the primary fair representation comparison. Judge-Full isolates the additional benefit of the larger context budget.
- This Dev analysis freezes the baseline result and does not authorize heldout access or any retuning.

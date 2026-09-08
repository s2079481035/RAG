# Phase 3B Closing Report

## Decision: GO

The pre-registered operational criteria support the claim that the primary policy preserves QA quality within the declared noninferiority margin while reducing retrieval and latency cost relative to Fixed Heavy.

## Decision Criteria

| Criterion | Passed |
|---|---|
| observed_test_fsr_target | True |
| answer_f1_noninferiority | True |
| retrieved_chunks_reduction | True |
| total_latency_reduction | True |

- Primary policy: `temperature_conservative_risk_10`
- Observed Test FSR: 8.45% (target 10.00%)
- Primary Test FSR question-bootstrap 95% CI: [5.64%, 11.77%]
- Answer F1 delta 95% CI: [-0.0193, 0.0185]
- Retrieved-chunk delta 95% CI: [-11.6610, -10.8510]
- Total-latency delta 95% CI: [-250.0802, -229.5747]
- FSR delta 95% CI: [0.0564, 0.1177]

## Qualification

The pre-registered primary policy has strict two-dimensional Pareto status `False`. `raw_conservative_risk_10` has the same Test F1 and FSR to numerical precision and changes average retrieved chunks by -0.005 per question. This raw-policy comparison is descriptive and does not replace the frozen primary policy.

The observed primary FSR meets the target, but its confidence interval is not wholly below 10%. Temperature Scaling also did not jointly improve held-out NLL and ECE, and the raw conservative alpha=0.10 policy produced effectively the same Test operating point. The GO verdict therefore does not establish guaranteed risk control or a distinct deployment benefit from Temperature Scaling itself.

## Thesis Boundary

Phase 3B is complete enough to report regardless of the verdict. A NO-GO blocks a positive deployment-style claim, not thesis finalization around a transparent negative trade-off. Do not retune alpha, temperature, threshold, retrieval, or generation from Test outcomes.

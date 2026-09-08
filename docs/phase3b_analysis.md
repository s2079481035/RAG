# Phase 3B Analysis

Phase 3B evaluates a frozen Phase 3A Controller. Temperature is fitted on `dev_calibration`; thresholds are selected on disjoint `dev_policy`; Test performs no refit or reselection.

## Table 1: Calibration Quality

| Split | Method | NLL | Brier | ECE | AUROC | Hard Partial high-conf. FSR |
|---|---|---|---|---|---|---|
| dev_calibration | raw | 0.1665 | 0.0490 | 0.0268 | 0.9623 | 6.15% |
| dev_calibration | temperature | 0.1661 | 0.0490 | 0.0254 | 0.9623 | 6.15% |
| dev_policy | raw | 0.1614 | 0.0439 | 0.0201 | 0.9543 | 9.82% |
| dev_policy | temperature | 0.1602 | 0.0440 | 0.0210 | 0.9543 | 8.93% |

On dev_policy, joint NLL/ECE improvement is **not observed**. AUROC is descriptive because scalar Temperature Scaling preserves ranking apart from numerical ties.

## Table 2: Risk-Controlled Controller (Test)

| Policy | FSR | UER | Macro F1 | Stop rate | Avg final stage | Hard Partial FSR |
|---|---|---|---|---|---|---|
| fixed_dense | 100.00% | 0.00% | 0.4444 | 100.00% | 1.000 | 100.00% |
| fixed_hybrid | 33.33% | 47.06% | 0.4872 | 50.00% | 2.000 | 34.78% |
| fixed_heavy | 0.00% | 100.00% | 0.1304 | 0.00% | 3.000 | 0.00% |
| always_stop_earliest | 100.00% | 0.00% | 0.4444 | 100.00% | 1.000 | 100.00% |
| always_continue_final | 0.00% | 100.00% | 0.1304 | 0.00% | 3.000 | 0.00% |
| oracle_stop_analysis_only | 0.00% | 0.00% | 1.0000 | 75.00% | 1.300 | 0.00% |
| uncalibrated_adaptive_phase3a_threshold | 33.21% | 4.46% | 0.8327 | 80.82% | 1.225 | 33.05% |
| temperature_calibrated_threshold_0.5 | 28.47% | 6.57% | 0.8329 | 78.14% | 1.261 | 28.39% |
| raw_point_risk_20 | 25.09% | 7.91% | 0.8342 | 76.37% | 1.285 | 25.21% |
| raw_conservative_risk_20 | 15.46% | 15.90% | 0.8007 | 68.40% | 1.402 | 15.92% |
| raw_point_risk_10 | 13.70% | 20.18% | 0.7708 | 65.06% | 1.457 | 14.23% |
| raw_conservative_risk_10 | 8.45% | 28.62% | 0.7206 | 57.80% | 1.579 | 8.43% |
| raw_point_risk_05 | 8.45% | 28.62% | 0.7206 | 57.80% | 1.579 | 8.43% |
| raw_conservative_risk_05 | 0.00% | 100.00% | 0.1304 | 0.00% | 3.000 | 0.00% |
| temperature_point_risk_20 | 25.09% | 8.32% | 0.8305 | 76.10% | 1.289 | 25.21% |
| temperature_conservative_risk_20 | 15.81% | 15.54% | 0.8026 | 68.72% | 1.397 | 16.33% |
| temperature_point_risk_10 | 13.36% | 20.96% | 0.7656 | 64.43% | 1.467 | 13.82% |
| temperature_conservative_risk_10 | 8.45% | 28.69% | 0.7201 | 57.76% | 1.580 | 8.43% |
| temperature_point_risk_05 | 8.45% | 28.69% | 0.7201 | 57.76% | 1.580 | 8.43% |
| temperature_conservative_risk_05 | 0.00% | 100.00% | 0.1304 | 0.00% | 3.000 | 0.00% |

FSR and UER include only actionable Controller decisions actually reached by each sequential policy. Final-stage insufficient evidence is reported separately in the CSV and is not mislabeled as a controllable False Stop.

## Table 3: End-to-End RAG (Test)

| Policy | EM | F1 | SF recall | Complete cov. | Chunks | Rerank | Tokens | Latency ms | Pareto |
|---|---|---|---|---|---|---|---|---|---|
| fixed_dense | 0.3710 | 0.4659 | 0.8959 | 0.8000 | 5.00 | 0.000 | 728.9 | 233.79 | True |
| fixed_hybrid | 0.4030 | 0.5074 | 0.9499 | 0.9000 | 11.50 | 0.000 | 1523.3 | 346.63 | False |
| fixed_heavy | 0.4070 | 0.5108 | 0.9613 | 0.9220 | 20.65 | 1.000 | 2653.7 | 587.79 | True |
| always_stop_earliest | 0.3710 | 0.4659 | 0.8959 | 0.8000 | 5.00 | 0.000 | 728.9 | 233.79 | False |
| always_continue_final | 0.4070 | 0.5108 | 0.9613 | 0.9220 | 20.65 | 1.000 | 2653.7 | 587.79 | False |
| oracle_stop_analysis_only | 0.4130 | 0.5150 | 0.9613 | 0.9220 | 7.30 | 0.100 | 1012.9 | 285.21 | False |
| uncalibrated_adaptive_phase3a_threshold | 0.4000 | 0.4984 | 0.9366 | 0.8740 | 6.69 | 0.052 | 939.2 | 284.96 | True |
| temperature_calibrated_threshold_0.5 | 0.3980 | 0.4965 | 0.9394 | 0.8790 | 6.96 | 0.067 | 974.0 | 291.52 | False |
| raw_point_risk_20 | 0.3990 | 0.4989 | 0.9417 | 0.8830 | 7.15 | 0.079 | 998.1 | 296.41 | True |
| raw_conservative_risk_20 | 0.3970 | 0.4977 | 0.9467 | 0.8930 | 8.04 | 0.130 | 1106.5 | 316.25 | False |
| raw_point_risk_10 | 0.4000 | 0.5023 | 0.9493 | 0.8980 | 8.44 | 0.149 | 1156.0 | 325.08 | True |
| raw_conservative_risk_10 | 0.4090 | 0.5104 | 0.9533 | 0.9060 | 9.38 | 0.207 | 1271.6 | 347.58 | True |
| raw_point_risk_05 | 0.4090 | 0.5104 | 0.9533 | 0.9060 | 9.38 | 0.207 | 1271.6 | 347.58 | True |
| raw_conservative_risk_05 | 0.4070 | 0.5108 | 0.9613 | 0.9220 | 20.65 | 1.000 | 2653.7 | 614.06 | True |
| temperature_point_risk_20 | 0.3990 | 0.4994 | 0.9417 | 0.8830 | 7.17 | 0.080 | 1001.4 | 297.01 | True |
| temperature_conservative_risk_20 | 0.3960 | 0.4967 | 0.9462 | 0.8920 | 8.01 | 0.128 | 1101.8 | 315.34 | False |
| temperature_point_risk_10 | 0.4010 | 0.5033 | 0.9493 | 0.8980 | 8.51 | 0.154 | 1165.3 | 326.93 | True |
| temperature_conservative_risk_10 | 0.4090 | 0.5104 | 0.9533 | 0.9060 | 9.38 | 0.207 | 1272.1 | 347.67 | False |
| temperature_point_risk_05 | 0.4090 | 0.5104 | 0.9533 | 0.9060 | 9.38 | 0.207 | 1272.1 | 347.67 | False |
| temperature_conservative_risk_05 | 0.4070 | 0.5108 | 0.9613 | 0.9220 | 20.65 | 1.000 | 2653.7 | 614.06 | True |

The pre-registered primary policy has strict two-dimensional Pareto status `False`. `raw_conservative_risk_10` has the same Test F1 and FSR to numerical precision and changes average retrieved chunks by -0.005 per question. This raw-policy comparison is descriptive and does not replace the frozen primary policy.

## Table 4: Risk-Level Comparison (Test)

| Risk target | Selection | Threshold | Test FSR | Observed target met | Answer F1 | Chunks | Latency ms |
|---|---|---|---|---|---|---|---|
| 20.00% | point | 0.560 | 25.09% | False | 0.4994 | 7.17 | 297.01 |
| 20.00% | conservative | 0.770 | 15.81% | True | 0.4967 | 8.01 | 315.34 |
| 10.00% | point | 0.840 | 13.36% | False | 0.5033 | 8.51 | 326.93 |
| 10.00% | conservative | 0.930 | 8.45% | True | 0.5104 | 9.38 | 347.67 |
| 5.00% | point | 0.930 | 8.45% | False | 0.5104 | 9.38 | 347.67 |
| 5.00% | conservative | 1.000 | 0.00% | True | 0.5108 | 20.65 | 614.06 |

## Paired Question Bootstrap

| Metric | Observed delta | 95% CI |
|---|---|---|
| answer_f1 | -0.0004 | [-0.0193, 0.0185] |
| retrieval_cost | -11.2660 | [-11.6610, -10.8510] |
| latency_ms | -240.1203 | [-250.0802, -229.5747] |
| false_stop_rate | 0.0845 | [0.0564, 0.1177] |

All deltas are primary risk policy minus Fixed Heavy and use 2,000 paired question-level replicates.

## Research Questions

- RQ1 Calibration: NLL/ECE jointly improved on held-out dev_policy: **False**.
- RQ2 Risk transfer: Dev FSR was 4.69%; Test FSR was 8.45% for the primary alpha=0.10 policy. The Test question-bootstrap 95% CI was [5.64%, 11.77%]. Its upper bound exceeds the 10.00% target, so the result is empirical risk control rather than a statistical guarantee.
- RQ3 Cost: versus Fixed Heavy, retrieved chunks changed by -11.266 and total latency by -240.120 ms.
- RQ4 QA quality: primary Test F1 was 0.5104, versus 0.5108 for Fixed Heavy.
- RQ5 Sweet spot: the pre-registered positive-claim rule is **met**.
- RQ6 Conservative stability: point-selected policies missed all three nominal Test targets. Conservative alpha=0.20 and alpha=0.10 met their observed targets; conservative alpha=0.05 met its target only by always reaching the forced final stage. No stronger guarantee is claimed from bootstrap confidence-bound selection.

## Interpretation Limits

This study supports only risk-aware, risk-constrained, or empirically controlled false-stop language. It does not establish guaranteed, reliable, or safe stopping. The underlying Test questions were already evaluated during Phase 3A, so this is not a fresh study-level holdout.

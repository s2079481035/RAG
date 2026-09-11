# 2Wiki Retrieval Ceiling Analysis

This analysis uses only `train_core`, `dev_calibration`, and `dev_policy`. The official labeled Dev (`heldout`) is not read.

## False Stop Scope Audit

Formal sequential-policy FSR is computed only over reached `dense@5` and `hybrid@10` decisions. `rerank@20` is removed with `[:-1]` in `phase3b_metrics.operational_metrics`, policy selection/bootstrap, and policy evaluation. A terminal incomplete state is therefore a Terminal Retrieval Failure, not a Controller False Stop.

Training-time `dev_metrics.json` is an all-stage classification diagnostic and must not be reported as sequential-policy FSR. No Phase 3B result requires recomputation.

## Retrieval Ceiling And Oracle

| Split | Questions | Dense coverage | Hybrid coverage | Final coverage | TRFR | Oracle chunks | Oracle reranker calls | Oracle latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_core | 163454 | 33.07% | 35.85% | 37.18% | 62.82% | 14.94 | 0.642 | 982.87 |
| dev_calibration | 2000 | 33.90% | 37.65% | 38.95% | 61.05% | 14.73 | 0.624 | 1003.12 |
| dev_policy | 2000 | 32.20% | 35.05% | 36.60% | 63.40% | 15.06 | 0.649 | 1028.04 |

TRFR is checked as exactly `1 - Final Complete Evidence Coverage`. The Oracle stops at the first sufficient cumulative state and otherwise reaches the forced final stage; it is an analysis bound, not a deployment method.

## Stage Rescue

| Split | Rescue | Eligible incomplete | Rescued | Rescue rate | Coverage gain | Extra chunks / rescue | Extra latency ms / rescue |
|---|---|---:|---:|---:|---:|---:|---:|
| train_core | Dense_to_Hybrid_Rescue | 109402 | 4542 | 4.15% | 2.78% | 146.59 | 31518.44 |
| train_core | Dense_to_Final_Rescue | 109402 | 6716 | 6.14% | 4.11% | 247.99 | 22674.85 |
| train_core | Hybrid_to_Final_Rescue | 104860 | 2174 | 2.07% | 1.33% | 440.87 | 4024.22 |
| dev_calibration | Dense_to_Hybrid_Rescue | 1322 | 75 | 5.67% | 3.75% | 107.76 | 23900.84 |
| dev_calibration | Dense_to_Final_Rescue | 1322 | 101 | 7.64% | 5.05% | 199.42 | 18551.09 |
| dev_calibration | Hybrid_to_Final_Rescue | 1247 | 26 | 2.09% | 1.30% | 437.42 | 2941.08 |
| dev_policy | Dense_to_Hybrid_Rescue | 1356 | 57 | 4.20% | 2.85% | 143.95 | 32273.82 |
| dev_policy | Dense_to_Final_Rescue | 1356 | 88 | 6.49% | 4.40% | 234.60 | 21850.29 |
| dev_policy | Hybrid_to_Final_Rescue | 1299 | 31 | 2.39% | 1.55% | 384.23 | 2571.80 |

Extra cost per rescue divides the total incremental cost paid by all eligible incomplete questions by the number actually rescued.

## Dev Policy By Gold Supporting-Fact Count

| Gold facts | Questions | Dense coverage | Hybrid coverage | Final coverage |
|---|---:|---:|---:|---:|
| 2 | 1575 | 40.70% | 44.25% | 46.22% |
| 3 | 2 | 100.00% | 100.00% | 100.00% |
| 4+ | 423 | 0.24% | 0.47% | 0.47% |

The 3-fact stratum contains only two dev-policy questions. All bucket comparisons are descriptive; no inferential claim is made.

## Controller Recoverability Diagnostics

RFSR is pending frozen Train-derived Controller predictions. No placeholder value is inferred while multi-seed training is running. Re-run this analysis with `--prediction LABEL PATH THRESHOLD_OR_SAVED` after dev-policy inference.

## Answers To The Audit Questions

1. Dev-policy final retrieval coverage is 36.60%; the retrieval ceiling failure rate is 63.40%.
2. Of Dense Continue questions, 88/1356 (6.49%) become sufficient by a later stage.
3. Of Hybrid Continue questions, 31/1299 (2.39%) are rescued by Rerank.
4. Recoverable false-stop attribution is pending completed Controller predictions; it is not fabricated from retrieval labels.
5. On dev_policy, 1268/1356 (93.51%) Dense Continue states and 1268/1299 (97.61%) Hybrid Continue states are unrecoverable within the frozen ladder. Across both actionable slots, only 119/2655 (4.48%) Continue states are potentially recoverable before accounting for policy reachability; 1268/2000 questions are terminal-incomplete.
6. Final forced stops are excluded from formal sequential FSR; terminal incompleteness is reported as TRFR.
7. This analysis does not change the frozen Phase 4 protocol, model, retrieval parameters, labels, or thresholds, and it does not consult heldout.

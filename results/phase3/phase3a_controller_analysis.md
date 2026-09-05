# Phase 3A Controller Analysis

All test metrics use thresholds selected on dev. Standard deviation is the sample standard deviation across training seeds.

| Group | Metric | Mean | Std | Seeds |
|---|---|---:|---:|---:|
| query_evidence_concat | macro_f1 | 0.7523 | 0.0169 | 3 |
| query_evidence_concat | stop_f1 | 0.9482 | 0.0007 | 3 |
| query_evidence_concat | auroc | 0.9081 | 0.0108 | 3 |
| query_evidence_concat | false_stop_rate | 0.5362 | 0.0559 | 3 |
| query_evidence_concat | hard_partial_false_stop_rate | 0.5511 | 0.0565 | 3 |
| query_evidence_concat | unnecessary_escalation_rate | 0.0287 | 0.0071 | 3 |
| query_evidence_concat | coverage_mae | N/A | N/A | 0 |
| query_evidence_concat | coverage_spearman | N/A | N/A | 0 |
| query_stage | macro_f1 | 0.5773 | 0.0351 | 3 |
| query_stage | stop_f1 | 0.9298 | 0.0039 | 3 |
| query_stage | auroc | 0.8207 | 0.0081 | 3 |
| query_stage | false_stop_rate | 0.8466 | 0.0642 | 3 |
| query_stage | hard_partial_false_stop_rate | 0.8462 | 0.0662 | 3 |
| query_stage | unnecessary_escalation_rate | 0.0252 | 0.0154 | 3 |
| query_stage | coverage_mae | N/A | N/A | 0 |
| query_stage | coverage_spearman | N/A | N/A | 0 |
| score_aware_balanced_stop_continue | macro_f1 | 0.7833 | 0.0137 | 3 |
| score_aware_balanced_stop_continue | stop_f1 | 0.9541 | 0.0010 | 3 |
| score_aware_balanced_stop_continue | auroc | 0.8892 | 0.0276 | 3 |
| score_aware_balanced_stop_continue | false_stop_rate | 0.4832 | 0.0422 | 3 |
| score_aware_balanced_stop_continue | hard_partial_false_stop_rate | 0.4892 | 0.0458 | 3 |
| score_aware_balanced_stop_continue | unnecessary_escalation_rate | 0.0243 | 0.0035 | 3 |
| score_aware_balanced_stop_continue | coverage_mae | N/A | N/A | 0 |
| score_aware_balanced_stop_continue | coverage_spearman | N/A | N/A | 0 |
| score_aware_baseline | macro_f1 | 0.7971 | 0.0133 | 3 |
| score_aware_baseline | stop_f1 | 0.9561 | 0.0014 | 3 |
| score_aware_baseline | auroc | 0.9246 | 0.0072 | 3 |
| score_aware_baseline | false_stop_rate | 0.4515 | 0.0415 | 3 |
| score_aware_baseline | hard_partial_false_stop_rate | 0.4572 | 0.0367 | 3 |
| score_aware_baseline | unnecessary_escalation_rate | 0.0244 | 0.0029 | 3 |
| score_aware_baseline | coverage_mae | N/A | N/A | 0 |
| score_aware_baseline | coverage_spearman | N/A | N/A | 0 |
| score_aware_coverage_aux_lambda_0.1 | macro_f1 | 0.7965 | 0.0101 | 3 |
| score_aware_coverage_aux_lambda_0.1 | stop_f1 | 0.9559 | 0.0014 | 3 |
| score_aware_coverage_aux_lambda_0.1 | auroc | 0.9147 | 0.0223 | 3 |
| score_aware_coverage_aux_lambda_0.1 | false_stop_rate | 0.4515 | 0.0268 | 3 |
| score_aware_coverage_aux_lambda_0.1 | hard_partial_false_stop_rate | 0.4572 | 0.0288 | 3 |
| score_aware_coverage_aux_lambda_0.1 | unnecessary_escalation_rate | 0.0249 | 0.0010 | 3 |
| score_aware_coverage_aux_lambda_0.1 | coverage_mae | 0.0674 | 0.0095 | 3 |
| score_aware_coverage_aux_lambda_0.1 | coverage_spearman | 0.4758 | 0.0183 | 3 |
| score_aware_coverage_aux_lambda_0.1_hard_partial_aware | macro_f1 | 0.7872 | 0.0404 | 3 |
| score_aware_coverage_aux_lambda_0.1_hard_partial_aware | stop_f1 | 0.9546 | 0.0043 | 3 |
| score_aware_coverage_aux_lambda_0.1_hard_partial_aware | auroc | 0.9040 | 0.0019 | 3 |
| score_aware_coverage_aux_lambda_0.1_hard_partial_aware | false_stop_rate | 0.4647 | 0.1107 | 3 |
| score_aware_coverage_aux_lambda_0.1_hard_partial_aware | hard_partial_false_stop_rate | 0.4696 | 0.1082 | 3 |
| score_aware_coverage_aux_lambda_0.1_hard_partial_aware | unnecessary_escalation_rate | 0.0258 | 0.0075 | 3 |
| score_aware_coverage_aux_lambda_0.1_hard_partial_aware | coverage_mae | 0.0664 | 0.0040 | 3 |
| score_aware_coverage_aux_lambda_0.1_hard_partial_aware | coverage_spearman | 0.4317 | 0.0127 | 3 |
| score_aware_hard_partial_aware | macro_f1 | 0.7914 | 0.0106 | 3 |
| score_aware_hard_partial_aware | stop_f1 | 0.9549 | 0.0021 | 3 |
| score_aware_hard_partial_aware | auroc | 0.9058 | 0.0077 | 3 |
| score_aware_hard_partial_aware | false_stop_rate | 0.4594 | 0.0522 | 3 |
| score_aware_hard_partial_aware | hard_partial_false_stop_rate | 0.4727 | 0.0474 | 3 |
| score_aware_hard_partial_aware | unnecessary_escalation_rate | 0.0258 | 0.0094 | 3 |
| score_aware_hard_partial_aware | coverage_mae | N/A | N/A | 0 |
| score_aware_hard_partial_aware | coverage_spearman | N/A | N/A | 0 |

Hard Partial results must be interpreted with overall Macro F1 and unnecessary escalation rate; a lower False Stop Rate alone can result from overpredicting Continue.

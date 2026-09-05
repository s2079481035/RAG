# Phase 3A Hard Partial Analysis

All buckets below contain actual Continue examples only. Consequently, within-bucket AUROC is undefined. `Stop vs bucket AUROC` compares each Continue bucket against all truly Sufficient test examples.

| Model | Continue bucket | N | False Stop Rate | Mean Stop P | P90 Stop P | Stop vs bucket AUROC |
|---|---|---:|---:|---:|---:|---:|
| QueryStage | easy_continue | 10 | 0.6000 | 0.4328 | 0.9786 | 0.8982 |
| QueryStage | medium_partial | 45 | 0.8667 | 0.5560 | 0.9995 | 0.8350 |
| QueryStage | hard_partial | 323 | 0.8050 | 0.6136 | 0.9994 | 0.8094 |
| ScoreAwareBaseline | easy_continue | 10 | 0.1000 | 0.0801 | 0.1519 | 0.9936 |
| ScoreAwareBaseline | medium_partial | 45 | 0.6222 | 0.5290 | 0.9687 | 0.9213 |
| ScoreAwareBaseline | hard_partial | 323 | 0.4768 | 0.4202 | 0.9971 | 0.9140 |
| FinalController | easy_continue | 10 | 0.1000 | 0.1273 | 0.2321 | 0.9937 |
| FinalController | medium_partial | 45 | 0.4667 | 0.4165 | 0.9088 | 0.9334 |
| FinalController | hard_partial | 323 | 0.4241 | 0.3870 | 0.9586 | 0.9303 |

The test distribution and coverage values are unchanged. These are frozen-error analyses, not threshold-selection results.

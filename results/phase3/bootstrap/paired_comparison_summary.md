# Phase 3A Paired Bootstrap Comparisons

All differences are `model - baseline` and reuse the saved question-level bootstrap draws. No model inference or Test tuning is performed.

| Comparison | Metric | Observed delta | 95% CI | Excludes zero |
|---|---|---:|---:|---:|
| ScoreAwareBaseline_minus_QueryStage | macro_f1 | 0.1860 | [0.1482, 0.2199] | yes |
| ScoreAwareBaseline_minus_QueryStage | auroc | 0.1022 | [0.0730, 0.1298] | yes |
| ScoreAwareBaseline_minus_QueryStage | false_stop_rate | -0.3228 | [-0.3874, -0.2542] | yes |
| FinalController_minus_QueryStage | macro_f1 | 0.2068 | [0.1709, 0.2404] | yes |
| FinalController_minus_QueryStage | auroc | 0.1176 | [0.0926, 0.1426] | yes |
| FinalController_minus_QueryStage | false_stop_rate | -0.3862 | [-0.4485, -0.3212] | yes |
| FinalController_minus_ScoreAwareBaseline | macro_f1 | 0.0209 | [-0.0025, 0.0450] | no |
| FinalController_minus_ScoreAwareBaseline | auroc | 0.0154 | [0.0028, 0.0282] | yes |
| FinalController_minus_ScoreAwareBaseline | false_stop_rate | -0.0635 | [-0.1136, -0.0155] | yes |

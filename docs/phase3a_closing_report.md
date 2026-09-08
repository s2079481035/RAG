# Phase 3A Closing Report

## Scope and protocol

Phase 3A asks whether an evidence-aware Controller can distinguish Partial Evidence from truly Sufficient Evidence reliably enough to justify later calibration and risk-controlled stopping. All model-selection decisions were made on Dev. Test thresholds were copied from Dev and were not refit on Test. The core comparison uses the same `sentence_256` chunks, cumulative evidence trajectory, BAAI/bge-reranker-base backbone, optimizer, four training epochs, and seeds 42, 123, and 2026.

The Test set contains 1,000 questions and 3,000 stage decisions. Question-level bootstrap resamples question IDs and keeps all decisions from a sampled question together. Generation is a separate fixed `rerank@20` evaluation and does not execute the learned stopping policy.

## Executive result

Evidence-aware classification is substantially better than Query+Stage, and score-aware packing is consistently better than concat-truncate. However, the best stable three-seed Test result still falsely stops on about 45% of Continue decisions. Coverage auxiliary supervision predicts coverage moderately well but does not improve the Controller consistently across seeds. Hard-Partial-aware sampling also fails to improve the baseline.

The Phase 3A decision is therefore:

- **No-Go for describing or deploying the current Controller as a reliable stopping policy.**
- **Conditional Go for a narrowly scoped Phase 3B calibration/risk study**, provided the protocol is frozen on Dev and the already inspected Test set is not presented as a fresh independent confirmation.

## RQ1: Is supporting-fact coverage a reasonable answerability proxy?

The deterministic, stratified Dev audit contains 180 records: 12 Insufficient, 89 Partial, and 79 Sufficient. Automatic and human labels agree on all audited records.

| Comparison | Accuracy | Macro F1 | Sufficient precision | Sufficient recall |
|---|---:|---:|---:|---:|
| Automatic Stop vs Human Stop | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Automatic three-class vs Human three-class | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Priority disagreement counts are both zero: Automatic Sufficient to Human Not Sufficient = 0, and Automatic Continue to Human Sufficient = 0.

This supports gold supporting-fact coverage as an answerability proxy **within the audited Dev protocol**. It is not a population accuracy estimate because the audit was stratified, used one reviewer, and was not an independent blinded annotation study. It also does not prove that the Controller saw all relevant facts: only 22/180 records had all gold facts visible in packed evidence, while 158/180 had at least one gold fact outside the model-visible packed view. Human labels validate the full retrieved-evidence proxy, not lossless packing.

Source: [`manual_audit_analysis.md`](manual_audit_analysis.md), [`manual_audit_confusion.csv`](../results/phase3/manual_audit_confusion.csv).

## RQ2: Are the Phase 2 conclusions stable across seeds?

The core ordering is stable in all three seeds: Query+Stage < Query+Evidence concat < Query+Evidence score-aware. The table reports Test mean and sample standard deviation across seeds.

| Controller | Macro F1 | AUROC | False Stop Rate | Hard Partial FSR | Unnecessary Escalation |
|---|---:|---:|---:|---:|---:|
| Query+Stage | 0.5773 +/- 0.0351 | 0.8207 +/- 0.0081 | 0.8466 +/- 0.0642 | 0.8462 +/- 0.0662 | 0.0252 +/- 0.0154 |
| Query+Evidence concat | 0.7523 +/- 0.0169 | 0.9081 +/- 0.0108 | 0.5362 +/- 0.0559 | 0.5511 +/- 0.0565 | 0.0287 +/- 0.0071 |
| Query+Evidence score-aware | **0.7971 +/- 0.0133** | **0.9246 +/- 0.0072** | **0.4515 +/- 0.0415** | **0.4572 +/- 0.0367** | **0.0244 +/- 0.0029** |

Relative to concat, score-aware packing raises Macro F1 by 0.0448 and AUROC by 0.0165, and lowers False Stop Rate by 0.0847. Relative to Query+Stage, concat raises Macro F1 by 0.1750 and lowers False Stop Rate by 0.3104.

The ranking metrics are reasonably stable, but Dev-selected thresholds vary substantially by seed. This is additional evidence that discrimination and decision calibration must be reported separately.

Source: [`multi_seed_summary.csv`](../results/phase3/multi_seed_summary.csv), [`seed_results.csv`](../results/phase3/seed_results.csv), [`collection_manifest.json`](../results/phase3/collection_manifest.json).

## Question-level bootstrap

The paired analysis uses 2,000 shared question-level bootstrap draws. The rows below are model minus baseline for the fixed seed-42 frozen Test predictions.

| Comparison | Metric | Observed delta | 95% CI | Excludes zero |
|---|---|---:|---:|:---:|
| Score-aware baseline - Query+Stage | Macro F1 | +0.1860 | [0.1482, 0.2199] | yes |
| Score-aware baseline - Query+Stage | AUROC | +0.1022 | [0.0730, 0.1298] | yes |
| Score-aware baseline - Query+Stage | False Stop Rate | -0.3228 | [-0.3874, -0.2542] | yes |
| Final Controller - Score-aware baseline | Macro F1 | +0.0209 | [-0.0025, 0.0450] | no |
| Final Controller - Score-aware baseline | AUROC | +0.0154 | [0.0028, 0.0282] | yes |
| Final Controller - Score-aware baseline | False Stop Rate | -0.0635 | [-0.1136, -0.0155] | yes |

The first comparison strongly supports evidence-aware score-aware input over Query+Stage. The second comparison shows a favorable seed-42 auxiliary result, but it does not establish robustness to training randomness: its Macro F1 interval includes zero, and the three-seed means below do not reproduce the Controller improvement. Question bootstrap quantifies Test-question uncertainty for fixed models; it does not replace multi-seed training uncertainty.

Source: [`paired_comparison_summary.md`](../results/phase3/bootstrap/paired_comparison_summary.md), [`bootstrap_summary.csv`](../results/phase3/bootstrap/bootstrap_summary.csv).

## RQ3: Are False Stops concentrated in Hard Partial evidence?

Among the 378 cumulative Test Continue decisions, 10 are Easy Continue, 45 are Medium Partial, and 323 are Hard Partial. Thus Hard Partial accounts for 85.4% of Continue decisions.

| Model | Easy FSR (n=10) | Medium FSR (n=45) | Hard FSR (n=323) | Hard false stops / all false stops |
|---|---:|---:|---:|---:|
| Query+Stage | 0.6000 | 0.8667 | 0.8050 | 260/305 (85.2%) |
| Score-aware baseline | 0.1000 | 0.6222 | 0.4768 | 154/183 (84.2%) |
| Dev-selected Final Controller | 0.1000 | 0.4667 | 0.4241 | 137/159 (86.2%) |

False Stops are concentrated in Hard Partial by **absolute count**, primarily because Hard Partial dominates the Continue distribution. Hard Partial is not the highest conditional-error bucket: Medium Partial has a higher FSR for all three displayed models, although its estimate is based on only 45 decisions. The defensible conclusion is therefore prevalence-driven concentration, not proof that Hard Partial is intrinsically the most difficult conditional bucket.

Source: [`hard_partial_analysis.md`](../results/phase3/hard_partial/hard_partial_analysis.md).

## RQ4: Does coverage auxiliary supervision help?

Lambda was selected on Dev from {0.1, 0.3, 1.0}; the frozen choice was 0.1. The auxiliary head learns a nontrivial coverage signal on Test (MAE 0.0674 +/- 0.0095; Spearman 0.4758 +/- 0.0183), but this does not translate into a stable Controller improvement.

| Test metric | Score-aware baseline | Coverage auxiliary | Auxiliary - baseline |
|---|---:|---:|---:|
| Macro F1 | 0.7971 | 0.7965 | -0.0006 |
| AUROC | 0.9246 | 0.9147 | -0.0099 |
| False Stop Rate | 0.4515 | 0.4515 | 0.0000 |
| Hard Partial FSR | 0.4572 | 0.4572 | 0.0000 |
| Unnecessary Escalation | 0.0244 | 0.0249 | +0.0005 |

Answer: **No stable improvement is demonstrated.** The favorable fixed seed-42 bootstrap result should be reported as seed-specific rather than generalized as an auxiliary-task success.

Source: [`coverage_lambda_selection.json`](../results/phase3/coverage_lambda_selection.json), [`ablation_summary.csv`](../results/phase3/ablation_summary.csv).

## RQ5: Does Hard Partial sampling help without collapsing to Continue?

| Sampling | Macro F1 | AUROC | False Stop Rate | Hard Partial FSR | Unnecessary Escalation |
|---|---:|---:|---:|---:|---:|
| Natural score-aware | **0.7971** | **0.9246** | **0.4515** | **0.4572** | 0.0244 |
| Balanced Stop/Continue | 0.7833 | 0.8892 | 0.4832 | 0.4892 | **0.0243** |
| Hard-Partial-aware | 0.7914 | 0.9058 | 0.4594 | 0.4727 | 0.0258 |

Hard-Partial-aware sampling does not reduce False Stop Rate; relative to natural sampling it lowers Macro F1 by 0.0056 and AUROC by 0.0189 while increasing Hard Partial FSR by 0.0155. Balanced Stop/Continue sampling is also worse on the principal discrimination and safety metrics. The combined auxiliary plus Hard-Partial-aware variant is worse than baseline and has high False Stop Rate variability (0.4647 +/- 0.1107).

Answer: **No.** The tested sampling schemes do not provide the desired safety improvement, and their failure is not explained by a simple collapse to always Continue because unnecessary escalation remains low.

Source: [`sampling_summary.csv`](../results/phase3/sampling_summary.csv), [`ablation_summary.csv`](../results/phase3/ablation_summary.csv).

## RQ6: Is the Controller ready for Phase 3B?

The score-aware model has useful ranking ability (Test AUROC 0.9246 +/- 0.0072), and its improvement over Query+Stage is supported by paired question bootstrap. Nevertheless, an average False Stop Rate of 0.4515 means that nearly half of actual Continue decisions are stopped prematurely at the Dev-selected operating points. Stop F1 near 0.956 should not be used to conceal this error because Stop examples dominate the 3,000 stage decisions (2,622 Stop vs 378 Continue).

The current model is **not a reliable stopping policy**. Phase 3B is justified only as a controlled calibration and risk-threshold investigation, not as deployment validation. Phase 3B should preregister a target risk metric, fit calibration and thresholds on Dev only, preserve per-question evaluation, and report the False Stop versus unnecessary-escalation tradeoff. Because the present Test set has already been inspected, later reuse must be declared non-independent or replaced by a fresh holdout/external evaluation for confirmatory claims.

The Dev-selected auxiliary Controller must not be silently replaced after seeing Test. Any Phase 3B comparison with the simpler score-aware baseline needs a newly frozen protocol and an explicit selection boundary.

## End-to-end generation evaluation

The Qwen2.5-7B-Instruct generation evaluation uses fixed cumulative `rerank@20` evidence, a frozen short-answer prompt, greedy decoding, `max_new_tokens=32`, first-nonempty-line extraction, and standard HotpotQA normalization.

| Split | Samples | Exact Match | Token F1 | Multiline | Longer than 16 tokens | Empty | Context truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dev | 1,000 | 0.4350 | 0.5289 | 0 | 11 | 0 | 1 |
| Test | 1,000 | 0.4070 | 0.5108 | 1 | 9 | 0 | 0 |

Test is 0.0280 lower in EM and 0.0181 lower in token F1. The 50-record stratified Dev audit found no extraction or scoring failures. Output-format failures are uncommon, and no empty answer appears in either split, so the evaluator is suitable as a stable end-to-end QA measurement.

This experiment does **not** measure the learned Controller's end-to-end quality/cost tradeoff because generation always consumes fixed `rerank@20` evidence. In addition, the local model manifest records `model_revision: null`; the path and configuration were gate-checked, but the exact local weight files are not cryptographically pinned in the committed manifest.

Source: [`Dev evaluation`](../results/phase3/generation/dev/evaluation.md), [`Test evaluation`](../results/phase3/generation/test/evaluation.md), [`test_gate_approval.json`](../results/phase3/generation/dev/test_gate_approval.json).

## Claims supported by Phase 3A

- Query+Evidence substantially outperforms Query+Stage under matched training and evaluation conditions.
- Score-aware packing is the strongest tested representation and improves both discrimination and False Stop Rate over concat-truncate.
- Supporting-fact coverage agrees with human answerability in the stratified Dev audit, subject to the audit limitations above.
- Most False Stops occur on Hard Partial records by count because Hard Partial dominates the Continue class.
- Coverage regression is learnable, but the auxiliary loss does not yield a robust multi-seed Controller improvement.
- Balanced and Hard-Partial-aware sampling do not improve the natural score-aware baseline.
- The frozen HotpotQA generation evaluator is operational and stable across Dev and Test.

## Claims not supported

- The current Controller is a reliable or deployment-ready stopping policy.
- Hard Partial has a higher conditional False Stop Rate than every other Continue bucket.
- Coverage auxiliary supervision robustly improves stopping decisions across seeds.
- Hard-Partial-aware sampling lowers False Stops.
- Fixed `rerank@20` generation demonstrates end-to-end gains from dynamic stopping.
- The 180-record stratified, single-reviewer audit estimates population-level label accuracy.

## Final Go / No-Go

**Research Go:** continue to Phase 3B with a small, preregistered calibration/risk-control study because the score-aware Controller has meaningful discrimination and a clear operating-point problem.

**Policy No-Go:** do not claim reliable stopping, deploy the current thresholded Controller, or add RL/agent complexity. The next phase should first determine whether Dev-only calibration and conservative risk thresholds can reduce False Stops to a predefined acceptable range without making unnecessary escalation prohibitive.

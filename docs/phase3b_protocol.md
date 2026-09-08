# Phase 3B Protocol

## Research objective

Phase 3B studies whether a frozen evidence-aware sufficiency Controller can trade retrieval cost for empirically controlled False Stop risk while preserving end-to-end HotpotQA answer quality. It does not change the backbone, packing strategy, training objective, sampling strategy, retrieval model, reranker, or generation protocol.

The primary metric order is: False Stop Rate, answer F1/EM, retrieval cost, latency, reranker calls, token cost, then Macro F1/AUROC.

## Frozen Base Sufficiency Controller

The Base Sufficiency Controller is the Phase 3A Dev-selected group's seed-42 run. Seed 42 is the pre-registered canonical replicate rather than a post-hoc best-seed choice:

- Run: `experiments/phase3a/lambda_dev/lambda_0.1_seed42`
- Backbone: `BAAI/bge-reranker-base`
- Input: Query + cumulative evidence with score-aware packing
- Objective: binary Stop/Continue cross-entropy plus `0.1 * SmoothL1(coverage)`
- Sampling: natural
- Checkpoint selection: highest Dev Stop F1 at the fixed training threshold, followed by Dev-only threshold selection in Phase 3A

`scripts/prepare_phase3b.py` verifies these fields against the Phase 3A selection and run manifests and records hashes without copying or retraining the checkpoint.

## Split protocol

The original 1,000-question Dev split is deterministically divided by `question_id`, seed 42, into:

- `dev_calibration`: 500 questions, used only to fit one temperature.
- `dev_policy`: 500 questions, used only for threshold/risk-policy selection.

All stage decisions from one question remain in the same subset. Stage-level random splitting is forbidden. Test cannot be read by preparation, calibration, or policy-selection commands.

## Calibration

Raw Stop probabilities are compared with scalar Temperature Scaling. Temperature is fitted only on the actionable `dense@5` and `hybrid@10` decisions in `dev_calibration` by minimizing binary NLL. Both methods report NLL, Brier score, equal-width 15-bin ECE, and AUROC on the same actionable-stage distribution in `dev_calibration` and `dev_policy`. Reliability diagrams use only Dev subsets.

Temperature Scaling is monotonic and is not expected to improve AUROC. Its purpose is probability reliability and interpretable risk thresholds.

## Actionable risk definition

`dense@5` and `hybrid@10` are actionable Controller stages: a Continue decision can retrieve more evidence. `rerank@20` is the forced final stage and always generates an answer because no stronger registered stage exists.

The primary Phase 3B False Stop Rate is therefore:

```text
false Stop decisions reached by the sequential policy
-------------------------------------------------
all true Continue decisions reached by the sequential policy
```

An actionable decision is counted only if the preceding policy decisions actually reach it. The final-stage insufficient-evidence rate is reported separately. Phase 3A all-stage FSR is retained only as a historical comparison. Treating forced final generation as a thresholded Stop decision would make strict risk targets partly reflect retrieval exhaustion rather than a controllable policy error.

## Threshold and risk policy

On `dev_policy`, thresholds from 0.00 through 1.00 in 0.01 increments are evaluated. Each threshold records Controller metrics and a sequential trajectory:

1. Dense@5 -> Stop or Hybrid@10
2. Hybrid@10 -> Stop or Rerank@20
3. Rerank@20 -> forced generation

For each risk target `alpha` in {0.20, 0.10, 0.05}, the point policy chooses the lowest-cost threshold satisfying observed FSR <= alpha. The conservative policy instead requires the question-level-bootstrap 95% upper FSR bound <= alpha. Cost is frozen as average retrieved chunks for selection; measured latency and token cost are evaluation outputs, not tuning objectives.

The bootstrap uses 2,000 question-level replicates and keeps both actionable stage decisions from each sampled question together. The result is described as bootstrap-confidence-bound conservative selection, not a probabilistic guarantee.

## End-to-end evaluation

The Phase 3A prompt, greedy decoding, 32-token output limit, first-line answer extraction, and HotpotQA normalization remain unchanged. Stage-level generation is cached once per `(question_id, stage)` so all policies compare identical generations for an identical evidence state.

Required baselines are Fixed Dense, Fixed Hybrid, Fixed Heavy/Rerank, raw adaptive, temperature-calibrated adaptive, point and conservative risk policies at 20%/10%/5%, Always Stop Earliest, Always Continue to Final, and an analysis-only Oracle Stop.

Oracle uses gold supporting-fact coverage only as a theoretical boundary and is never described as deployable.

## Cost and latency

The policy trajectory stores final stage, cumulative chunks, unique titles, reranker calls, Controller calls, LLM input/output tokens, and component latencies. Fresh component benchmarks must use warm-up and CUDA synchronization around GPU work and report per-query values plus mean, median, and P95.

Historical Phase 2 batched/allocated latency may be shown as an estimate, but cannot be labeled the official Phase 3B measured latency. Retrieved K is not treated as the sole compute-cost measure.

## Test gate and prior exposure

Calibration parameters, threshold sweep, risk targets, selected point/conservative thresholds, and the primary report policy are hashed into `results/phase3b/test_gate.json` before Phase 3B Test execution. Test cannot refit or reselect any of them.

The same underlying Test questions were already evaluated in Phase 3A. Therefore Phase 3B Test is evaluation-only for the newly frozen policy but is **not a fresh study-level holdout**. Confirmatory claims require a fresh holdout or external dataset; later reports must state this limitation.

## Pre-registered closing rule

The primary temperature-scaled, conservative alpha=0.10 policy supports a positive efficient-risk-control claim only if: its observed Test FSR is at most 0.10; the paired Answer F1 delta versus Fixed Heavy has a 95% CI lower bound of at least -0.02; and the paired retrieved-chunk and total-latency delta CIs both have upper bounds below zero. Failure is a NO-GO for that positive claim, not a reason to retune Test or discard a scientifically useful negative result.

## Prohibited changes

Phase 3B does not add RL, agents, GraphRAG, a new large backbone, Set Transformer, new retrieval/reranking models, Test threshold tuning, Test risk-target selection, or prompt tuning based on Test.

## Planned outputs

- `results/phase3b/calibration_metrics.csv`
- `results/phase3b/dev_threshold_sweep.csv`
- `results/phase3b/selected_thresholds.json`
- `results/phase3b/test_risk_policy.csv`
- `results/phase3b/end_to_end_summary.csv`
- `results/phase3b/bootstrap_ci.csv`
- `figures/phase3b/reliability_raw.png`
- `figures/phase3b/reliability_temperature.png`
- `figures/phase3b/risk_cost_curve.png`
- `figures/phase3b/quality_cost_curve.png`
- per-sample JSONL trajectories and complete run manifests

No unexecuted result is inserted into analysis or closing reports.

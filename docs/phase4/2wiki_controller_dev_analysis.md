# 2Wiki In-domain Controller Dev Gate

This report uses only `dev_calibration` and `dev_policy` derived from the official 2Wiki Train split. The labeled official Dev (`heldout`) is not read.

## Multi-seed Stability

Each run uses its threshold selected on `dev_calibration`. Formal FSR below scores only sequentially reached `dense@5` and `hybrid@10` decisions; the forced `rerank@20` terminal state is excluded.

| Controller | Macro F1 | Stop F1 | AUROC | FSR | Hard Partial FSR | UER |
|---|---:|---:|---:|---:|---:|---:|
| final_evidence_aware | 0.9873 +/- 0.0005 | 0.9798 +/- 0.0008 | 0.9994 +/- 0.0001 | 0.47% +/- 0.02% | 0.48% +/- 0.02% | 2.27% +/- 0.14% |
| query_stage | 0.9299 +/- 0.0029 | 0.8890 +/- 0.0050 | 0.9792 +/- 0.0035 | 2.81% +/- 0.66% | 2.90% +/- 0.68% | 11.41% +/- 2.83% |

The operational model is seed 42 by preregistration, not the best `dev_policy` seed. All-stage training metrics remain diagnostics and are not substituted for sequential FSR.

## Calibration

Temperature is fitted on seed 42 actionable `dev_calibration` decisions: T = 1.34738343.

| Split | Method | NLL | Brier | ECE | AUROC |
|---|---|---:|---:|---:|---:|
| dev_calibration | raw | 0.0295 | 0.0074 | 0.0063 | 0.9995 |
| dev_calibration | temperature | 0.0271 | 0.0072 | 0.0055 | 0.9995 |
| dev_policy | raw | 0.0259 | 0.0069 | 0.0059 | 0.9996 |
| dev_policy | temperature | 0.0240 | 0.0066 | 0.0046 | 0.9996 |

Scalar Temperature Scaling preserves AUROC ranking. Here it improves NLL, Brier score, and ECE on both Train-derived Dev subsets; this is a calibration result, not a discrimination gain.

## Dev Risk-Efficiency Boundary

| Target | Selection | Threshold | FSR | FSR CI high | Hard Partial FSR | Chunks | Rerank calls | Latency ms | Final coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20.00% | point | 0.010 | 5.21% | 6.04% | 4.90% | 14.338 | 0.589 | 986.61 | 35.15% |
| 20.00% | conservative | 0.010 | 5.21% | 6.04% | 4.90% | 14.338 | 0.589 | 986.61 | 35.15% |
| 10.00% | point | 0.010 | 5.21% | 6.04% | 4.90% | 14.338 | 0.589 | 986.61 | 35.15% |
| 10.00% | conservative | 0.010 | 5.21% | 6.04% | 4.90% | 14.338 | 0.589 | 986.61 | 35.15% |
| 5.00% | point | 0.020 | 4.08% | 4.82% | 3.78% | 14.499 | 0.604 | 993.60 | 35.25% |
| 5.00% | conservative | 0.020 | 4.08% | 4.82% | 3.78% | 14.499 | 0.604 | 993.60 | 35.25% |

## Frozen Primary Dev Policy

- Policy: `temperature_conservative_risk_10`
- Threshold: `0.01`
- Observed sequential Dev-policy FSR: 5.21%
- Question-bootstrap FSR 95% CI: [4.35%, 6.04%]
- Average retrieved chunks: 14.338
- Average reranker calls: 0.589
- Estimated retrieval latency: 986.61 ms
- Final complete evidence coverage: 35.15%
- Cost reduction vs always-final: chunks 29.21%, reranker calls 41.10%, estimated retrieval latency 32.79%
- Recoverable FSR: 26.13%
- Unrecoverable early-stop rate: 4.28%
- Recoverable share of false stops: 21.32%
- Recoverable share of reached Continue states: 4.25%

## Gate Boundary

No heldout metric, model selection, threshold tuning, retrieval change, or label change is performed here. Heldout remains locked until this Dev gate and any remaining external-baseline protocol choices are committed.

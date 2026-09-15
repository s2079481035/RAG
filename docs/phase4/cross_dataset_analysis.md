# Cross-Dataset Analysis

HotpotQA and 2WikiMultiHopQA use the same frozen three-stage ladder, generator, answer evaluator, and risk-policy interpretation. Corpus scope differs and absolute latency should therefore be compared cautiously.

| Dataset | System | Answer F1 | FSR | Coverage | Chunks | Reranker | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | FixedHeavy | 0.5108 | 0.0000 | 0.9220 | 20.6460 | 1.0000 | 587.7891 |
| HotpotQA | Risk10 | 0.5104 | 0.0845 | 0.9060 | 9.3800 | 0.2070 | 347.6689 |
| 2WikiMultiHopQA | FixedHeavy | 0.1922 | N/A | 0.3978 | 20.2933 | 1.0000 | 1676.3053 |
| 2WikiMultiHopQA | Risk10 | 0.2026 | 0.1231 | 0.3552 | 13.2687 | 0.5010 | 1124.3924 |

The transferable claim is about the risk-efficiency boundary, not identical absolute quality or calibration across datasets.

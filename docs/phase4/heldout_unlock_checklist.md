# Phase 4 Heldout Unlock Checklist

This audit reads only frozen configuration and Train-derived Dev manifests. It does not read or run heldout data.

| Requirement | Status |
|---|---|
| `controller_frozen` | PASS |
| `controller_threshold_frozen` | PASS |
| `controller_classification_threshold_frozen` | PASS |
| `risk10_threshold_frozen` | PASS |
| `retrieval_frozen` | PASS |
| `chunking_frozen` | PASS |
| `generator_frozen` | PASS |
| `generation_prompt_frozen` | PASS |
| `llm_judge_prompt_frozen` | PASS |
| `adaptive_rag_protocol_frozen` | FAIL |
| `adaptive_rag_model_frozen` | PASS |
| `score_threshold_baseline_frozen` | PASS |
| `score_threshold_frozen` | PASS |
| `metrics_frozen` | PASS |
| `adaptive_rag_dev_artifacts_frozen` | PASS |
| `no_outstanding_dev_tuning` | FAIL |
| `heldout_consulted == false` | PASS |

## READY_FOR_HELDOUT = NO

Even when this checklist is fully green, heldout remains operationally locked until the researcher records explicit human confirmation. This script never starts heldout evaluation.

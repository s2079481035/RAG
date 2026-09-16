# S2G-RAG Reproduction Status

## Status: PARTIAL

The official paper, repository, method, configuration, and executable entry points were audited successfully. No model inference was started.

## Stop Condition

The official BM25 and E5 inference paths require a trained S2G-Judge LoRA through `SUFF_LORA_PATH`. At audited commit `5d842a67a0a99a7b545bbad0dc402ceaae0e5eff`:

- no trained adapter is committed;
- no release asset exists;
- no model-hub identifier or checkpoint download URI is documented;
- the README instead provides a teacher-labeling and LoRA-training procedure.

Recreating this adapter would be a new training experiment, could require proprietary teacher API access, and would no longer be a direct official-checkpoint smoke test. This triggers the pre-registered `official checkpoint unavailable` and `large model retraining required` stop conditions.

## Actions Not Taken

- No HotpotQA or 2Wiki example was sent through S2G-RAG.
- No result-based subset was selected.
- No base model was substituted for the trained judge.
- No algorithm, prompt, retrieval backend, or evaluator was rewritten.
- No Ours model, threshold, calibration, retrieval ladder, or heldout output was touched.

The planned 32+32 smoke test, deterministic 1,000+1,000 subsets, and formal reference runs are therefore not applicable. Their empty, non-numerical status is recorded in `results/external_baselines/s2g_smoke_test.json` and `s2g_reference_summary.csv`.

## Interpretation

This is not evidence that S2G-RAG fails. It means the publicly audited artifacts are insufficient for a faithful no-retraining reproduction under the fixed two-to-three-day budget. S2G-RAG remains a method-level related-work comparison, not a numerical baseline in the main result table.

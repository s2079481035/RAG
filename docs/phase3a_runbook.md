# Phase 3A Runbook

Phase 2 is frozen. Do not rerun `run_phase2_selected.sh`; Phase 3A consumes its existing chunk, Controller-data, and retrieval outputs.

## Protocol Locks

- Controller seeds: `42`, `123`, `2026`.
- Checkpoint selection: Dev Stop F1 at threshold `0.5`, matching Phase 2.
- Decision threshold: selected on Dev with the Phase 2 threshold grid.
- Coverage lambda: selected from `0.1`, `0.3`, and `1.0` on Dev only.
- Final Controller: selected using mean multi-seed Dev metrics before any Phase 3A Test evaluation.
- Test never selects a threshold, lambda, sampling ratio, prompt, or extraction rule.
- Weighted samplers draw exactly one training-set-sized epoch with replacement. Dev and Test distributions are unchanged.

## 0. Close Phase 2 Without Retraining

After switching to this branch on the server, refresh only the compact Phase 2 summaries. This adds the missing `split=dev` counterfactual metadata and exports already-computed packing visibility fields. It does not run retrieval, training, or model inference.

```bash
export PYTHON_BIN=python3.12
bash run_phase3a.sh phase2-close
```

## 1. Finish the Manual Audit

Fill all 180 rows in `docs/manual_sufficiency_audit.csv`. Use `yes/no` for `human_sufficient_to_answer`, one of `insufficient/partial/sufficient` for `human_label`, and a fixed reviewer ID. Add `human_notes` whenever the human and automatic three-class labels disagree. Do not alter automatic fields.

For easier review, start the dependency-free browser UI on the server:

```bash
python3.12 scripts/review_manual_sufficiency_audit.py --port 8765
```

Forward port `8765` with VS Code Remote SSH and open the forwarded address in a local browser. The server binds only to `127.0.0.1`. It displays one logical CSV record at a time, keeps automatic diagnostics collapsed during the initial judgment, derives the binary human label from the three-class choice, and atomically saves only the human annotation fields. Progress resumes from the saved CSV after a restart.

```bash
export PYTHON_BIN=python3.12
bash run_phase3a.sh manual-audit
```

Outputs:

- `docs/manual_audit_analysis.md`
- `results/phase3/manual_audit_confusion.csv`

The audit judges answerability from full retrieved evidence. It does not prove that all evidence was visible after 512-token packing and never changes Test labels.

## 2. Train and Select on Dev

Use the same local Controller checkpoint as Phase 2 when needed:

```bash
export PYTHON_BIN=python3.12
export GPU_ID=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export CONTROLLER_MODEL_PATH=/absolute/path/to/BAAI/bge-reranker-base

tmux new -s phase3a
bash run_phase3a.sh train-dev 2>&1 | tee results/phase3_train_dev.log
```

`GPU_ID` is copied to `CUDA_VISIBLE_DEVICES` inside the run script and the selected physical GPU is printed before model loading. PyTorch will still call that one visible device `cuda:0`; use the printed physical index or `nvidia-smi` PID, not the logical PyTorch name, to verify placement.

This stage trains all core seeds, performs the three-lambda Dev sweep, trains sampling/auxiliary ablations, and records the final Controller choice in `results/phase3/final_controller_selection.json`. It intentionally creates no Phase 3A Test predictions.

## 3. Frozen Test Evaluation

Run only after `train-dev` completed and inspect both selection manifests:

```bash
cat results/phase3/coverage_lambda_selection.json
cat results/phase3/final_controller_selection.json
bash run_phase3a.sh test 2>&1 | tee results/phase3_test.log
```

This evaluates all predeclared variants, collects mean and sample standard deviation across seeds, runs 2,000 paired question-level bootstrap replicates, and writes the Hard Partial analysis. Every sampled question keeps all of its stage decisions together.

Within Easy/Medium/Hard Continue buckets, AUROC is mathematically undefined because each bucket has one class. The report therefore records `N/A` and additionally computes each Continue bucket versus all truly Sufficient examples.

## 4. Generation Dev Gate

```bash
export LLM_MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct
bash run_phase3a.sh generation-dev 2>&1 | tee results/phase3_generation_dev.log
```

Review all 50 rows in `results/phase3/generation/dev/manual_audit_50.csv`. Fill `human_extraction_correct`, `human_scoring_reasonable`, `human_notes`, and `reviewer`. Both binary review fields must be `yes` for the Test gate to open. If the gate is rejected, diagnose and revise only on Dev, then create a new versioned generation config instead of overwriting the old results.

## 5. Generation Test

```bash
bash run_phase3a.sh generation-test 2>&1 | tee results/phase3_generation_test.log
```

The command hashes the approved Dev configuration before Test generation. Any prompt, decoding, extraction, or normalization change invalidates approval.

## Output Checklist

- All Controller runs contain `run_manifest.json`, `dev_predictions.jsonl`, and frozen Test `original_predictions.jsonl`.
- Auxiliary runs also save `true_coverage`, `predicted_coverage`, MAE, and Spearman correlation.
- `results/phase3/seed_results.csv` contains per-seed results.
- `results/phase3/multi_seed_summary.csv` contains mean and standard deviation.
- `results/phase3/ablation_summary.csv` compares the baseline, auxiliary, Hard Partial sampling, and combined variants.
- `results/phase3/sampling_summary.csv` compares natural, balanced Stop/Continue, and Hard-Partial-aware sampling.
- `results/phase3/bootstrap/bootstrap_replicates.jsonl` preserves all bootstrap draws' metrics.
- `results/phase3/hard_partial/*_samples.jsonl` preserves per-sample bucket assignments.
- Generation preserves raw output, extracted answer, normalization inputs, and per-example EM/F1.

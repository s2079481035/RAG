# Phase 3B Runbook

Phase 3A is frozen. Do not retrain it or rerun Phase 2. Run every GPU step on one explicitly selected GPU and keep Dev and Test in separate commands.

## Server setup

```bash
cd ~/RAG/RAG_paper
git switch research/sufficiency-phase3b
git pull --ff-only

export PYTHON_BIN=python3.12
export GPU_ID=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export LLM_MODEL_PATH=/home/sunjb/RAG/ekrag/ekrag/models/Qwen2.5-7B-Instruct
```

`GPU_ID=1` makes physical GPU 1 the only visible device; PyTorch correctly reports it as `cuda:0` inside the process.

## Dev-only phase

Start with the CPU preparation, calibration, and threshold selection:

```bash
bash run_phase3b.sh prepare-dev
```

Then run the expensive frozen stage generations and fresh latency benchmarks sequentially:

```bash
tmux new -s phase3b-dev
bash run_phase3b.sh generation-dev 2>&1 | tee results/phase3b_generation_dev.log
bash run_phase3b.sh benchmark-dev 2>&1 | tee results/phase3b_benchmark_dev.log
bash run_phase3b.sh evaluate-dev
```

Inspect the Dev tables and selected thresholds before freezing Test:

```bash
cat results/phase3b/calibration_metrics.csv
cat results/phase3b/selected_thresholds.json
cat results/phase3b/dev_policy_end_to_end.csv
python3.12 scripts/freeze_phase3b_test_gate.py
cat results/phase3b/test_gate.json
```

Do not use `--force` casually. Existing outputs deliberately block accidental overwrites.

## Frozen Test phase

After `test_gate.json` exists, run Test exactly once for this protocol:

```bash
tmux new -s phase3b-test
bash run_phase3b.sh generation-test 2>&1 | tee results/phase3b_generation_test.log
bash run_phase3b.sh benchmark-test 2>&1 | tee results/phase3b_benchmark_test.log
bash run_phase3b.sh evaluate-test
bash run_phase3b.sh bootstrap-test
bash run_phase3b.sh analyze
```

Monitor without attaching:

```bash
tmux ls
pgrep -af '[g]enerate_phase3b|[b]enchmark_phase3b'
nvidia-smi -i "$GPU_ID"
tail -f results/phase3b_generation_dev.log
```

## Final checks

```bash
python3.12 -m unittest discover -s tests -v
git status --short --untracked-files=all
```

Large per-sample JSONL and logs are ignored by Git. Commit configs, scripts, compact CSV/JSON manifests, reports, and figures after reviewing their sizes.

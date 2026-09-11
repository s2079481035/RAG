# Phase 4 Server Runbook

Run each gate separately. Do not launch `heldout` commands until all Train-derived choices and their manifests are frozen.

## 0. Environment

```bash
cd ~/RAG/RAG_paper
git switch research/sufficiency-phase4
git pull --ff-only

export PYTHON_BIN=python3.12
export GPU_ID=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

git show -s --format='%h %D %s' phase3b_final
```

The final line must show tag `phase3b_final` on commit `1ed762c`.

Create the content-level freeze manifest before processing 2Wiki:

```bash
$PYTHON_BIN scripts/audit_phase4_freeze.py \
  --checkpoint experiments/phase3a/lambda_dev/lambda_0.1_seed42/best.pt \
  --dense-model /path/to/bge-large-en-v1.5 \
  --reranker-model /path/to/bge-reranker-v2-m3 \
  --generator-model /home/sunjb/RAG/ekrag/ekrag/models/Qwen2.5-7B-Instruct

cat docs/phase4/freeze_audit.md
```

## 1. Official Data

Place the official files at:

```text
data/2wiki/source/train.json
data/2wiki/source/dev.json
```

Do not use official unlabeled Test. Record where the archive came from and retain its checksum.

## 2. Prepare and Audit

```bash
$PYTHON_BIN -m unittest discover -s tests -p 'test_phase4*.py' -v

$PYTHON_BIN scripts/prepare_phase4_2wiki.py \
  --tokenizer /path/to/bge-large-en-v1.5

cat docs/phase4/2wiki_dataset_audit.md
cat docs/phase4/2wiki_corpus_audit.md
```

Stop if the gold-to-chunk mapping rate is not exactly 1.0.

## 3. Index

```bash
$PYTHON_BIN scripts/build_phase2_index.py \
  --config configs/phase4/protocol.json \
  --variant sentence_256 \
  --model /path/to/bge-large-en-v1.5
```

## 4. Retrieval

Before any full split, run an isolated throughput check. Its deterministic subset is
for runtime and memory estimation only and must not be reported as an experiment result:

```bash
$PYTHON_BIN scripts/run_phase2_retrieval.py \
  --config configs/phase4/protocol.json \
  --variant sentence_256 \
  --splits dev_policy \
  --limit-per-split 100 \
  --output-label dev_policy_100 \
  --bm25-workers 1 \
  --dense-model /path/to/bge-large-en-v1.5 \
  --reranker-model /path/to/bge-reranker-v2-m3
```

Before using parallel BM25 for formal retrieval, repeat the same smoke subset with a
new output label, `--bm25-workers 8`, and `--dense-search-batch-size 1`. Compare every
saved ranking and score after excluding latency fields. Parallel scheduling is accepted
only if the outputs match. Batched FAISS search is a separate throughput option and must
pass the same ranking-ID and floating-score tolerance check before formal use.

```bash
$PYTHON_BIN scripts/compare_phase4_retrieval_runs.py \
  results/phase4/2wiki/retrieval_smoke/dev_policy_100/sentence_256/dev_policy.jsonl \
  results/phase4/2wiki/retrieval_smoke/dev_policy_100_parallel8/sentence_256/dev_policy.jsonl \
  --atol 1e-6 \
  --output results/phase4/2wiki/retrieval_smoke/dev_policy_100_parallel8/parity.json
```

After testing both parallel BM25 and batched FAISS, freeze the execution-only choice:

```bash
$PYTHON_BIN scripts/select_phase4_retrieval_execution.py
cat docs/phase4/retrieval_execution_audit.md
```

First run Train-derived splits:

```bash
$PYTHON_BIN scripts/run_phase2_retrieval.py \
  --config configs/phase4/protocol.json \
  --variant sentence_256 \
  --splits train_core,dev_calibration,dev_policy \
  --dense-model /path/to/bge-large-en-v1.5 \
  --reranker-model /path/to/bge-reranker-v2-m3
```

The formal retrieval baseline on labeled official Dev is evaluation-only:

```bash
$PYTHON_BIN scripts/run_phase2_retrieval.py \
  --config configs/phase4/protocol.json \
  --variant sentence_256 \
  --splits heldout \
  --dense-model /path/to/bge-large-en-v1.5 \
  --reranker-model /path/to/bge-reranker-v2-m3

$PYTHON_BIN scripts/evaluate_phase2_retrieval.py \
  --config configs/phase4/protocol.json \
  --variants sentence_256 \
  --split heldout
```

## 5. Controller Examples

```bash
$PYTHON_BIN scripts/build_phase2_controller_data.py \
  --retrieval-config configs/phase4/protocol.json \
  --controller-config configs/phase4/protocol.json \
  --variant sentence_256 \
  --splits train_core,dev_calibration,dev_policy \
  --report docs/phase4/2wiki_trajectory_audit_train_derived.md
```

Inspect `docs/phase4/2wiki_trajectory_audit_train_derived.md` before training. Build
the held-out Controller data later with a distinct
`docs/phase4/2wiki_trajectory_audit_heldout.md` report; do not overwrite this audit.

Run the retrieval-ceiling analysis only on Train-derived splits. Its recoverability
fields are evaluation diagnostics and are never training inputs or labels:

```bash
$PYTHON_BIN scripts/analyze_phase4_retrieval_ceiling.py
cat docs/phase4/2wiki_retrieval_ceiling_analysis.md
```

After a Controller has produced frozen `dev_policy` predictions, re-run with one or
more `--prediction LABEL PATH THRESHOLD_OR_SAVED` arguments to add operational FSR,
RFSR, and unrecoverable early-stop diagnostics. Do not supply held-out predictions.

Fit the simple score baseline only on `dev_policy`:

```bash
$PYTHON_BIN scripts/evaluate_phase4_score_threshold.py fit
cat results/phase4/score_threshold/selected_thresholds.json
```

Do not run its `evaluate` mode until the held-out gate is frozen.

## 6. Core In-domain Training

Run only the two preregistered core groups for seeds 42, 123, and 2026:

```bash
for SEED in 42 123 2026; do
  $PYTHON_BIN scripts/train_phase3a_controller.py \
    --config configs/phase4/controller.json \
    --baseline query_stage \
    --representation score_aware_packing \
    --seed "$SEED" \
    --coverage-lambda 0 \
    --sampling natural \
    --run-dir "experiments/phase4/2wiki/query_stage/seed${SEED}"

  $PYTHON_BIN scripts/train_phase3a_controller.py \
    --config configs/phase4/controller.json \
    --baseline query_evidence \
    --representation score_aware_packing \
    --seed "$SEED" \
    --coverage-lambda 0.1 \
    --sampling natural \
    --run-dir "experiments/phase4/2wiki/final_evidence_aware/seed${SEED}"
done
```

Do not evaluate `heldout` until the in-domain calibration and policy-selection manifests are committed. Later runbook sections are added only after those gates are implemented and tested.

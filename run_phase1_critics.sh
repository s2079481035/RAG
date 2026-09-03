#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
GPU_ID="${GPU_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

QUERY_ONLY_RUN="experiments/sufficiency_critic/query_only_seed42"
QUERY_EVIDENCE_RUN="experiments/sufficiency_critic/query_evidence_seed42"

if [[ ! -f data/sufficiency/train.jsonl || ! -f data/sufficiency/dev.jsonl || ! -f data/sufficiency/test.jsonl ]]; then
  "$PYTHON_BIN" scripts/build_sufficiency_dataset.py --jsonl-only
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_sufficiency_critic.py \
  --input-mode query_only \
  --run-dir "$QUERY_ONLY_RUN"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_sufficiency_critic.py \
  --input-mode query_evidence \
  --run-dir "$QUERY_EVIDENCE_RUN"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluate_sufficiency_critic.py \
  --query-only-run "$QUERY_ONLY_RUN" \
  --query-evidence-run "$QUERY_EVIDENCE_RUN"

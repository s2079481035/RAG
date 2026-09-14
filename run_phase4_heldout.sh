#!/usr/bin/env bash
set -euo pipefail

: "${RUN_HELDOUT:?Set RUN_HELDOUT=YES after committing the human authorization manifest}"
if [[ "$RUN_HELDOUT" != "YES" ]]; then
  echo "RUN_HELDOUT must equal YES" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DENSE_MODEL_PATH="${DENSE_MODEL_PATH:-/home/sunjb/RAG/ekrag/ekrag/models/bge-large-en-v1.5}"
RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:-/home/sunjb/RAG/ekrag/ekrag/models/bge-reranker-v2-m3}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/home/sunjb/RAG/ekrag/ekrag/models/Qwen2.5-7B-Instruct}"
export LLM_MODEL_PATH

RUN_MANIFEST="results/phase4/heldout_run_manifest.json"
if [[ -s "$RUN_MANIFEST" ]]; then
  python3.12 scripts/start_phase4_heldout.py --resume
else
  python3.12 scripts/start_phase4_heldout.py
fi

RETRIEVAL="results/phase4/2wiki/retrieval/sentence_256/heldout.jsonl"
if [[ ! -s "$RETRIEVAL" ]]; then
  echo "[run] frozen retrieval on heldout"
  python3.12 scripts/run_phase2_retrieval.py \
    --config configs/phase4/protocol.json \
    --variant sentence_256 \
    --splits heldout \
    --bm25-workers 8 \
    --dense-search-batch-size 1 \
    --dense-model "$DENSE_MODEL_PATH" \
    --reranker-model "$RERANKER_MODEL_PATH"
else
  echo "[resume] frozen heldout retrieval exists"
fi

CONTROLLER_DATA="data/2wiki/controller/sentence_256/heldout.jsonl"
if [[ ! -s "$CONTROLLER_DATA" ]]; then
  echo "[run] build heldout cumulative Controller trajectories"
  python3.12 scripts/build_phase2_controller_data.py \
    --retrieval-config configs/phase4/protocol.json \
    --controller-config configs/phase4/controller.json \
    --variant sentence_256 \
    --splits heldout \
    --report docs/phase4/2wiki_heldout_trajectory_audit.md
else
  echo "[resume] heldout Controller trajectories exist"
fi

CONTROLLER_PRED="experiments/phase4/2wiki/final_evidence_aware/seed42/evaluation/heldout/original_predictions.jsonl"
if [[ ! -s "$CONTROLLER_PRED" ]]; then
  echo "[run] frozen evidence-aware Controller on heldout"
  python3.12 scripts/evaluate_phase3a_controller.py \
    --run-dir experiments/phase4/2wiki/final_evidence_aware/seed42 \
    --split heldout
else
  echo "[resume] heldout Controller predictions exist"
fi

for SEED in 42 123 2026; do
  ROUTER_PRED="experiments/phase4/2wiki/adaptive_rag_router/seed${SEED}/evaluation/heldout/predictions.jsonl"
  if [[ ! -s "$ROUTER_PRED" ]]; then
    echo "[run] frozen Adaptive Router seed ${SEED} on heldout"
    python3.12 scripts/evaluate_phase4_heldout_router.py \
      --run-dir "experiments/phase4/2wiki/adaptive_rag_router/seed${SEED}"
  else
    echo "[resume] heldout Router seed ${SEED} exists"
  fi
done

SCORE_MANIFEST="results/phase4/score_threshold/heldout_manifest.json"
if [[ ! -s "$SCORE_MANIFEST" ]]; then
  echo "[run] frozen score-threshold baseline on heldout"
  python3.12 scripts/evaluate_phase4_score_threshold.py evaluate
else
  echo "[resume] heldout score-threshold baseline exists"
fi

JUDGE_512_MANIFEST="results/phase4/2wiki/llm_judge/heldout/judge_512_manifest.json"
if [[ ! -s "$JUDGE_512_MANIFEST" ]]; then
  echo "[run] frozen Judge-512 on heldout"
  python3.12 scripts/run_phase4_heldout_llm_judge.py \
    --representations judge_512 \
    --model "$LLM_MODEL_PATH" \
    --packing-tokenizer "$DENSE_MODEL_PATH"
else
  echo "[resume] heldout Judge-512 exists"
fi

JUDGE_FULL_MANIFEST="results/phase4/2wiki/llm_judge/heldout/judge_full_manifest.json"
if [[ ! -s "$JUDGE_FULL_MANIFEST" ]]; then
  echo "[run] frozen Judge-Full on heldout"
  python3.12 scripts/run_phase4_heldout_llm_judge.py \
    --representations judge_full \
    --model "$LLM_MODEL_PATH" \
    --packing-tokenizer "$DENSE_MODEL_PATH"
else
  echo "[resume] heldout Judge-Full exists"
fi

GENERATION_MANIFEST="results/phase4/2wiki/heldout/generation/generation_manifest.json"
if [[ ! -s "$GENERATION_MANIFEST" ]]; then
  echo "[run] frozen three-stage answer generation on heldout"
  python3.12 scripts/generate_phase4_heldout_stage_answers.py --model "$LLM_MODEL_PATH"
else
  echo "[resume] heldout stage generations exist"
fi

echo "[run] final frozen heldout analysis"
python3.12 scripts/analyze_phase4_heldout.py
echo "[complete] Phase 4 heldout evaluation"

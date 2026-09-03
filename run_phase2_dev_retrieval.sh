#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
GPU_ID="${GPU_ID:-0}"
SOURCE_PARQUET="${SOURCE_PARQUET:-data/source/hotpotqa_distractor_validation.parquet}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

"$PYTHON_BIN" -m unittest discover -s tests -v

if [[ ! -f data/phase2/chunk_manifest.json \
   || ! -f data/phase2/chunks/sentence_128.jsonl \
   || ! -f data/phase2/chunks/sentence_256.jsonl ]]; then
  "$PYTHON_BIN" scripts/build_phase2_chunks.py --source-parquet "$SOURCE_PARQUET"
fi

for variant in sentence_128 sentence_256; do
  if [[ ! -f "data/phase2/indices/$variant/index_manifest.json" ]]; then
    "$PYTHON_BIN" scripts/build_phase2_index.py --variant "$variant"
  fi
  if [[ ! -f "results/phase2/retrieval/$variant/dev.jsonl" ]]; then
    "$PYTHON_BIN" scripts/run_phase2_retrieval.py --variant "$variant" --splits dev
  fi
done

"$PYTHON_BIN" scripts/evaluate_phase2_retrieval.py \
  --variants sentence_128,sentence_256 --split dev

echo "Review results/phase2/retrieval_eval/dev/retrieval_summary.md"
echo "Then record the pre-registered choice with:"
echo "  $PYTHON_BIN scripts/select_phase2_chunk_variant.py"

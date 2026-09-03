#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
GPU_ID="${GPU_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SELECTED_VARIANT="$($PYTHON_BIN -c 'import json; print(json.load(open("configs/phase2/chunk_retrieval.json"))["retrieval_evaluation"]["selected_variant_after_dev"] or "")')"
if [[ -z "$SELECTED_VARIANT" ]]; then
  echo "Chunk variant is not selected. Run the dev retrieval stage and select_phase2_chunk_variant.py first." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/run_phase2_retrieval.py \
  --variant "$SELECTED_VARIANT" --splits train,test
"$PYTHON_BIN" scripts/evaluate_phase2_retrieval.py \
  --variants "$SELECTED_VARIANT" --split test
"$PYTHON_BIN" scripts/build_phase2_controller_data.py \
  --variant "$SELECTED_VARIANT" --splits train,dev,test

BASE="experiments/phase2/controller/$SELECTED_VARIANT"

"$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
  --baseline query_only --run-dir "$BASE/query_only_cumulative"
"$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
  --baseline query_stage --run-dir "$BASE/query_stage_cumulative"
"$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
  --baseline query_stage_stats --run-dir "$BASE/query_stage_stats_cumulative"
"$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
  --baseline evidence_only --representation uniform_packing \
  --run-dir "$BASE/evidence_only_uniform_cumulative"

for representation in concat_truncate uniform_packing score_aware_packing; do
  "$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
    --baseline query_evidence --representation "$representation" \
    --run-dir "$BASE/query_evidence_${representation}_cumulative"
done

"$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
  --baseline query_evidence --representation uniform_packing --evidence-mode raw \
  --run-dir "$BASE/query_evidence_uniform_packing_raw"

for aggregation in mean max mean_max; do
  "$PYTHON_BIN" scripts/train_phase2_hierarchical.py --variant "$SELECTED_VARIANT" \
    --aggregation "$aggregation" \
    --run-dir "$BASE/query_evidence_hierarchical_${aggregation}_cumulative"
done

STANDARD_RUNS=(
  "$BASE/query_only_cumulative"
  "$BASE/query_stage_cumulative"
  "$BASE/query_stage_stats_cumulative"
  "$BASE/evidence_only_uniform_cumulative"
  "$BASE/query_evidence_concat_truncate_cumulative"
  "$BASE/query_evidence_uniform_packing_cumulative"
  "$BASE/query_evidence_score_aware_packing_cumulative"
  "$BASE/query_evidence_uniform_packing_raw"
)
HIERARCHICAL_RUNS=(
  "$BASE/query_evidence_hierarchical_mean_cumulative"
  "$BASE/query_evidence_hierarchical_max_cumulative"
  "$BASE/query_evidence_hierarchical_mean_max_cumulative"
)

for run in "${STANDARD_RUNS[@]}"; do
  "$PYTHON_BIN" scripts/evaluate_phase2_controller.py --run-dir "$run" --split dev
  "$PYTHON_BIN" scripts/evaluate_phase2_controller.py --run-dir "$run" --split test
done

for run in \
  "$BASE/query_evidence_concat_truncate_cumulative" \
  "$BASE/query_evidence_uniform_packing_cumulative" \
  "$BASE/query_evidence_score_aware_packing_cumulative"; do
  "$PYTHON_BIN" scripts/evaluate_phase2_controller.py --run-dir "$run" --split dev \
    --conditions evidence_order_shuffle,cross_question_evidence_swap,stage_metadata_removal,title_only_evidence
done

for run in "${HIERARCHICAL_RUNS[@]}"; do
  "$PYTHON_BIN" scripts/evaluate_phase2_hierarchical.py --run-dir "$run" --split dev
  "$PYTHON_BIN" scripts/evaluate_phase2_hierarchical.py --run-dir "$run" --split test
done

"$PYTHON_BIN" scripts/collect_phase2_results.py --split test \
  --runs "${STANDARD_RUNS[@]}" "${HIERARCHICAL_RUNS[@]}"
"$PYTHON_BIN" scripts/build_manual_sufficiency_audit.py \
  --run-dir "$BASE/query_evidence_uniform_packing_cumulative" --count 180

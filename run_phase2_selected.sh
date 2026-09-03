#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
GPU_ID="${GPU_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SELECTED_VARIANT="$("$PYTHON_BIN" -c 'import json; print(json.load(open("configs/phase2/chunk_retrieval.json"))["retrieval_evaluation"]["selected_variant_after_dev"] or "")')"
if [[ -z "$SELECTED_VARIANT" ]]; then
  echo "Chunk variant is not selected. Run the dev retrieval stage and select_phase2_chunk_variant.py first." >&2
  exit 1
fi

run_once() {
  local label="$1"
  local completion_marker="$2"
  shift 2
  if [[ -f "$completion_marker" ]]; then
    echo "Skipping completed $label: $completion_marker"
    return
  fi
  "$@"
}

run_training_once() {
  local run_dir="$1"
  shift
  local manifest="$run_dir/run_manifest.json"
  if [[ -f "$manifest" ]]; then
    local status
    status="$("$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "$manifest")"
    if [[ "$status" == "complete" ]]; then
      echo "Skipping completed training run: $run_dir"
      return
    fi
    echo "Incomplete training run requires explicit recovery: $run_dir (status=$status)" >&2
    exit 1
  fi
  if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Training directory is non-empty without a manifest: $run_dir" >&2
    exit 1
  fi
  "$@"
}

run_once "train/test retrieval" \
  "results/phase2/retrieval/$SELECTED_VARIANT/run_manifest_train_test.json" \
  "$PYTHON_BIN" scripts/run_phase2_retrieval.py \
    --variant "$SELECTED_VARIANT" --splits train,test
run_once "test retrieval evaluation" \
  "results/phase2/retrieval_eval/test/evaluation_manifest.json" \
  "$PYTHON_BIN" scripts/evaluate_phase2_retrieval.py \
    --variants "$SELECTED_VARIANT" --split test
run_once "Controller dataset" \
  "data/phase2/controller/$SELECTED_VARIANT/dataset_manifest_train_dev_test.json" \
  "$PYTHON_BIN" scripts/build_phase2_controller_data.py \
    --variant "$SELECTED_VARIANT" --splits train,dev,test

BASE="experiments/phase2/controller/$SELECTED_VARIANT"

run_training_once "$BASE/query_only_cumulative" \
  "$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
    --baseline query_only --run-dir "$BASE/query_only_cumulative"
run_training_once "$BASE/query_stage_cumulative" \
  "$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
    --baseline query_stage --run-dir "$BASE/query_stage_cumulative"
run_training_once "$BASE/query_stage_stats_cumulative" \
  "$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
    --baseline query_stage_stats --run-dir "$BASE/query_stage_stats_cumulative"
run_training_once "$BASE/evidence_only_uniform_cumulative" \
  "$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
    --baseline evidence_only --representation uniform_packing \
    --run-dir "$BASE/evidence_only_uniform_cumulative"

for representation in concat_truncate uniform_packing score_aware_packing; do
  run_training_once "$BASE/query_evidence_${representation}_cumulative" \
    "$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
      --baseline query_evidence --representation "$representation" \
      --run-dir "$BASE/query_evidence_${representation}_cumulative"
done

run_training_once "$BASE/query_evidence_uniform_packing_raw" \
  "$PYTHON_BIN" scripts/train_phase2_controller.py --variant "$SELECTED_VARIANT" \
    --baseline query_evidence --representation uniform_packing --evidence-mode raw \
    --run-dir "$BASE/query_evidence_uniform_packing_raw"

for aggregation in mean max mean_max; do
  run_training_once "$BASE/query_evidence_hierarchical_${aggregation}_cumulative" \
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
  run_once "dev evaluation for $run" \
    "$run/evaluation/dev/evaluation_manifest_original.json" \
    "$PYTHON_BIN" scripts/evaluate_phase2_controller.py --run-dir "$run" --split dev
  run_once "test evaluation for $run" \
    "$run/evaluation/test/evaluation_manifest_original.json" \
    "$PYTHON_BIN" scripts/evaluate_phase2_controller.py --run-dir "$run" --split test
done

for run in \
  "$BASE/query_evidence_concat_truncate_cumulative" \
  "$BASE/query_evidence_uniform_packing_cumulative" \
  "$BASE/query_evidence_score_aware_packing_cumulative"; do
  run_once "counterfactual evaluation for $run" \
    "$run/evaluation/dev/evaluation_manifest_evidence_order_shuffle_cross_question_evidence_swap_stage_metadata_removal_title_only_evidence.json" \
    "$PYTHON_BIN" scripts/evaluate_phase2_controller.py --run-dir "$run" --split dev \
      --conditions evidence_order_shuffle,cross_question_evidence_swap,stage_metadata_removal,title_only_evidence
done

for run in "${HIERARCHICAL_RUNS[@]}"; do
  run_once "hierarchical dev evaluation for $run" \
    "$run/evaluation/dev/evaluation_manifest_original.json" \
    "$PYTHON_BIN" scripts/evaluate_phase2_hierarchical.py --run-dir "$run" --split dev
  run_once "hierarchical test evaluation for $run" \
    "$run/evaluation/test/evaluation_manifest_original.json" \
    "$PYTHON_BIN" scripts/evaluate_phase2_hierarchical.py --run-dir "$run" --split test
done

run_once "Phase 2 result collection" "results/phase2/collection_manifest.json" \
  "$PYTHON_BIN" scripts/collect_phase2_results.py --split test \
    --runs "${STANDARD_RUNS[@]}" "${HIERARCHICAL_RUNS[@]}"
run_once "manual sufficiency audit" "docs/manual_sufficiency_audit.csv" \
  "$PYTHON_BIN" scripts/build_manual_sufficiency_audit.py \
    --run-dir "$BASE/query_evidence_uniform_packing_cumulative" --count 180

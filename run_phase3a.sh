#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CONFIG="${PHASE3A_CONFIG:-configs/phase3a/controller.json}"
GEN_CONFIG="${PHASE3A_GENERATION_CONFIG:-configs/phase3a/generation.json}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

MODEL_ARGS=()
if [[ -n "${CONTROLLER_MODEL_PATH:-}" ]]; then
  MODEL_ARGS=(--model "$CONTROLLER_MODEL_PATH")
fi

run_once() {
  local description="$1"
  local marker="$2"
  shift 2
  if [[ -e "$marker" ]]; then
    echo "[skip] $description: $marker"
    return
  fi
  echo "[run] $description"
  "$@"
}

train_controller() {
  local run_dir="$1"
  local baseline="$2"
  local representation="$3"
  local seed="$4"
  local coverage_lambda="$5"
  local sampling="$6"
  if [[ -f "$run_dir/run_manifest.json" ]]; then
    if "$PYTHON_BIN" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "complete" else 1)' "$run_dir/run_manifest.json"; then
      echo "[skip] completed training: $run_dir"
      return
    fi
    echo "Incomplete run directory requires inspection before retry: $run_dir" >&2
    exit 2
  fi
  echo "[run] train $run_dir"
  "$PYTHON_BIN" scripts/train_phase3a_controller.py \
    --config "$CONFIG" \
    --baseline "$baseline" \
    --representation "$representation" \
    --seed "$seed" \
    --coverage-lambda "$coverage_lambda" \
    --sampling "$sampling" \
    --run-dir "$run_dir" \
    "${MODEL_ARGS[@]}"
}

evaluate_test() {
  local run_dir="$1"
  run_once "test $run_dir" "$run_dir/evaluation/test/evaluation_manifest_original.json" \
    "$PYTHON_BIN" scripts/evaluate_phase3a_controller.py --run-dir "$run_dir" --split test
}

selected_lambda() {
  "$PYTHON_BIN" -c 'import json; print(json.load(open("results/phase3/coverage_lambda_selection.json"))["selected_coverage_lambda"])'
}

core_runs() {
  local seed
  for seed in 42 123 2026; do
    echo "experiments/phase3a/core/query_stage/seed${seed}"
    echo "experiments/phase3a/core/query_evidence_concat/seed${seed}"
    echo "experiments/phase3a/core/score_aware_baseline/seed${seed}"
  done
}

selected_aux_runs() {
  local lambda="$1"
  echo "experiments/phase3a/lambda_dev/lambda_${lambda}_seed42"
  echo "experiments/phase3a/ablation/coverage_aux/seed123"
  echo "experiments/phase3a/ablation/coverage_aux/seed2026"
}

hard_sampling_runs() {
  local seed
  for seed in 42 123 2026; do
    echo "experiments/phase3a/ablation/hard_partial_sampling/seed${seed}"
  done
}

combined_runs() {
  local seed
  for seed in 42 123 2026; do
    echo "experiments/phase3a/ablation/coverage_aux_hard_partial/seed${seed}"
  done
}

balanced_runs() {
  local seed
  for seed in 42 123 2026; do
    echo "experiments/phase3a/sampling/balanced_stop_continue/seed${seed}"
  done
}

close_phase2() {
  local variant base
  variant="$($PYTHON_BIN -c 'import json; print(json.load(open("configs/phase2/chunk_retrieval.json"))["retrieval_evaluation"]["selected_variant_after_dev"])')"
  base="experiments/phase2/controller/$variant"
  phase2_runs=(
    "$base/query_only_cumulative"
    "$base/query_stage_cumulative"
    "$base/query_stage_stats_cumulative"
    "$base/evidence_only_uniform_cumulative"
    "$base/query_evidence_concat_truncate_cumulative"
    "$base/query_evidence_uniform_packing_cumulative"
    "$base/query_evidence_score_aware_packing_cumulative"
    "$base/query_evidence_uniform_packing_raw"
    "$base/query_evidence_hierarchical_mean_cumulative"
    "$base/query_evidence_hierarchical_max_cumulative"
    "$base/query_evidence_hierarchical_mean_max_cumulative"
  )
  echo "[run] refresh compact Phase 2 summaries only"
  "$PYTHON_BIN" scripts/collect_phase2_results.py --split test --force --runs "${phase2_runs[@]}"
}

train_dev() {
  local seed lambda
  for seed in 42 123 2026; do
    train_controller "experiments/phase3a/core/query_stage/seed${seed}" query_stage score_aware_packing "$seed" 0 natural
    train_controller "experiments/phase3a/core/query_evidence_concat/seed${seed}" query_evidence concat_truncate "$seed" 0 natural
    train_controller "experiments/phase3a/core/score_aware_baseline/seed${seed}" query_evidence score_aware_packing "$seed" 0 natural
  done

  for lambda in 0.1 0.3 1.0; do
    train_controller "experiments/phase3a/lambda_dev/lambda_${lambda}_seed42" query_evidence score_aware_packing 42 "$lambda" natural
  done
  run_once "select coverage lambda on Dev" "results/phase3/coverage_lambda_selection.json" \
    "$PYTHON_BIN" scripts/select_phase3a_coverage_lambda.py \
      --runs \
        experiments/phase3a/lambda_dev/lambda_0.1_seed42 \
        experiments/phase3a/lambda_dev/lambda_0.3_seed42 \
        experiments/phase3a/lambda_dev/lambda_1.0_seed42
  lambda="$(selected_lambda)"
  echo "[selected on Dev] coverage lambda=$lambda"

  train_controller experiments/phase3a/ablation/coverage_aux/seed123 query_evidence score_aware_packing 123 "$lambda" natural
  train_controller experiments/phase3a/ablation/coverage_aux/seed2026 query_evidence score_aware_packing 2026 "$lambda" natural
  for seed in 42 123 2026; do
    train_controller "experiments/phase3a/sampling/balanced_stop_continue/seed${seed}" query_evidence score_aware_packing "$seed" 0 balanced_stop_continue
    train_controller "experiments/phase3a/ablation/hard_partial_sampling/seed${seed}" query_evidence score_aware_packing "$seed" 0 hard_partial_aware
    train_controller "experiments/phase3a/ablation/coverage_aux_hard_partial/seed${seed}" query_evidence score_aware_packing "$seed" "$lambda" hard_partial_aware
  done

  mapfile -t auxiliary < <(selected_aux_runs "$lambda")
  mapfile -t hard < <(hard_sampling_runs)
  mapfile -t combined < <(combined_runs)
  run_once "select final Controller on multi-seed Dev" "results/phase3/final_controller_selection.json" \
    "$PYTHON_BIN" scripts/select_phase3a_final_controller.py \
      --runs \
        experiments/phase3a/core/score_aware_baseline/seed42 \
        experiments/phase3a/core/score_aware_baseline/seed123 \
        experiments/phase3a/core/score_aware_baseline/seed2026 \
        "${auxiliary[@]}" "${hard[@]}" "${combined[@]}"
}

test_and_collect() {
  if [[ ! -f results/phase3/final_controller_selection.json ]]; then
    echo "Run '$0 train-dev' before any Phase 3A test evaluation." >&2
    exit 2
  fi
  local lambda run final_seed42
  lambda="$(selected_lambda)"
  mapfile -t core < <(core_runs)
  mapfile -t auxiliary < <(selected_aux_runs "$lambda")
  mapfile -t hard < <(hard_sampling_runs)
  mapfile -t combined < <(combined_runs)
  mapfile -t balanced < <(balanced_runs)
  all_runs=("${core[@]}" "${auxiliary[@]}" "${balanced[@]}" "${hard[@]}" "${combined[@]}")
  for run in "${all_runs[@]}"; do
    evaluate_test "$run"
  done
  run_once "collect Phase 3A multi-seed results" "results/phase3/collection_manifest.json" \
    "$PYTHON_BIN" scripts/collect_phase3a_results.py --runs "${all_runs[@]}"

  final_seed42="$($PYTHON_BIN -c 'import json; d=json.load(open("results/phase3/final_controller_selection.json")); print(next(r["run_dir"] for r in d["selected_runs"] if r["seed"] == 42))')"
  run_once "question-level paired bootstrap" "results/phase3/bootstrap/bootstrap_manifest.json" \
    "$PYTHON_BIN" scripts/bootstrap_phase3a.py \
      --prediction "QueryStage=experiments/phase3a/core/query_stage/seed42/evaluation/test/original_predictions.jsonl" \
      --prediction "ScoreAwareBaseline=experiments/phase3a/core/score_aware_baseline/seed42/evaluation/test/original_predictions.jsonl" \
      --prediction "FinalController=${final_seed42}/evaluation/test/original_predictions.jsonl" \
      --compare QueryStage,FinalController --replicates 2000
  run_once "Hard Partial frozen-test analysis" "results/phase3/hard_partial/analysis_manifest.json" \
    "$PYTHON_BIN" scripts/analyze_phase3a_hard_partial.py \
      --prediction "QueryStage=experiments/phase3a/core/query_stage/seed42/evaluation/test/original_predictions.jsonl" \
      --prediction "ScoreAwareBaseline=experiments/phase3a/core/score_aware_baseline/seed42/evaluation/test/original_predictions.jsonl" \
      --prediction "FinalController=${final_seed42}/evaluation/test/original_predictions.jsonl"
}

case "${1:-}" in
  phase2-close)
    close_phase2
    ;;
  manual-audit)
    "$PYTHON_BIN" scripts/analyze_manual_sufficiency_audit.py
    ;;
  train-dev)
    train_dev
    ;;
  test)
    test_and_collect
    ;;
  generation-dev)
    run_once "generate Dev answers" "results/phase3/generation/dev/generation_manifest.json" \
      "$PYTHON_BIN" scripts/generate_phase3a.py --config "$GEN_CONFIG" --split dev
    run_once "evaluate Dev answers" "results/phase3/generation/dev/evaluation_metrics.json" \
      "$PYTHON_BIN" scripts/evaluate_phase3a_generation.py \
        --input results/phase3/generation/dev/generated.jsonl
    run_once "build 50-example Dev generation audit" "results/phase3/generation/dev/manual_audit_50.manifest.json" \
      "$PYTHON_BIN" scripts/build_phase3a_generation_audit.py --config "$GEN_CONFIG"
    ;;
  generation-test)
    if [[ -f results/phase3/generation/dev/test_gate_approval.json ]] && \
      "$PYTHON_BIN" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "approved" else 1)' results/phase3/generation/dev/test_gate_approval.json; then
      echo "[skip] approved Dev generation audit"
    else
      "$PYTHON_BIN" scripts/approve_phase3a_generation_audit.py --config "$GEN_CONFIG" --force
    fi
    run_once "generate Test answers" "results/phase3/generation/test/generation_manifest.json" \
      "$PYTHON_BIN" scripts/generate_phase3a.py --config "$GEN_CONFIG" --split test \
        --approval results/phase3/generation/dev/test_gate_approval.json
    run_once "evaluate Test answers" "results/phase3/generation/test/evaluation_metrics.json" \
      "$PYTHON_BIN" scripts/evaluate_phase3a_generation.py \
        --input results/phase3/generation/test/generated.jsonl
    ;;
  *)
    echo "Usage: $0 {phase2-close|manual-audit|train-dev|test|generation-dev|generation-test}" >&2
    exit 2
    ;;
esac

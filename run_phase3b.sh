#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CONFIG="${PHASE3B_CONFIG:-configs/phase3b/protocol.json}"
if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

GEN_MODEL_ARGS=()
if [[ -n "${LLM_MODEL_PATH:-}" ]]; then
  GEN_MODEL_ARGS=(--model "$LLM_MODEL_PATH")
fi

DOWNLOAD_ARGS=()
if [[ "${ALLOW_DOWNLOAD:-0}" == "1" ]]; then
  DOWNLOAD_ARGS=(--allow-download)
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

require_gpu_selection() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set GPU_ID or CUDA_VISIBLE_DEVICES explicitly before GPU work." >&2
    exit 2
  fi
  echo "[gpu] physical CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (PyTorch uses the first visible card as cuda:0)"
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
      --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader
  fi
}

prepare_dev() {
  run_once "freeze Controller and split Dev by question" "results/phase3b/protocol_manifest.json" \
    "$PYTHON_BIN" scripts/prepare_phase3b.py --config "$CONFIG"
  run_once "fit temperature on dev_calibration" "results/phase3b/calibration_manifest.json" \
    "$PYTHON_BIN" scripts/calibrate_phase3b.py --config "$CONFIG"
  run_once "select risk policies on dev_policy" "results/phase3b/policy_selection_manifest.json" \
    "$PYTHON_BIN" scripts/select_phase3b_risk_policy.py --config "$CONFIG"
}

generate_split() {
  local split="$1"
  require_gpu_selection
  run_once "generate frozen stage answers for $split" \
    "results/phase3b/generation/$split/generation_manifest.json" \
    "$PYTHON_BIN" scripts/generate_phase3b_stage_answers.py \
      --config "$CONFIG" --split "$split" "${GEN_MODEL_ARGS[@]}" "${DOWNLOAD_ARGS[@]}"
}

benchmark_split() {
  local split="$1"
  require_gpu_selection
  run_once "benchmark retrieval components for $split" \
    "results/phase3b/latency/${split}_retrieval_manifest.json" \
    "$PYTHON_BIN" scripts/benchmark_phase3b_retrieval.py \
      --config "$CONFIG" --split "$split" "${DOWNLOAD_ARGS[@]}"
  run_once "benchmark Controller for $split" \
    "results/phase3b/latency/${split}_controller_manifest.json" \
    "$PYTHON_BIN" scripts/benchmark_phase3b_controller.py \
      --config "$CONFIG" --split "$split" "${DOWNLOAD_ARGS[@]}"
}

evaluate_split() {
  local split="$1"
  local marker="results/phase3b/${split}_evaluation_manifest.json"
  run_once "evaluate sequential policies on $split" "$marker" \
    "$PYTHON_BIN" scripts/evaluate_phase3b_policies.py --config "$CONFIG" --split "$split"
}

case "${1:-}" in
  prepare-dev)
    prepare_dev
    ;;
  generation-dev)
    generate_split dev_policy
    ;;
  benchmark-dev)
    benchmark_split dev_policy
    ;;
  evaluate-dev)
    evaluate_split dev_policy
    ;;
  dev-all)
    prepare_dev
    generate_split dev_policy
    benchmark_split dev_policy
    evaluate_split dev_policy
    ;;
  freeze-test)
    run_once "freeze all Dev choices before Test" "results/phase3b/test_gate.json" \
      "$PYTHON_BIN" scripts/freeze_phase3b_test_gate.py --config "$CONFIG"
    ;;
  generation-test)
    generate_split test
    ;;
  benchmark-test)
    benchmark_split test
    ;;
  evaluate-test)
    evaluate_split test
    ;;
  bootstrap-test)
    run_once "paired question-level Test bootstrap" "results/phase3b/bootstrap_manifest.json" \
      "$PYTHON_BIN" scripts/bootstrap_phase3b.py --config "$CONFIG"
    ;;
  test-all)
    if [[ ! -f results/phase3b/test_gate.json ]]; then
      echo "Run '$0 freeze-test' and inspect the frozen gate before Test." >&2
      exit 2
    fi
    generate_split test
    benchmark_split test
    evaluate_split test
    run_once "paired question-level Test bootstrap" "results/phase3b/bootstrap_manifest.json" \
      "$PYTHON_BIN" scripts/bootstrap_phase3b.py --config "$CONFIG"
    run_once "write Phase 3B analysis and closing report" "docs/phase3b_closing_report.md" \
      "$PYTHON_BIN" scripts/analyze_phase3b.py --config "$CONFIG"
    ;;
  analyze)
    run_once "write Phase 3B analysis and closing report" "docs/phase3b_closing_report.md" \
      "$PYTHON_BIN" scripts/analyze_phase3b.py --config "$CONFIG"
    ;;
  *)
    echo "Usage: $0 {prepare-dev|generation-dev|benchmark-dev|evaluate-dev|dev-all|freeze-test|generation-test|benchmark-test|evaluate-test|bootstrap-test|test-all|analyze}" >&2
    exit 2
    ;;
esac

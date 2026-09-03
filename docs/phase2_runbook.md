# Phase 2 Server Runbook

Phase 1 artifacts and `docs/experiment_log.md` are frozen at commit `702bfc3`. Phase 2 uses only `configs/phase2`, `data/phase2`, `experiments/phase2`, `results/phase2`, and the Phase 2 audit documents.

## Environment

Run from the repository root on the RTX 4090 server:

```bash
git switch research/sufficiency-phase2
git pull --ff-only
export PYTHON_BIN=python3.12
export GPU_ID=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SOURCE_PARQUET=/path/to/hotpotqa/distractor/validation/0000.parquet
```

The required packages are PyTorch, Transformers, sentence-transformers, FAISS, rank-bm25, NumPy, scikit-learn, pyarrow or datasets, and matplotlib. The model cache must contain `BAAI/bge-large-en-v1.5`, `BAAI/bge-reranker-v2-m3`, and `BAAI/bge-reranker-base` when offline mode is enabled.

The Controller loads `BAAI/bge-reranker-base` with its native scalar reranking head and then explicitly reinitializes only `classifier.out_proj` for two-class Continue/Stop prediction. The resolved config records this strategy. This avoids version-dependent `ignore_mismatched_sizes` failures for the 1-to-2 bias shape while preserving the pretrained encoder and classification-head dense layer.

Controller training and evaluation require `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. The Python entry points enforce the configured value and record it in each training manifest; this is required for the deterministic CUDA matrix multiplications requested by the experiment protocol.

Dense, BM25, and reranker all consume the same `[TITLE] title [TEXT] chunk_text` representation. Scores and per-query timings are preserved in the ignored ranking JSONL and summarized in tracked manifests/results.

Corpus membership is intentionally held fixed to Phase 1 to isolate granularity. That frozen KB is a 14,810-instance shared gold-context pool, not all 73,700 distractor contexts from the official split. `docs/kb_dedup_audit.md` records this limitation; Phase 2 results must not be described as full-corpus HotpotQA distractor retrieval.

## Gate 1: Dev-Only Chunk Selection

```bash
tmux new -s phase2-retrieval
bash run_phase2_dev_retrieval.sh 2>&1 | tee results/phase2/dev_retrieval.log
```

Inspect `results/phase2/retrieval_eval/dev/retrieval_summary.md`, then apply the pre-registered rule. The selector reads only dev rows and records the full decision:

```bash
python3.12 scripts/select_phase2_chunk_variant.py
git diff -- configs/phase2/chunk_retrieval.json
cat results/phase2/chunk_selection.json
```

Do not run test retrieval before `selected_variant_after_dev` is populated. The scripts enforce this gate.

## Gate 2: Selected Variant and Controllers

The full run trains the five fair baselines, three packing representations, and three frozen-encoder hierarchical aggregations. It refuses to overwrite existing runs.

```bash
tmux new -s phase2-controller
bash run_phase2_selected.sh 2>&1 | tee results/phase2/controller.log
```

The hierarchical models are feasibility diagnostics: the shared backbone is frozen and only the mean/max aggregation classifier is trained. Their effective training batch is 16 through gradient accumulation, but they are not presented as end-to-end architecture-matched replacements.

## Raw Versus Cumulative Baseline

The full script trains cumulative evidence as the main policy and also includes a Query+Evidence Uniform-Packing raw-stage baseline. The standalone equivalent is:

```bash
python3.12 scripts/train_phase2_controller.py \
  --variant sentence_256 \
  --baseline query_evidence \
  --representation uniform_packing \
  --evidence-mode raw \
  --run-dir experiments/phase2/controller/sentence_256/query_evidence_uniform_raw
```

Replace `sentence_256` with the value recorded in `results/phase2/chunk_selection.json`. Raw and cumulative label distributions and non-monotonic ratios are generated together by `build_phase2_controller_data.py`.

## Output Discipline

Large chunk, ranking, Controller JSONL, index, and checkpoint files are ignored by Git and rebuilt on the server. Commit the configs, scripts, compact CSV/JSON/Markdown summaries, plots, audit CSV, and logs needed for traceability. Before committing:

```bash
git status --short
python3.12 -m unittest discover -s tests -v
```

No calibration, conformal prediction, risk loss, or test-time threshold fitting is part of Phase 2.

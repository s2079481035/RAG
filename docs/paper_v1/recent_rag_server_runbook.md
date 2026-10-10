# Recent RAG supplement: server runbook

All edits originate locally and reach the server through Git. Never edit the server checkout directly.

```bash
cd /home/sunjb/RAG/RAG_paper
git fetch origin exp/external-related-baselines
git switch exp/external-related-baselines
git pull --ff-only origin exp/external-related-baselines
git rev-parse HEAD
```

The printed commit must equal the locally recorded experiment commit.

## Environment and safety

```bash
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/sunjb/.cache/huggingface
export LLM_MODEL_PATH=/home/sunjb/RAG/ekrag/ekrag/models/Qwen2.5-7B-Instruct
nvidia-smi
```

Do not stop, renice, or modify other users' processes. The runner refuses inference unless the visible GPU has at least the configured cap plus 2 GiB free. Its default model allocation cap is 12 GiB.

## IRCoT-inspired Dev smoke

Preflight makes the immutable manifest and performs no model inference:

```bash
python3.12 scripts/run_recent_rag_dev.py \
  --model "$LLM_MODEL_PATH" \
  --n 32 \
  --output results/paper_v1/ircot_inspired_dev32_preflight \
  --preflight
```

After checking the manifest and GPU headroom, use a different output directory for inference:

```bash
python3.12 scripts/run_recent_rag_dev.py \
  --model "$LLM_MODEL_PATH" \
  --n 32 \
  --output results/paper_v1/ircot_inspired_dev32 \
  --gpu-memory-gib 12
```

Inspect every `trace`: after the first round, `query` must equal the previous generated reasoning sentence and must not equal the original question by construction. An empty/repeated query marks that question invalid and stops its retrieval loop instead of silently becoming repeated fixed-query retrieval. The batch continues so the smoke test can report the valid-trajectory rate.

Only after the 32-row output is complete and valid:

```bash
python3.12 scripts/run_recent_rag_dev.py \
  --model "$LLM_MODEL_PATH" \
  --n 100 \
  --output results/paper_v1/ircot_inspired_dev100 \
  --gpu-memory-gib 12
```

## Result return

Commit only manifests, summaries, and reasonably sized predictions. Never commit model weights, caches, or frozen source data. Before committing, rerun:

```bash
python3.12 -m unittest discover -s tests -p 'test_phase4*.py' -v
git diff --check
```

Latency collected while the GPU is shared or while CPU offload occurs is diagnostic only and must not be placed beside frozen dedicated-run latency as a comparable number.

## SIM-RAG adapted Dev smoke

Use the author-linked general-purpose checkpoint. The script preserves its answer/rationale-conditioned binary Critic and counts reasoner, query-generator, and Critic calls/tokens. The Critic runs on CPU to preserve GPU headroom for other users, so latency is diagnostic.

```bash
python3.12 scripts/run_sim_rag_dev.py \
  --reasoner "$LLM_MODEL_PATH" \
  --critic /home/sunjb/RAG/models/SIM-RAG-Llama3-2B \
  --n 32 \
  --output results/paper_v1/sim_rag_dev32_preflight \
  --preflight
```

Then run the 32-question smoke in a new directory:

```bash
python3.12 scripts/run_sim_rag_dev.py \
  --reasoner "$LLM_MODEL_PATH" \
  --critic /home/sunjb/RAG/models/SIM-RAG-Llama3-2B \
  --n 32 \
  --output results/paper_v1/sim_rag_dev32 \
  --reasoner-gpu-memory-gib 10
```

The HF model name contains `Llama3-2B`, but its published config identifies a 24+24-layer T5 with d_model 2048. Do not describe the Critic architecture from the repository name. This is an adapted reproduction because the reasoner, query prompt, chunk granularity, and corpus differ from the paper setting.

Completed run: commit `33b82bf57e3fe72180d5d0e7431f92c3df38e8cb`, 32/32 rows, output `results/paper_v1/sim_rag_dev32/`. The official snapshot was pinned to `0e1cc45ab595543557837b237020ebcd16aeb357`; local and server SHA-256 values matched for all three weight shards and the index. Do not run 100: the Dev32 smoke already found 9 invalid-query stops and is sufficient for the adapted-runtime conclusion.

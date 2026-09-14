# Phase 4 Heldout Execution Protocol

This protocol implements the researcher's one-time authorization for the official labeled 2WikiMultiHopQA Dev split, used as heldout.

## Frozen Authorization

- `RUN_HELDOUT` must equal `YES`.
- Researcher must be `sunjb`.
- Frozen Dev-choice commit must be `48791e7` and remain an ancestor of the execution commit.
- Every file and Adaptive Router artifact protected by `results/phase4/heldout_unlock_manifest.json` must retain its frozen SHA-256.
- The authorization manifest must be committed, and the worktree must be clean, before the first heldout read.

## One-Time State

`results/phase4/heldout_run_manifest.json` is written immediately before heldout access. An interrupted run may resume only with the same authorization and execution commit. Once `results/phase4/heldout_completion_manifest.json` reports `status=complete`, all heldout entry points reject another formal run.

The pipeline never passes an overwrite flag. Existing large retrieval, prediction, Judge, and generation artifacts are resumed as immutable stage outputs.

## Evaluation Boundary

- The ladder is `Dense@5 -> Hybrid@10 -> Rerank@20` with cumulative evidence.
- Controller False Stop Rate includes reached Dense and Hybrid decisions only.
- Rerank is terminal; incomplete terminal evidence is counted as Terminal Retrieval Failure.
- Temperature, classification threshold, Risk20/10/5 thresholds, score baseline, prompts, models, Router seeds, and generation settings come only from frozen Dev manifests.
- Bootstrap uses 5,000 paired question-level replicates.
- Large JSONL caches, model checkpoints, and logs remain uncommitted. Final CSVs, manifests, reports, figures, and this protocol are committed.

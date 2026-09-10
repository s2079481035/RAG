# Phase 4 Retrieval Execution Audit

This gate uses a deterministic 100-question subset for engineering parity and throughput only. No effect metric is consulted.

- Selected BM25 workers: 8
- Selected Dense search batch size: 1
- Parallel BM25 parity: passed
- Batched FAISS parity: failed; rejected from formal retrieval
- Timed pipeline speedup: 4.02x
- First-stage speedup: 4.73x

Estimated hours exclude one-time model/index loading:

- `dev_calibration`: 0.19 h
- `dev_policy`: 0.19 h
- `heldout`: 1.19 h
- `train_core`: 15.50 h

The rejected batched run is retained as negative engineering evidence and must not be used for formal results.

# Adaptive-RAG Reproduction Audit

## Status: PENDING

Official repository: <https://github.com/starsuzi/Adaptive-RAG>

The official NAACL 2024 implementation routes questions by predicted query complexity among no-retrieval, single-step retrieval, and multi-step retrieval systems. Its documented environment uses Python 3.8, PyTorch 1.13.1 with CUDA 11.7, Elasticsearch 7.10.2, IRCoT-derived retrieval code, FLAN-T5/GPT answer models, and a T5-large complexity classifier.

These components are not identical to this study's shared benchmark-context corpus, BGE/BM25/RRF/reranker ladder, Qwen generator, or cost accounting. Therefore:

1. An official-environment smoke reproduction will be recorded with repository commit, dependencies, model, dataset, and config.
2. Any fair same-pipeline comparison will be named **Adaptive-RAG-style Query Complexity Router**, not Official Adaptive-RAG.
3. Official results and adapted results will not appear as interchangeable rows.

No reproduction result has been run yet.

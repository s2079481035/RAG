# 检索阶段延迟（hotpotqa, n=200, GPU1）

| 阶段 | mean(ms) | median(ms) | std(ms) |
|---|---|---|---|
| embed | 22.1 | 20.9 | 3.2 |
| dense20 | 9.4 | 8.7 | 3.2 |
| hybrid20 | 71.4 | 67.2 | 20.3 |
| rerank20 | 127.4 | 124.3 | 26.5 |

> 说明：embed = 单查询 bge-large-en-v1.5 编码；dense20 = FAISS 全库搜索取 top-20；hybrid20 = dense 全库 + BM25 + RRF 融合；rerank20 = v2-m3 对 20 个候选取分。注：hybrid20 含 dense 全库搜索（比 dense20 的 top-20 搜索贵）。

# Recall@K 矩阵 (Retrieval-Budget Pilot)

| 数据集 | 检索器 | Recall@3 | Recall@5 | Recall@10 | Recall@20 |
| --- | --- | --- | --- | --- | --- |
| nq | dense | 0.834 | 0.886 | 0.925 | 0.962 |
| nq | hybrid | 0.851 | 0.905 | 0.938 | 0.970 |
| nq | rerank | 0.772 | 0.843 | 0.909 | 0.970 |
| hotpotqa | dense | 0.833 | 0.881 | 0.912 | 0.934 |
| hotpotqa | hybrid | 0.846 | 0.893 | 0.929 | 0.956 |
| hotpotqa | rerank | 0.872 | 0.919 | 0.943 | 0.956 |

Hit@K (HotpotQA 主口径):
- HotpotQA dense Hit@3 = 0.981
- HotpotQA dense Hit@5 = 0.986
- HotpotQA dense Hit@10 = 0.992
- HotpotQA dense Hit@20 = 0.995

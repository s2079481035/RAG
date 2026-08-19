# Learned Retrieval Sufficiency Critic 评估
> 数据: HotpotQA test 1000 queries × 3 stages（test 未参与训练/选择）

## D. Critic 离线指标（test, threshold=0.5）
| 阶段 | N | suff率 | P | R | F1 | Acc |
|---|---|---|---|---|---|---|
| dense@3 | 1000 | 0.684 | 0.896 | 0.879 | 0.887 | 0.847 |
| hybrid@5 | 1000 | 0.792 | 0.927 | 0.919 | 0.923 | 0.879 |
| rerank@20 | 1000 | 0.914 | 0.968 | 0.949 | 0.958 | 0.924 |

## E. End-to-End：Fixed vs Oracle ES-3 vs Learned Critic（test n=1000）
| 系统 | FinalRecall | FullCov | Avg文档成本 | Avg延迟(ms) | FalseEarlyStop | 阶段停止比(d/h/r) |
|---|---|---|---|---|---|---|
| Fixed rerank@20 | 0.956 | 0.914 | 20.00 | 220.9 | 0.000 | 0/0/1000 |
| Oracle ES-3 | 0.959 | 0.919 | 6.18 | 72.8 | 0.000 | 684/146/170 |
| Learned Critic | 0.915 | 0.834 | 6.46 | 82.9 | 0.096 | 671/142/187 |

> Critic 单次推理: 4.7 ms/query（含 tokenize）。延迟含 critic 调用次数。

## F. 与 Oracle 的差距
- Recall 差距: oracle 0.959 vs critic 0.915 = **+0.043**
- FullCov 差距: oracle 0.919 vs critic 0.834 = **+0.085**
- 成本差距: oracle 6.18 vs critic 6.46 docs
- False Early Stop: oracle 0 按构造; critic 96 个查询误停（9.6%）
- 误停造成的 Recall 损失上限: 这些查询在误停点平均覆盖率 = 0.495

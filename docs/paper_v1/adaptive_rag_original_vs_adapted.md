# Original Adaptive-RAG 与冻结 adapted baseline

**Adaptive-RAG-style Query Router is an adapted baseline, not a faithful reproduction of original Adaptive-RAG.**

原论文：https://aclanthology.org/2024.naacl-long.389/；代码：https://github.com/starsuzi/Adaptive-RAG。

原方法以 query 为 T5-Large classifier 输入，根据模型预测结果产生的 silver labels 与数据集复杂性偏置训练，在检索前选择 No Retrieval / Single-step Retrieval / Iterative Retrieval。与读取当前累计证据的逐阶段决策不同。

## Table A — LITERATURE-REPORTED, NON-COMPARABLE PROTOCOLS

| Dataset | Generator | Retriever | N | EM (%) | F1 (%) | Retrieval-generation steps | Relative time |
|---|---|---|---:|---:|---:|---:|---:|
| HotpotQA | FLAN-T5-XL | BM25 | 500 | 42.00 | 53.82 | 3.55 | 5.99 |
| 2WikiMultiHopQA | FLAN-T5-XL | BM25 | 500 | 40.60 | 49.75 | 2.63 | 4.68 |

来源：原论文 Table 2，PDF 第 7 页 / proceedings p.7042；相对时间以单步方法为 1，不是秒。`source_type=original_paper`。不能据此与 Ours 计算提升百分比。

## Table B — 统一冻结 2Wiki heldout

| System | N | Answer F1 | Complete coverage | SF Recall | Avg chunks | Reranker calls | Total latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adapted Router（三 seed mean） | 12576 | 0.19576064593012324 | 0.3675254452926209 | 0.6900895886344359 | 15.487171331636981 | 0.6858301526717557 | 1240.07964224177 |
| Ours（冻结 Risk10） | 12576 | 0.20256212239279647 | 0.35520038167938933 | 见 CSV | 13.268686386768447 | 0.5010337150127226 | 1124.3924006067339 |

来源：`results/phase4/final/table_external_baselines.csv` 与 `table_end_to_end.csv`。完整数字由 `scripts/build_recent_rag_comparison.py` 无训练地汇总到 `results/paper_v1/unified_baseline_comparison.csv`，原始来源 SHA-256 在 `existing_result_provenance.json`。

Dev 原文件 `results/phase4/2wiki/adaptive_rag_dev_summary.csv` 的三个 seeds 42/123/2026 与 mean/std 全部保留；heldout 三 seeds 与 mean/std 同样保留。没有重新选 seed。Router 选择 Dense@5 / Hybrid@10 / Rerank@20，而非原方法的检索策略空间。

Dev 和 heldout 的三个 seeds 均不预测 Medium（0.0）；heldout route accuracy mean 为 0.8822890161153519，route Macro F1 mean 为 0.5935941014885876，recoverable under-retrieval mean 为 0.07615430741555067。完整 route distribution、coverage、recall、成本及已有 latency 都保留；源文件未记录的 Dev latency 不估填。三 seed std 不是置信区间，亦不用于重新选择 seed。

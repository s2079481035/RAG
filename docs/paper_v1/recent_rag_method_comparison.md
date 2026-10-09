# 六类 Adaptive / Iterative RAG 定量与方法对位

审计：2026-10-09。该文件是补充实验工作记录，不表示新增 GPU 实验已完成。冻结主实验不变。

## Table 1 — LITERATURE-REPORTED, NON-COMPARABLE PROTOCOLS

| Method | Year | Dataset | Task | Retriever | Generator | EM % | Answer F1 % | Other metric | Source |
|---|---:|---|---|---|---|---:|---:|---|---|
| Adaptive-RAG | 2024 | HotpotQA | QA | BM25 | FLAN-T5-XL | 42.00 | 53.82 | 3.55 retrieval-generation steps | Table 2 p7 |
| Adaptive-RAG | 2024 | 2Wiki | QA | BM25 | FLAN-T5-XL | 40.60 | 49.75 | 2.63 steps | Table 2 p7 |
| IRCoT | 2023 | HotpotQA | QA | BM25 | code-davinci-002 | 49.3 | 60.7 | NR | Table 3 p14 |
| IRCoT | 2023 | 2Wiki | QA | BM25 | code-davinci-002 | 57.7 | 68.0 | NR | Table 3 p14 |
| SIM-RAG full | 2025 | HotpotQA | QA | BM25 | Llama3-8B | 32.7 | 43.3 | NR | Table 1 p7 |
| SIM-RAG full | 2025 | 2Wiki | QA | BM25 | Llama3-8B | 34.1 | 40.2 | NR | Table 1 p7 |
| S2G-RAG | 2026 | HotpotQA | QA | BM25 | Llama3-8B | 43.3 | 56.5 | max 4 turns, top6 | Table 1 p6 |
| S2G-RAG | 2026 | 2Wiki | QA | BM25 | Llama3-8B | 41.7 | 48.6 | max 4 turns, top6 | Table 1 p6 |
| Stop-RAG authors pipeline | 2025 | HotpotQA | QA | Contriever + BGE | Llama3.1-8B | 52.4 | 66.1 | NR | Table 1 p7 |
| Stop-RAG authors pipeline | 2025 | 2Wiki | QA | Contriever + BGE | Llama3.1-8B | 68.2 | 75.7 | 5.1 rounds; document recall 82.6% | Tables 1–2 p7–8 |
| SURE-RAG calibrated | 2026 | HotpotQA-RAG v3 | Verification | constructed evidence | candidate supplied | NR | NR | Macro F1 0.9075; Risk@30 0.1670 | Tables III/V p6 |

完整可追溯记录见 `results/paper_v1/literature_metrics.csv`：19 行，包含附加 generator / retriever 设置、split、语料、样本数、模型、训练信息、风险定义、页码与原始 URL。全部为 `original_paper`；未冒充本地复现。NR 表示该提取记录无可用报告值，不将代码默认值冒充论文运行参数。尚需逐项补全未核实的语料规模、原文成本表与训练细节；这些字段不能据此断言论文从未报告。

论文 PDF 原始来源：

- https://aclanthology.org/2024.naacl-long.389.pdf
- https://aclanthology.org/2023.acl-long.557.pdf
- https://arxiv.org/pdf/2505.02811v2
- https://aclanthology.org/2026.acl-long.1185.pdf
- https://arxiv.org/pdf/2510.14337
- https://arxiv.org/pdf/2605.03534

## Table 2 — 冻结统一环境

机器可读完整表：`results/paper_v1/unified_baseline_comparison.csv`。2Wiki heldout 12576 questions，包括 FixedHeavy、ScoreThresholdRisk10、Judge-512、Judge-Full、AdaptiveRouter 三 seeds/mean/std、冻结 Ours Risk10。HotpotQA Test 单独成组。Dev Router 单独成组，不能当作 heldout 数字。

| 2Wiki heldout system | Answer F1 | Complete coverage | Avg chunks | Total latency ms |
|---|---:|---:|---:|---:|
| Fixed Heavy | 0.19215672075640125 | 0.39782124681933845 | 20.293256997455472 | 1676.3052861272201 |
| Score Threshold Risk10 | 0.19215672075640125 | 0.39782124681933845 | 20.293256997455472 | 1676.3052861272201 |
| LLM Judge-512 | 0.19794994770897936 | 0.3928912213740458 | 17.685114503816795 | 1788.9903085523201 |
| LLM Judge-Full | 0.19844733253365426 | 0.3934478371501272 | 18.041825699745548 | 1890.598273004002 |
| Adaptive-RAG-style mean | 0.19576064593012324 | 0.3675254452926209 | 15.487171331636981 | 1240.07964224177 |
| Ours Risk10 | 0.20256212239279647 | 0.35520038167938933 | 13.268686386768447 | 1124.3924006067339 |

Answer EM、SF Recall、tokens、calls 与风险见 CSV。Frozen latency 是原有组件测量的组合，不能当作新增共享 GPU 环境 wall-clock。Judge total tokens 加上单独记录的 Judge 输入/输出；Router 的 T5 classifier 不是生成式 LLM 调用，其 latency 已在原表中计入。std 行不将边际标准差简单相加。

当前只有既有统一环境方法可横向比较。六篇原论文均不得与 Ours 直接计算提升百分比。SURE verification F1 与 Answer F1 分列；其 risk 不等于 FSR。SIM-RAG 与 IRCoT 新实验暂未加入正式结果表。

## 仍需完成的工作

1. SIM-RAG checkpoint 下载与 32→100 Dev smoke；审计训练集重叠和完整调用成本。
2. IRCoT-inspired 服务器运行、逐轮 query 轨迹核验及 32→100 Dev 扩展。
3. 新增方法若进入 Table 2，需在同一冻结问题集合上取得全部 comparator 结果；不能把 Dev smoke 接到 12576-question heldout 表。
4. 完整的六篇成本/训练字段逐项审计，以及对应原表视觉复核尚未全部完成。

QUANTITATIVE_COMPARISON_READY = NO

NO_MORE_EXPERIMENTS_RECOMMENDED = NO

# SURE-RAG：任务与发布边界

2026-10-09 核查官方仓库 https://github.com/mouwumou/SURE-RAG ，commit `fa9ec380d32eb9092512a8d00f906828b036b0fb`。这更新了旧审计中“未找到官方仓库”的结论。

**NOT REPRODUCIBLE AS TRAINED MODEL FROM CURRENT RELEASE**

README 明确未提供 trained pair scorer、trained aggregation backend、benchmark data / paper output files 或完整训练 pipeline。现有 toy backend 和 precomputed-score demo 不能作为训练模型的 benchmark predictions。不启动训练。

输入为 question、candidate answer（可为草稿/中间 claims）、evidence。DeBERTa-v3-base 对 claim-passage 关系评分，set-level aggregation 综合 coverage、relation strength、uncertainty/disagreement、conflict 与可选 retrieval features。论文算法输出 supported/refuted/insufficient 和 answer/abstain；新版协议还允许 retrieve_more、regenerate、human_review。因此不能简单写成“只能用于最终答案之后”。

论文 https://arxiv.org/abs/2605.03534 的 HotpotQA-RAG v3 是构造的验证 benchmark：4026 examples、900 question groups、20130 claim-evidence pairs；不是我们的共享语料检索轨迹，也不是相同 split。三 seed 为 13/21/42。

| 原论文指标 | Raw SURE | Calibrated SURE | 对照 | 定位 |
|---|---:|---:|---|---|
| Verification Macro F1 | 0.8951 ± 0.0069 | 0.9075 ± 0.0060 | Concat cross-encoder 0.8888 ± 0.0109 | Table III |
| Binary safety ECE | 0.0304 | 0.0198 | NR | §VI-E |
| Risk@30 | 0.1642 | 0.1670 | Mean-pool 0.2588 | Table V |
| Risk@50 | 0.3600 | 0.3510 | Mean-pool 0.3783 | Table V |
| Risk@70 | 0.5457 | 0.5493 | Mean-pool 0.5350 | Table V |

上述均为 `source_type=original_paper`，不是本地复现。LLM Judge 的 matched sampled ID 比较需要单独记录其样本范围，不能直接把全量 SURE 结果与 sampled GPT-4o 数字相减。Verification Macro F1 不进入 Answer F1 列。

Selective risk 分母是被选择输出的答案数，分子是其中 unsafe 的答案。Risk@30 表示选择覆盖率 30%，不是证据覆盖率 30%。我们的 sequential FSR 分母是到达的、gold 应继续的可行动决策数，分子是其中错误 Stop，强制终止阶段不计入。输入、标签、单位、分母、动作成本均不同，禁止两者直接排名。

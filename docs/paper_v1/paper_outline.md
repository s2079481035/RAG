# 《电子与信息学报》中文论文 V1 结构建议

## 总体状态

`READY_FOR_WRITING = YES`

理由：Phase 1–4、HotpotQA、2Wiki frozen heldout、question-level bootstrap、内部 baselines 与外部方法审计均已收口；主要支持结论与反例都可追溯。当前缺口是作者的论证取舍、中文表达和基于现有数据的图表整理，不是实验缺口。

## 1 引言

**建议内容**

1. 多阶段 RAG 的固定深度存在质量—成本矛盾。
2. 停止决策的关键不是只判断问题难度，而是判断“当前证据是否充分”。
3. Partial evidence 尤其容易造成 false stop；但 evidence insufficiency 又不等价于 future retrieval utility。
4. 本文在冻结三阶段 ladder 中研究 evidence-aware critic、Hard Partial、保守风险阈值及 retrieval-limited boundary。
5. 贡献表述保持为“系统研究与实证框架”，不要写成普适风险保证或检索问题已解决。

**状态**：已有材料=是（research story、Phase 3A/3B/4 closing）；缺少理解=是（需作者确定核心贡献排序）；缺少文字=是（需正式中文引言）；缺少图表=可选（建议一张系统动机图）；不需要新实验=是。

## 2 相关工作

### 2.1 Pre-retrieval routing

围绕 Adaptive-RAG 与 query-complexity routing，强调其在 evidence 产生前做策略选择。

**状态**：已有材料=是；缺少理解=否；缺少文字=是；缺少图表=否；不需要新实验=是。

### 2.2 Evidence-aware iterative retrieval and stopping

对位 S2G-RAG 的 sufficiency+gap+query regeneration，以及 Stop-RAG 的 future-utility value learning。

**状态**：已有材料=是（external audits）；缺少理解=是（需作者决定文献叙述比重）；缺少文字=是；缺少图表=可选（decision-timing matrix）；不需要新实验=是。

### 2.3 Answer verification and LLM Judge

区分 SURE-RAG 的 answer/abstain 与本文的 stop/continue retrieval；说明 LLM Judge 是同协议内部 baseline。

**状态**：已有材料=是；缺少理解=否；缺少文字=是；缺少图表=否；不需要新实验=是。

## 3 问题定义

建议定义：

- Question `q`，累计 model-visible evidence `E_t`，固定 stages `t∈{Dense@5, Hybrid@10, Rerank@20}`。
- Actionable stages 只有前两级；最终 Rerank@20 是 forced terminal。
- Gold sufficiency：累计 evidence 覆盖全部 unique gold supporting facts。
- Stop/Continue、False Stop Rate、Unnecessary Escalation Rate、Hard Partial FSR、RFSR、TRFR。
- 明确 FSR denominator 是 reached actionable Continue states，terminal incomplete 不计入 Controller false stop。
- 区分 evidence sufficiency 与 future retrieval utility。

**状态**：已有材料=是（protocol、metric implementation audits）；缺少理解=是（需作者统一中文符号与定义顺序）；缺少文字=是；缺少图表=是（建议三阶段状态/动作示意）；不需要新实验=是。

## 4 方法

### 4.1 Multi-stage retrieval

描述 Dense@5 → Hybrid@10 → Rerank@20、cumulative evidence、sentence_256、最终强制阶段及成本统计。

**状态**：已有材料=是；缺少理解=否；缺少文字=是；缺少图表=是（pipeline 图）；不需要新实验=是。

### 4.2 Evidence-aware sufficiency critic

描述 question+evidence 输入、score-aware packing、cross-encoder、binary Stop/Continue head。先写最终方法，再把 query-only/query-stage 作为消融。

**状态**：已有材料=是；缺少理解=是（需作者决定 Phase 1 preliminary 是否进主文）；缺少文字=是；缺少图表=可并入 pipeline；不需要新实验=是。

### 4.3 Hard-partial-aware training

建议标题谨慎改为“Hard Partial 建模与 Coverage Auxiliary”，避免暗示 hard-partial sampling 成功。写明：

- coverage auxiliary head 是最终建模组成；
- coverage regression signal 可学习；
- 多 seed 不支持其稳定改善分类；
- hard-partial-aware sampling 是负结果。

**状态**：已有材料=是；缺少理解=是（作者必须决定如何解释“最终使用但消融无稳定收益”）；缺少文字=是；缺少图表=是（bucket 分布/FSR 图）；不需要新实验=是。

### 4.4 Risk-constrained stopping

描述 calibration/policy split、Temperature Scaling、候选阈值、point 与 conservative selection、risk targets、一次性 frozen evaluation。明确经验风险控制而非形式保证。

**状态**：已有材料=是；缺少理解=是（需作者确定“risk-constrained”中文术语与保证边界）；缺少文字=是；缺少图表=是（risk-cost curve/reliability diagram 已有 Phase 3B 图）；不需要新实验=是。

## 5 实验设置

### 5.1 Datasets and splits

HotpotQA 的 Train/Dev/Test 与 2Wiki 的 train_core/dev_calibration/dev_policy/official Dev heldout；说明 2Wiki 官方 Test 无标签未用。

### 5.2 Retrieval, models, and generation

统一列出 chunk、retriever、reranker、critic backbone、Qwen2.5-7B-Instruct、greedy decoding 与 answer normalization。

### 5.3 Baselines

FixedDense/Hybrid/Heavy、Query-only、Query+Stage、Score Threshold、LLM Judge、Adaptive-RAG-style Router；S2G/Stop/SURE 只进外部方法审计，不进统一数值主表。

### 5.4 Metrics and statistics

Macro F1、AUROC、FSR、UER、Hard Partial FSR、coverage、Answer EM/F1、chunks、reranker calls、latency；question-level bootstrap 保持同 question 的全部 stages。

**状态**：已有材料=是；缺少理解=是（需作者决定主文保留哪些实现细节）；缺少文字=是；缺少图表=否（主要是设置表）；不需要新实验=是。

## 6 实验结果

### 6.1 Evidence state 是否必要

主证据：Hotpot 三 seed score-aware vs Query+Stage；paired bootstrap；2Wiki Dev 三 seed与heldout classification。Phase 1 结果作为动机或附录。

**状态**：已有材料=是；缺少理解=否；缺少文字=是；缺少图表=是（主结果表）；不需要新实验=是。

### 6.2 Hard Partial 分析

先报告 Hard Partial 占 Continue 的 85.4%，再报告 bucket FSR；明确“数量集中”不等于“条件错误率最高”。Coverage Auxiliary 和 sampling 作为正/负消融一起报告。

**状态**：已有材料=是；缺少理解=是（需决定负结果篇幅）；缺少文字=是；缺少图表=是（bucket bar/box plot，仅用已存结果）；不需要新实验=是。

### 6.3 Risk-efficiency trade-off

HotpotQA：Risk10 vs FixedHeavy 的 F1 非劣、chunks/latency 降低、FSR CI 未完全低于目标。2Wiki：F1 与成本结果有利，但 heldout FSR CI 完全高于 10%。

**状态**：已有材料=是；缺少理解=是（需作者选择以 trade-off 还是 risk generalization gap 为主标题）；缺少文字=是；缺少图表=是（两数据集 risk-cost 图）；不需要新实验=是。

### 6.4 HotpotQA vs 2Wiki

使用 frozen cross-dataset table：Hotpot final coverage `0.922`，2Wiki `0.39782124681933845`；对比 Risk10 的 FSR、chunks、reranker 与 latency。避免直接比较绝对 QA 难度或硬件依赖延迟。

**状态**：已有材料=是；缺少理解=是（需作者定义 retrieval-sufficient/limited 的措辞，避免二分法过强）；缺少文字=是；缺少图表=是（cross-dataset table/diagram）；不需要新实验=是。

### 6.5 External baselines

主表只放同 frozen ladder 的 Score Threshold、LLM Judge、Adaptive-RAG-style Router。S2G-RAG、Stop-RAG、SURE-RAG 放方法/复现状态表，不填性能数值。

**状态**：已有材料=是；缺少理解=否；缺少文字=是；缺少图表=否（表格足够）；不需要新实验=是。

## 7 讨论

### 7.1 Retrieval-sufficient vs retrieval-limited

用 Hotpot `0.922` final coverage 与 2Wiki `0.39782124681933845` final coverage 展示同一 policy 在不同 ceiling 下的行为；说明这是观测到的 regime，不是数据集本体标签。

**状态**：已有材料=是；缺少理解=是（需作者给出严格但不过度的 regime 定义）；缺少文字=是；缺少图表=可选；不需要新实验=是。

### 7.2 Evidence sufficiency vs future retrieval utility

结合 Stop-RAG 理论对位和 2Wiki rescue rates，说明 current insufficiency 与 continuation value 可分离；Ours 估计前者，不宣称解决后者。

**状态**：已有材料=是；缺少理解=是（这是最需要作者亲自打磨的理论段）；缺少文字=是；缺少图表=可选（二维概念图）；不需要新实验=是。

### 7.3 Risk generalization gap

Hotpot observed FSR `0.08445945945945946` 与 2Wiki `0.12310521976279572`；强调 2Wiki CI 全高于 10%，表明 Dev 风险目标未跨分布泛化。Temperature Scaling 不应被写成充分保证。

**状态**：已有材料=是；缺少理解=是（需确定该负结果在摘要/结论中的显著程度）；缺少文字=是；缺少图表=是（risk target vs observed CI）；不需要新实验=是。

## 8 局限性

至少包括：

1. Sufficiency proxy 基于 gold supporting facts；人工审计规模 180、单 reviewer、分层而非总体随机抽样。
2. Model-visible packing 可能隐藏 full retrieved evidence 中的 gold facts。
3. Coverage Auxiliary 无稳定三 seed分类收益；hard-partial sampling 失败。
4. Hotpot FSR CI 未完全低于 10%，2Wiki heldout 明确未达 10%。
5. 2Wiki shared benchmark-context corpus 与 frozen ladder 有约 60.22% terminal failure，结论不外推至更强 retriever/query reformulation。
6. LLM Judge 与 QA generator 非独立，latency 依赖硬件和语料规模。
7. Adaptive Router 是 adapted baseline；S2G/Stop/SURE 没有同协议数值比较。
8. Hotpot Test 在 Phase 3A 已被检查，Phase 3B 不是全新独立 confirmatory test。

**状态**：已有材料=是；缺少理解=否；缺少文字=是；缺少图表=否；不需要新实验=是。

## 9 结论

建议只收束到三点：

1. 当前 evidence state 对停止判别比 query/stage shortcut 更有信息。
2. 保守阈值能够暴露并管理风险—效率 trade-off，但不构成跨分布保证。
3. Retrieval-limited regime 使 sufficiency 与 future retrieval utility 分离，系统必须同时报告 false stop 与 terminal retrieval failure。

**状态**：已有材料=是；缺少理解=是（需作者决定最终一句强调“方法”还是“边界发现”）；缺少文字=是；缺少图表=否；不需要新实验=是。

## 建议主图主表

1. **图1 方法总览**：固定三阶段 ladder、packed evidence、critic 与强制终止。
2. **表1 Evidence-state ablation**：Query+Stage、concat、score-aware，多 seed + bootstrap。
3. **图2 Hard Partial 分析**：bucket 样本数与 FSR，避免只画一个百分比。
4. **表2 Hotpot/2Wiki frozen quality-cost**：FixedHeavy、Risk10、Router/Judge（按数据集拆分）。
5. **图3 Risk-efficiency boundary**：风险目标、observed FSR CI、chunks/latency。
6. **图4 Retrieval ceiling/rescue**：Dense→Hybrid→Final coverage 与 rescue rate。
7. **表3 Related-work decision matrix**：Adaptive-RAG、S2G-RAG、Stop-RAG、SURE-RAG、Ours。

所有图表均由已冻结 CSV/JSON 重绘，不新增模型推理、重采样或指标计算。

## 最需要作者本人思考的 10 个问题

1. 论文唯一主问题究竟是“如何更可靠地停止”，还是“如何刻画 sufficiency controller 的风险—效率边界”？两者只能有一个作为标题级主线。
2. Coverage Auxiliary 已进入最终 Controller，但三 seed 未显示稳定分类收益；正文应把它定义为 representation regularizer、辅助可解释任务，还是如实降为非关键组件？
3. “Hard-partial-aware training”这一节名是否会误导读者认为 sampling 成功？是否应改为“Hard Partial 建模与诊断”？
4. 10% 风险目标在 2Wiki heldout 失败，摘要中应以多强的措辞主动披露这一 generalization gap？
5. “Retrieval-sufficient / retrieval-limited”是方便讨论的经验 regime，还是要给出操作性定义？阈值应避免人为设定。
6. 论文贡献中是否把 negative findings（coverage auxiliary/sampling 不稳定、score threshold 退化、Router 不预测 Medium）作为系统研究价值明确列出？
7. 2Wiki Answer F1 很低但 adaptive policy 相对 FixedHeavy 有提升；正文应如何避免把“相对改进”误写成“系统已可用”？
8. S2G-RAG 与 Stop-RAG 是最接近的 novelty 邻居；作者希望把核心边界落在“risk control”还是“fixed-ladder evidence-state modeling”？
9. 中文术语如何统一：False Stop、Unnecessary Escalation、Hard Partial、Terminal Retrieval Failure、future retrieval utility 是否保留英文缩写？
10. 在篇幅限制下，哪些数字必须进主文，哪些放补充材料？建议优先保留跨 seed、paired bootstrap、frozen heldout 与明确负结果。

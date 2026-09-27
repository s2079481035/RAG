# 相关工作定位素材

## 写作总原则

相关工作部分应围绕“决策发生在 RAG 流程的哪个时点、观察什么状态、优化什么目标”组织，而不是按模型大小排列。外部方法没有同协议数值结果时，只做方法级对位，不写“优于”。

## 1. Adaptive-RAG

- **它解决什么问题？**  在执行检索前，根据问题复杂度选择不检索、单步检索或多步检索等策略，避免所有问题使用同一推理预算。
- **它的输入是什么？**  主要是原始 query，以及由 query 可推断的复杂度特征；决策时尚未观察当前检索 evidence。
- **它的决策是什么？**  Pre-retrieval strategy/route selection。
- **为什么这样设计？**  问题复杂度与所需检索深度相关，提前路由可以减少容易问题的无谓成本。
- **它和我们的共同点是什么？**  都试图按样本动态分配检索计算，并比较质量与成本。
- **它和我们的核心区别是什么？**  Adaptive-RAG-style routing 根据 query prior 一次性选路线；Ours 在检索后读取当前 evidence，并在可行动 stage 逐次判断 Stop/Continue。
- **我们绝对不能声称什么？**  不能声称本研究完整复现或击败官方 Adaptive-RAG。本仓库的是“Adaptive-RAG-style Query-Complexity Router”，且 Medium route rate 为 0 是该实现的冻结失败模式。

**建议正文句式**：与检索前 query-complexity routing 不同，本文在每个可行动检索阶段显式观察累计证据，使决策能够响应实际 retrieval outcome，而不仅依赖问题难度先验。

## 2. S2G-RAG

- **它解决什么问题？**  在迭代 RAG 中判断当前证据是否充分；若不充分，定位缺失信息并生成下一轮检索 query。
- **它的输入是什么？**  Question + accumulated selected evidence。
- **它的决策是什么？**  Binary sufficiency，并在 insufficient 时输出 structured gap items；gap 被转成新 query。
- **为什么这样设计？**  单纯 Continue 不能说明缺什么；显式 gap 可改变后续检索方向，提高证据获取能力并控制 context growth。
- **它和我们的共同点是什么？**  都把 question+current evidence 作为 sufficiency state，并在证据足够时停止额外检索。
- **它和我们的核心区别是什么？**  S2G-RAG 重点是 gap generation、query reformulation 与 evidence memory acquisition；Ours 使用固定三阶段 ladder，重点是 partial/sufficient discrimination、conservative threshold、FSR 与 risk-efficiency boundary。
- **我们绝对不能声称什么？**  不能声称 Ours 数值优于 S2G-RAG。官方 trained Judge LoRA 未获得，N=0 表示未执行，不是性能为零；也不能把自行重训或替换 judge 称为官方复现。

**建议正文句式**：S2G-RAG 与本文在 evidence-aware sufficiency state 上最接近，但其贡献侧重利用结构化信息缺口主动生成新检索查询；本文则研究在冻结检索阶梯内如何辨别 Hard Partial 与 Sufficient evidence，并量化保守停止的风险—效率边界。

## 3. Stop-RAG

- **它解决什么问题？**  学习何时停止迭代检索，使停止动作与继续检索的未来回答效用相比较。
- **它的输入是什么？**  概念上为完整 trajectory state；发布的 value model 主要序列化 question 与 accumulated documents。
- **它的决策是什么？**  比较 STOP 与 CONTINUE 的 action value。
- **为什么这样设计？**  当前证据不足并不保证继续检索有益；价值函数可把未来 retrieval utility 与回答收益纳入决策。
- **它和我们的共同点是什么？**  都在检索轨迹中做 Stop/Continue 决策，并关注检索成本。
- **它和我们的核心区别是什么？**  Ours 估计 `P(current evidence sufficient | q,E)`；Stop-RAG 估计继续检索的 expected future utility。Ours 的监督来自 supporting-fact sufficiency/coverage，Stop-RAG 使用完整 trajectory、重复生成和 TD(λ) value target。
- **我们绝对不能声称什么？**  不能声称 Ours 优于 Stop-RAG；不能把 sufficiency label 当作其 value target；不能把三阶段 classifier 改造后仍称 Stop-RAG。其官方训练资源与本研究协议不一致，实验被按规则跳过。

**建议正文句式**：本文的停止依据是“当前证据是否充分”，而 Stop-RAG 的依据是“继续检索是否具有未来效用”；2Wiki 上较高的 terminal retrieval failure 与较低 stage rescue rate 表明，两种目标在 retrieval-limited regime 中可能显著分离。

## 4. SURE-RAG

- **它解决什么问题？**  验证候选答案是否被证据支持，并在不安全时选择 abstain。
- **它的输入是什么？**  Question + candidate answer + evidence set。
- **它的决策是什么？**  Supported / Refuted / Insufficient，随后 answer / abstain。
- **为什么这样设计？**  RAG 可能生成与检索证据不一致或证据不足的答案；claim-level verification 可控制选择性回答风险。
- **它和我们的共同点是什么？**  都关注 evidence reliability、Insufficient 状态与风险敏感决策。
- **它和我们的核心区别是什么？**  SURE-RAG 在候选答案生成后验证回答；Ours 在候选答案生成前判断是否继续 retrieval。二者的输入、动作、risk denominator 与系统位置均不同。
- **我们绝对不能声称什么？**  不能把 answer/abstain risk 与 false-stop risk 直接比较；不能声称 Ours 在 answer verification 上优于 SURE-RAG；不能将其 Insufficient label 直接等同于本文 Continue label。

**建议正文句式**：SURE-RAG 面向生成后的选择性回答安全，而本文面向生成前的检索资源分配；二者共享证据可靠性问题，但控制的是 RAG 流程中的不同决策接口。

## 5. LLM-as-a-Judge

- **它解决什么问题？**  利用通用 LLM 的语义判断能力评估当前证据能否支持回答，减少专门训练数据或任务模型依赖。
- **它的输入是什么？**  本研究 baseline 输入 question 与 model-visible evidence；Judge-512 使用受限 packed context，Judge-Full 使用更长 evidence context。
- **它的决策是什么？**  硬 Stop/Continue 标签；格式非法时安全地 Continue。
- **为什么这样设计？**  大模型可能识别跨句、多跳语义关系，且 prompt 可直接表达 sufficiency 标准。
- **它和我们的共同点是什么？**  都在相同 actionable stages 读取 question+evidence 并判断 sufficiency。
- **它和我们的核心区别是什么？**  Ours 是轻量监督 cross-encoder，输出连续 stop probability 并支持 threshold/risk sweep；Judge 是生成式 hard-label decision，消耗更多 token/latency，不能在没有概率时报告 AUROC。
- **我们绝对不能声称什么？**  不能声称轻量 critic 普遍胜过所有 LLM Judge；不能把本研究 prompt/model 代表整个范式；不能把使用同一 generator 的 Judge 当作独立 human oracle；不能人为构造 Judge AUROC。

**建议正文句式**：冻结结果显示，LLM Judge 可通过保守地预测 Continue 获得较低 FSR，但伴随 73.8%–76.7% 的 UER 与更高 token/latency；因此本文同时报告风险和升级成本，而不只比较错误停止率。

## 建议的相关工作分组

1. **Pre-retrieval routing**：Adaptive-RAG 及 query-complexity routing。
2. **Evidence-aware iterative acquisition**：S2G-RAG。
3. **Value-based stopping**：Stop-RAG。
4. **Answer verification and selective prediction**：SURE-RAG、LLM-based answer verification。
5. **本文定位**：evidence-aware sufficiency discrimination + hard-partial diagnostics + empirical risk-constrained stopping under a frozen multi-stage ladder。

## 外部数值比较边界

- S2G-RAG：`PARTIAL` reproduction；没有性能数字。
- Stop-RAG：protocol audit only；没有性能数字。
- SURE-RAG：method audit only；没有性能数字。
- 只有本仓库的 Score Threshold、LLM Judge 与 Adaptive-RAG-style Router 在相同 2Wiki frozen ladder 下具备直接描述性对照条件。
- 不对不同 retriever、generator、corpus 或 iterative-query protocol 做 paired significance test，也不写“显著优于外部方法”。

# Adaptive / iterative RAG 方法对位

审计日期：2026-10-09。主实验 Phase 1–4 冻结；本阶段只新增外部对照与文献审计。

| Method | Decision Input | Decision Time | Decision Objective | Continue Action | Cost/Risk Model |
|---|---|---|---|---|---|
| Adaptive-RAG | Question | 检索前 | Query complexity routing | No / single / iterative retrieval | QA 指标、检索生成步数、相对时间 |
| IRCoT | Question + retrieved paragraphs + preceding CoT | 每个推理句后 | Interleaved reasoning and retrieval | 最近生成的推理句作为新检索查询 | 轮数与段落预算；非 FSR 控制 |
| SIM-RAG | Question + context + candidate answer + rationale | 每轮候选答案后 | Information sufficiency critic | 生成新查询并检索 | 额外 reasoner / query / critic 调用必须全部计费 |
| S2G-RAG | Question + accumulated sentence evidence | 每轮检索后 | Sufficiency and structured gaps | Gap-driven query reformulation | Judge / extractor / reasoner；无相同 FSR 约束 |
| Stop-RAG | Question + accumulated documents（实现） | 每轮检索后 | Future retrieval utility | 继续原迭代检索系统 | TD(lambda) 双动作价值；验证集选择 margin |
| SURE-RAG | Question + candidate answer/claims + evidence | 候选内容形成后 | Answer-conditioned evidence verification | 论文 answer/abstain；发布协议另含 retrieve/regenerate/review | Unsafe emitted answers / emitted answers |
| Ours | Query + cumulative evidence | 冻结梯度中的可行动阶段 | Evidence-aware stopping with empirical false-stop risk control | Dense@5 → Hybrid@10 → Rerank@20 | Reached actionable decision FSR；Dev 选择、heldout 验证 |

## 对 novelty 的影响

SIM-RAG 与 S2G-RAG 已直接研究 sufficiency-guided stopping；Stop-RAG 已研究学习式 Stop/Continue。不能宣称首次提出 evidence sufficiency、首次训练 stopping controller、或全面优于 iterative RAG。SURE-RAG 也表明证据充分性不是单纯 relevance，但其判断以候选答案为条件。

可保留的贡献是：在固定检索梯度上，以累计证据识别 partial/sufficient 状态；系统诊断 hard partial、输入表示与检索上限；用明确分母的 empirical false-stop risk 与成本共同评估，并公开多 seed、question bootstrap 与冻结 heldout 的失败边界。Coverage auxiliary 不能写成稳定跨 seed 提升。

2Wiki 最终 complete coverage 为 0.39782124681933845（`results/phase4/final/table_retrieval.csv`）。当前不充分不代表下一阶段可补全，因此 current sufficiency 不等于 future retrieval utility。Risk10 heldout FSR 为 0.12310521976279572，不能声称保证风险低于 10%；HotpotQA 的区间上界同样超过 10%。

## 实施约束

本地编辑、验证、Git commit/push；服务器 fetch/pull 相同提交后执行。新增结果单独存放，记录 commit、配置、ID 与语料 SHA-256。先 Dev 32，再 100；更大子集提前冻结。新增 heldout 比较必须标记 post-hoc supplementary comparison。不得改 Ours、backbone、loss、calibration、dataset 或主实验 retrieval ladder。

每项新增方法投入上限两个工作日，无法跑通时记录具体证据；不能把未运行写成零分。IRCoT-inspired 必须记录每轮生成推理及其实际用于检索的 query，不能重复原问题增加 K。

IRCoT-inspired Dev32 的 trace 证实 post-first-round query 来自上一轮推理，共有 50 个 changed-query continuation rounds；但 17/32 题最终重复 query，只有 15/32 是无重复有效轨迹，Answer F1 为 0.0625。因此“形成了真实迭代机制”成立，“形成了可靠统一环境 baseline”不成立。

## Sources

- Adaptive-RAG: https://aclanthology.org/2024.naacl-long.389/
- IRCoT: https://aclanthology.org/2023.acl-long.557/
- SIM-RAG: https://doi.org/10.1145/3726302.3730018
- S2G-RAG: https://aclanthology.org/2026.acl-long.1185/
- Stop-RAG: https://arxiv.org/abs/2510.14337
- SURE-RAG: https://arxiv.org/abs/2605.03534

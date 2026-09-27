# Zcode 写作交接入口

本包用于中文论文 V1 和组会汇报制作。实验阶段已经结束。`READY_FOR_WRITING = YES` 表示写作证据已具备，不表示所有风险目标都达到。

## 1. 获取正确版本

仓库：<https://github.com/s2079481035/RAG>

交接分支：`exp/external-related-baselines`。不要默认使用 `main` 或较早的 Phase 3 分支。

```bash
git clone --branch exp/external-related-baselines --single-branch https://github.com/s2079481035/RAG.git
cd RAG
git rev-parse HEAD
git status --short
```

已有本地仓库时，先检查未提交修改，再切换对应分支；不要强制覆盖。实际写作记录当前完整 commit 到输出的 `source_manifest.md`。

核验基线：Phase 4 最终结果提交 `cc6951529fe77811bb287ed885b19ddbe9f7059c`；外部方法审计提交 `81b1721`。写作交接提交在其后，不代表实验重新运行。

## 2. GitHub 中有什么

2026-09-27 核对时，当前分支已提交代码与远端一致；`docs/paper_v1/` 当时仍未跟踪，本次将其与交接包一并纳入版本控制。

| 材料 | GitHub / 迁移方式 | 本次用途 |
|---|---|---|
| 研究代码、配置、冻结协议、结果 CSV/JSON、审计 Markdown | 当前分支已有跟踪文件 | 方法与数字溯源 |
| `docs/paper_v1/` 五份写作材料 | 随本次交接提交 | 论文主线、清单、结论、相关工作、大纲 |
| `docs/handoff_zcode/` | 随本次交接提交 | 接手约束和执行提示词 |
| `figures/phase3b/`、`figures/phase4/` | 已跟踪 | 核验来源后选用或重绘 |
| 模型权重、`experiments/`、原始数据、FAISS 索引、大型逐样本 JSONL、第三方克隆 | 多数被 `.gitignore` 排除；服务器另行保管 | 写作无需下载；不能声称完整复现实物已在 GitHub |
| 服务器新产生但未提交的文件、仓库外文件 | 不在本次本地 Git 核验范围 | 不把本地同步等同于服务器全盘备份 |

用 GitHub 导入 Zcode 即可开始写作。不要为了填补忽略文件而重新训练、推理、bootstrap 或评估。普通网页链接导入若不能读取完整仓库，应使用克隆或仓库 ZIP。

## 3. 建议顺序

1. 先执行 [接手与证据核对提示词](01_intake_prompt.md)。
2. 再执行 [中文论文 V1 提示词](02_paper_prompt.md)。
3. 执行 [组会 PPT 提示词](03_slides_prompt.md)，其数字仍须直接回查冻结表。
4. 执行 [交付审校提示词](04_review_prompt.md)。

每次只给一个阶段任务，便于保留上下文和检查成果。提示词文件已经包含输出路径，无需把长聊天历史全部复制给 Zcode。

## 4. 阅读顺序与来源优先级

先读 `docs/paper_v1/paper_outline.md`、`claim_evidence_matrix.md`、`research_story_evidence_map.md`、`result_inventory.md`、`related_work_positioning.md`。

然后核验原始冻结来源：

| 主题 | 主要来源 |
|---|---|
| Phase 1/2 输入消融 | `results/critic/comparison_20260902T125642Z/critic_comparison.csv`；`results/phase2/controller_baselines.csv`；`docs/phase2_analysis.md` |
| Phase 3A 多 seed、辅助任务、采样 | `results/phase3/multi_seed_summary.csv`；`ablation_summary.csv`；`sampling_summary.csv`；`docs/phase3a_closing_report.md` |
| Phase 3A 固定 seed 配对统计 | `results/phase3/bootstrap/paired_comparison_summary.csv` |
| HotpotQA 风险与端到端结果 | `results/phase3b/test_risk_policy.csv`；`bootstrap_ci.csv`；`docs/phase3b_analysis.md`；`docs/phase3b_closing_report.md` |
| 2Wiki 最终表 | `results/phase4/final/` 中的 CSV；`docs/phase4/2wiki_heldout_analysis.md`；`phase4_closing_report.md`；`cross_dataset_analysis.md` |
| 2Wiki ceiling/rescue | `results/phase4/2wiki/heldout/stage_rescue_analysis.csv`；`retrieval_ceiling.json`；`results/phase4/final/table_retrieval.csv` |
| 外部相关方法 | `docs/external_baselines/`；`results/external_baselines/external_baseline_manifest.json` |

同一实验的数字以冻结 CSV/JSON 及其 manifest 为准；统计区间以保存的 bootstrap 表为准；方法描述回查对应版本配置和代码。写作材料是索引，不覆盖原始证据。出现冲突时保留待核实标记，不自行选“更好看”的数字。

`docs/experiment_audit.md` 是早期历史审计，不能单独代表最终系统。较早的 `GO`、`READY` 状态也不能覆盖最终统计限制。

## 5. 必须保留的结论边界

- Phase 3A 的分类 FSR 与 Phase 3B/4 的 sequential reached-actionable FSR 不可直接拼接成“从 X 降到 Y”。后者只统计 Dense/Hybrid 的真实决策，terminal Rerank 不完整计入 TRFR。
- HotpotQA Risk10 FSR 为 `0.08445945945945946`，95% CI `[0.05638965371174415, 0.11765174222555261]`。点估计低于 10%，区间未全低于 10%。
- 2Wiki Risk10 FSR 为 `0.12310521976279572`，95% CI `[0.11810921353389348, 0.12837358336460408]`。区间全高于 10%，冻结策略未达到目标。
- 2Wiki FixedHeavy 完整证据覆盖率为 `0.39782124681933845`，TRFR 为 `0.6021787531806615`。这是当前 corpus/ladder 的结果，不是数据集不可解决性的证明。
- Coverage Auxiliary 的固定 seed 配对收益不能替代多 seed 结论；Hard Partial sampling 无稳定改善；Temperature Scaling 未证明独立部署收益。
- Adaptive-RAG-style Router 是同 ladder 下的适配基线，不是官方完整复现。S2G-RAG 为 PARTIAL、无有效推理数字；Stop-RAG/SURE-RAG 为方法审计。缺失结果不能填 0。
- 2Wiki 是域内训练后在官方 labeled Dev 上 heldout 评价，不是 Hotpot checkpoint 零样本迁移；corpus 为 shared benchmark-context，不是 Full Wikipedia。
- HotpotQA Test 在前期已被检查，不能包装成各阶段完全独立的确认性测试。
- 总延迟采用报告中的组合测量口径；不要改写成在线服务吞吐或部署 SLA。
- 人工审计的一致性受样本、分层、单 reviewer 与 full/packed evidence 区别限制；不证明总体标签无误。

## 6. 写入范围

新材料只写入 `writing/paper_v1/`、`writing/meeting_v1/`、`writing/review/`。现有 `results/`、实验 `configs/`、`scripts/`、冻结报告与数据保持不变。仅阅读、引用、排版和可视化已有数字，不新增实验、不提出 Phase 5、不生成新的统计指标。

图表可把已有比例显示为百分数、统一四舍五入；保留原始值和来源。不得自行补算置信区间、显著性、相对提升率或重新平均 seed。参考文献只允许核对原论文/正式页面的元数据，不允许用网上其他实验数值填入本项目结果。

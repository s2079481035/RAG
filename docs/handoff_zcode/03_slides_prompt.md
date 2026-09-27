# 提示词 3：制作组会汇报 PPT

---

请利用本仓库的冻结研究材料制作中文组会 PPT，默认 15–20 分钟、约 16 张主幻灯片，加必要备份页。直接交付可编辑 `.pptx`，不能只给大纲。

先读 `docs/handoff_zcode/README.md`、`docs/paper_v1/` 和已生成的 `writing/paper_v1/`。论文草稿仅用于组织语言，数据须回查冻结结果。遵守禁止新实验与修改指标的要求。

输出到 `writing/meeting_v1/`：

- `rag_research_meeting_v1.pptx`
- `rag_research_meeting_v1.pdf`（环境支持时）
- `speaker_notes.md`：逐页讲稿与预计用时。
- `slide_evidence_map.md`：每页 claim、图表、数据源、行标识/键、结论限制。
- `build_slides.js` 或等价可复用生成源码及实际依赖说明。
- `renders/`：逐页预览和总览；`qa_report.md`：内容与版式检查结果。

建议叙事顺序，可合并相邻页控制时间：

1. 题目和要回答的研究问题。
2. 为什么相似/相关证据不等于完整证据。
3. Dense → Hybrid → Rerank 三阶段与累计证据示意。
4. FSR、terminal failure、recoverable false stop 的定义和范围。
5. Evidence-aware critic 与 score-aware packing、coverage head。
6. 数据集、冻结时间线和评价设置。
7. Query/Stage 与 Evidence 输入消融。
8. Hard Partial：样本规模、错误集中程度和条件错误率。
9. Auxiliary/sampling 多 seed 结果及负结果。
10. HotpotQA 固定重检索与冻结 Risk10：质量、chunks、reranker、延迟。
11. 两个数据集的风险目标和已有 95% CI，标出 10% 目标线。
12. 2Wiki retrieval ceiling 与阶段 rescue。
13. 2Wiki quality-cost 和风险失败应如何同时解读。
14. ScoreThreshold、LLM Judge、Adaptive-RAG-style Router。
15. S2G/Stop/SURE 方法边界与审计状态；不放不存在的性能柱状图。
16. 当前支持什么、不能声称什么，以及需要导师讨论的写作选择。

视觉要求：16:9，中文字体检查缺字与替换；正文尽量不少于 20pt；每页一个主论点，方法用流程图、结果用图表、限制用简洁对照。稳定使用方法颜色，成本与质量分轴/分图避免误导。比例与百分数标注统一，CI 来源必须明确。无需装饰性图片。尽量使用原生可编辑图表和形状，复杂科学图可插入清晰 SVG/PNG。

每页页脚提供短来源，备注区给完整仓库路径和讲稿。保留模型/语料/测时口径限制。不能把重绘当成重算：直接使用冻结值与既有区间，不生成新的 bootstrap 或显著性结论，不把缺失结果填零。

演示文件制作后必须：

1. 验证 PPTX 可打开、图表和备注存在。
2. 导出或渲染所有页，检查文字裁切、元素重叠、中文字体、图表轴标、图片和页码。
3. 将问题修复后再次检查相关页。
4. 核对每页数字与 `slide_evidence_map.md`。

若缺少渲染工具，明确在 qa_report 中标记“未完成视觉验证”，仍交付可编辑文件和源代码，不声称检查已通过。不要依赖原机器上的 Codex 私有技能路径，使用当前环境实际可用工具。

# 提示词 2：生成中文论文 V1

---

请基于本仓库冻结证据，完成一篇面向《电子与信息学报》的中文论文 V1，直接写入文件。遵守 `docs/handoff_zcode/README.md` 和 `writing/review/intake_report.md`。若后者不存在，先完成 `01_intake_prompt.md` 的接手任务，再继续。

交付：

- `writing/paper_v1/manuscript_v1.md`：完整中文初稿，含候选标题、中英文摘要、关键词、正文、图表标题、参考文献。
- `writing/paper_v1/evidence_ledger.md`：每个表格数字、摘要数字、强 claim 对应源文件、行标识/键、split、seed、指标范围及保存的统计证据。
- `writing/paper_v1/references.bib`：已核实参考文献；未核实项不编造 BibTeX 信息。
- `writing/paper_v1/author_questions.md`：作者需要决定的事项，最多 10 个。
- `writing/paper_v1/figures/`：从已有冻结数据选用或重绘的主图，以及作图源代码和数据来源说明。
- 如环境支持 Word，再提供 `manuscript_v1.docx` 并渲染检查。不能导出时明确说明，不把 Markdown 改后缀冒充 Word。

以 `docs/paper_v1/paper_outline.md` 为骨架，覆盖引言、相关工作、问题定义、方法、实验设置、实验结果、讨论、局限、结论。建议主线为“证据充分性控制的风险与效率边界”，它是写作组织，不预设创新已被证明。

方法必须准确区分当前 evidence sufficiency 和未来 retrieval utility；解释三阶段累计证据、可决策 stage、terminal forced stop、score-aware packing、覆盖率辅助监督、Dev 校准与保守阈值选择。4.3 可命名为“Hard Partial 建模与诊断”，避免把失败的 sampling 写成成功核心组件。

实验设置交代数据来源、split、shared benchmark-context corpus、模型、固定 generation、seed 和 question-level bootstrap。所有公式和 FSR 分母须回查代码/冻结协议，明确 classification 与 sequential policy 指标不同。

结果必须同时保留：

1. Evidence state 相对 query/stage 输入的证据。
2. Hard Partial 的错误数量与条件错误率区别。
3. Coverage Auxiliary 固定 seed 与多 seed 结论不一致，不能只报有利 seed。
4. HotpotQA 质量成本权衡，FSR 点估计达标但 CI 未全低于 10%；非劣依据预注册 margin，不是“差异不显著所以相等”。
5. 2Wiki 冻结 Risk10 明确未达 10%，同时报告已有 Answer F1 与成本结果。
6. Retrieval ceiling、rescue、RFSR/TRFR 与 evidence sufficiency 的区别。
7. ScoreThreshold、LLM Judge、适配 Router 的适用边界。
8. S2G-RAG、Stop-RAG、SURE-RAG 仅使用实际审计状态和方法定位，不宣称数值胜出。

写作要求：

- 使用自然、克制的中文学术表达，避免“首次”“全面优于”“可靠风险保证”“显著提升”而无直接证据。
- 原表数字是唯一数值依据。只做显示精度和比例到百分数转换，不新增指标、相对改善率、均值或 CI。
- 摘要同时体现收益与 2Wiki 风险泛化失败，不把负结果藏到附录。
- 作者姓名、单位、基金、伦理声明等未知信息使用 `[作者补充]`，不从路径用户名推断。
- 正式参考文献可联网核对原论文和官方出版页面；不能核实就标记，不杜撰作者、年份、页码、DOI。目标期刊格式只有取得官方模板后才宣称符合。
- `docs/paper_v1/` 是证据导航，最终数字回查 CSV/JSON；遇冲突在 evidence ledger 标记，正文暂留待核实。
- 不需要新实验，不运行任何实验入口，不修改冻结文件。正文没有材料的论点应删除、缩窄或标 UNSUPPORTED。

完成前逐条核对摘要、结论、所有主表与 evidence ledger。只改新写作产物。输出完整初稿，不停留在大纲或建议。

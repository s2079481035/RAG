# 提示词 4：论文与 PPT 交付前审校

---

请以审稿人和数据核查者视角，审校 `writing/paper_v1/` 与 `writing/meeting_v1/`，不新增实验、不改冻结来源。

生成 `writing/review/final_review.md`，先列问题，按严重程度排序，每项给论文段落或幻灯片页码、原表路径和修正建议。可直接修正写作文件中的事实、引用和排版错误，不能替作者决定尚未确认的研究主张。

重点检查：

- 每个数字可定位到冻结 CSV/JSON 的具体行/键；显示精度之外没有新计算；没有把 Dev 结果写为 heldout。
- HotpotQA Test 的重复使用说明保留；2Wiki 域内训练没有写成零样本跨域。
- 2Wiki Risk10 FSR 区间全高于 10% 的失败结论出现在摘要、结果和 PPT 中。
- 非劣、相等、显著优于、风险保证四种结论没有混用。
- 固定 seed 的 paired bootstrap 没有冒充三 seed 稳定收益。
- Temperature Scaling、Coverage Auxiliary、Hard Partial sampling 的负结果完整。
- sequential FSR 只包含可决策 stage，TRFR 不混进 Controller 错误；RFSR 作为诊断，没有替换主 FSR。
- 2Wiki ceiling 仅对应冻结 corpus/ladder，Full evidence 与 packed evidence 清楚区分。
- Query+Stage 与 Adaptive-RAG-style Router 是不同基线；适配复现与官方原版区分。
- S2G PARTIAL/N=0 不当作零性能；Stop/SURE 无数值胜负结论；没有依赖未运行的实验。
- 参考文献确实存在且支持对应句子；未经核实的元数据清楚标记。
- PPT 已逐页渲染、图表可读、讲稿与图表一致；Word/PDF 若存在则检查页边距、截断与表格。
- `git diff --name-only` 和工作区状态没有本次对实验代码、配置、数据、冻结结果的改动；历史已有修改单独记录。

报告末尾给 `DRAFT_READY_FOR_AUTHOR_REVIEW = YES/NO` 和 `SLIDES_READY_FOR_MEETING = YES/NO`，列明阻塞项。作者信息和期刊模板未确认时，不能称为最终投稿版。请把审校做完并交付文件，不只输出通用建议。

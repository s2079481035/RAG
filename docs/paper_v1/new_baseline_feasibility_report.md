# 新增 baseline 可行性记录

2026-10-09；状态：准备中，未完成 GPU smoke。

## SIM-RAG

官方代码 commit `3d5f13a8b53764549d1d5b013bbd25c5e17b9c8e`；https://github.com/ucscirkm/SIM-RAG。

官方 https://huggingface.co/dyang39/SIM-RAG-Llama3-2B 有 safetensors、tokenizer、config。远端配置已读取：T5ForConditionalGeneration，24 encoder/24 decoder layers，d_model=2048，d_ff=5120，32 heads，float32 checkpoint。论文对应 Flan-T5 2.85B，不能按名字误认成 Llama 2B。权重发布约 11.4GB；转 BF16 仅权重估算约 5.7GB，实际显存还需 activations/cache，尚未实测。

`inference/inference.py::call_gate_batch` 输入包含 Question/context、Answer、Rationale，生成文本恰好为 `1` 时接受。必须保留 candidate/rationale，不能仅把 query/evidence 塞给新二分类器。代码还在 verdict 过滤前对整批发起下一轮 query/search；若移除这些额外工作必须标注优化差异，不能少算官方执行成本。

Tokenizer 加载代码假定 checkpoint 路径后有 `_tokenizer` sibling，而 HF release 包含 tokenizer 本身；部署须显式适配路径。当前状态是 CHECKPOINT_AVAILABLE_NOT_RUNTIME_VERIFIED，不是成功复现。

## IRCoT

官方代码 commit `3c1820f698eea5eeddb4fba3c56b64c961e063e4`；https://github.com/StonyBrookNLP/ircot。

新增 `scripts/run_recent_rag_dev.py` 为 inspired implementation：使用冻结共享 BM25 索引、原 Qwen generator 与最终 answer evaluator；每次产生一个推理句，并将该句实际用于下轮检索。保留 query、retrieved IDs、reasoning、calls、tokens、latency。最大 8 轮、top2、最多15累计 chunks。显式 adaptation 是 zero-shot 提示与 sentence chunks，不能称官方复现。

Dev ID 按固定前缀 SHA-256 排序，32 为预先冻结100的前缀。脚本将重复/空 continuation query 明确记为 invalid 并结束该题检索，汇总有效轨迹率；超预算 prompt 仍直接失败。失败不得当作成功 smoke。未完成运行前只可宣称机制已实现，不可宣称有效、结果可靠或增强 QA。

## 服务器与资源

服务器 `/home/sunjb/RAG/RAG_paper` 当前为 `research/sufficiency-phase4`，初查 HEAD `cc69515`。两张4090各24GB；GPU0约15.5GB已占用，GPU1约8.5GB已占用，任务属于其他用户。保留其任务和服务器未跟踪文件。可用内存约206GiB。

GPU 共享策略已询问用户，回复前只运行 CPU preflight。后续必须本地 commit/push → 服务器 Git 拉取相同 commit → 执行实验，禁止临时 scp 修改服务器脚本。新增输出独立目录，主实验冻结文件不写入。

不重训 Adaptive Router / S2G / Stop-RAG / SURE。S2G 的已有 PARTIAL 状态保留。SIM、IRCoT各最多两个工作日，实际未到截止不能用预算作为已失败理由。

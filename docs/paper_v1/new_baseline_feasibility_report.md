# 新增 baseline 可行性记录

2026-10-10；状态：两个 Dev32 GPU smoke 与质量审计均已完成。

## SIM-RAG

官方代码 commit `3d5f13a8b53764549d1d5b013bbd25c5e17b9c8e`；https://github.com/ucscirkm/SIM-RAG。

官方 https://huggingface.co/dyang39/SIM-RAG-Llama3-2B 有 safetensors、tokenizer、config。远端配置已读取：T5ForConditionalGeneration，24 encoder/24 decoder layers，d_model=2048，d_ff=5120，32 heads，float32 checkpoint。论文对应 Flan-T5 2.85B，不能按名字误认成 Llama 2B。权重发布约 11.4GB；转 BF16 仅权重估算约 5.7GB，实际显存还需 activations/cache，尚未实测。

`inference/inference.py::call_gate_batch` 输入包含 Question/context、Answer、Rationale，生成文本恰好为 `1` 时接受。必须保留 candidate/rationale，不能仅把 query/evidence 塞给新二分类器。代码还在 verdict 过滤前对整批发起下一轮 query/search；本 runner 只对 Reject 样本继续生成 query，已在 adaptation 中明确记录，并完整统计实际 reasoner/query/critic 调用。

Tokenizer 加载代码假定 checkpoint 路径后有 `_tokenizer` sibling，而 HF release 包含 tokenizer 本身；`scripts/run_sim_rag_dev.py` 从同一官方 HF snapshot 加载二者。该 adapted runner 保留 Question/context + Answer + Rationale → Accept/Reject，并在 Reject 后生成新 query；reasoner、query generator 与 Critic 的 calls/tokens 全部计入。

官方 snapshot 固定为 HF commit `0e1cc45ab595543557837b237020ebcd16aeb357`。服务器通过 `hf-mirror.com` 下载后，与本地 snapshot 核对三个 safetensors 和 index 的 SHA-256，全部一致。Server commit 为 `33b82bf57e3fe72180d5d0e7431f92c3df38e8cb`；输出为 `results/paper_v1/sim_rag_dev32/`。

### 32-question result

| Metric | Value |
|---|---:|
| Questions / unique IDs | 32 / 32 |
| Answer EM / F1 | 0.53125 / 0.546875 |
| SF recall / complete coverage | 0.5078125 / 0.21875 |
| Average retrieval rounds / chunks | 1.84375 / 3.28125 |
| Average reasoner / critic / total calls | 4.96875 / 2.84375 / 7.8125 |
| Average total tokens | 2571.65625 |
| Changed-query retrieval rounds | 59 |
| Questions with retrieval | 30 / 32 |
| Stop reasons | 22 critic accept; 9 invalid query; 1 forced horizon |
| Diagnostic latency | 141563.68 ms/question |

All 32 IDs match the pre-frozen prefix; row-level recomputation exactly matches `summary.json`. Every stored accepted flag equals `critic_raw == '1'`, and every stored valid query is unique within its question. Across 91 critic calls, 22 decode as `1`; 69 decode as blank. This blank is interpretable: the released T5 tokenizer encodes `0` as two tokens `['▁','0']`, while the official weighted branch uses `max_new_tokens=1`, so a Reject can decode only the blank prefix. It remains Reject under the official equality rule.

The checkpoint is therefore runtime-verified and the adapted mechanism is real: candidate-answer-conditioned rejection causes new queries and retrieval. It is not a reliable reproduction of the paper's Table 1 protocol because the reasoner, corpus, split, prompts, batching behavior, and released checkpoint role differ; 9/32 also terminate on an invalid query. Stop after 32 and do not extend to 100 or heldout within this supplement.

## IRCoT

官方代码 commit `3c1820f698eea5eeddb4fba3c56b64c961e063e4`；https://github.com/StonyBrookNLP/ircot。

新增 `scripts/run_recent_rag_dev.py` 为 inspired implementation：使用冻结共享 BM25 索引、原 Qwen generator 与最终 answer evaluator；每次产生一个推理句，并将该句实际用于下轮检索。保留 query、retrieved IDs、reasoning、calls、tokens、latency。最大 8 轮、top2、最多15累计 chunks。显式 adaptation 是 zero-shot 提示与 sentence chunks，不能称官方复现。

Dev ID 按固定前缀 SHA-256 排序，32 为预先冻结100的前缀。脚本将重复/空 continuation query 明确记为 invalid 并结束该题检索，汇总有效轨迹率；超预算 prompt 仍直接失败。失败不得当作成功 smoke。

### 32-question result

Server commit: `9c8369518873102879b70f2cc694cb6b0a56a01a`; output: `results/paper_v1/ircot_inspired_dev32_v2/`.

| Metric | Value |
|---|---:|
| Questions / unique IDs | 32 / 32 |
| Answer EM / F1 | 0.0625 / 0.0625 |
| SF recall / complete coverage | 0.53125 / 0.15625 |
| Average retrieval rounds / chunks | 2.5625 / 4.0625 |
| Average total LLM calls / tokens | 3.5625 / 1545.09375 |
| Changed-query continuation rounds | 50 |
| Valid iterative questions | 15 / 32 (0.46875) |
| Invalid repeated-query questions | 17 / 32 |
| Diagnostic latency | 50562.55 ms/question |

The stored trace assertion verifies that every post-first-round query equals the immediately preceding generated reasoning sentence. Thus this implementation did perform true reasoning-conditioned retrieval, rather than repeating the original question with a larger K. It is not a reliable baseline: more than half the questions repeat a continuation query, and only 2/32 have non-zero answer F1. Shared GPU execution and CPU offload also make latency non-comparable. Stop after 32; do not extend to 100 or heldout without redesigning the prompt/adaptation, which would exceed this frozen supplement's scope.

## 服务器与资源

服务器 `/home/sunjb/RAG/RAG_paper` 已快进到 `exp/external-related-baselines` 的 `33b82bf57e3fe72180d5d0e7431f92c3df38e8cb` 后运行。两张4090各24GB；选择启动时有约17.1GB空闲的 GPU1，reasoner 限额10GiB并保留安全余量；其他用户任务未被停止或修改。Critic 在 CPU float32 运行，reasoner 部分 CPU offload，所以 latency 仅作诊断。

执行遵循本地 commit/push → 服务器 Git 拉取相同 commit → 实验。服务器未直接修改代码；结果复制回本地后提交。新增输出使用独立目录，主实验冻结文件未写入。服务器上与 Git 冲突的旧 IRCoT 未跟踪目录先移动到仓库外备份，拉取后核对三文件 SHA-256 完全一致。

不重训 Adaptive Router / S2G / Stop-RAG / SURE。S2G 的已有 PARTIAL 状态保留。IRCoT 与 SIM-RAG 的 32 条 smoke 已回答可运行性边界；不建议继续扩跑。

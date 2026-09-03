# RAG 实验审计

审计日期：2026-09-02  
审计提交：`e6a87cf89f51a62a7f49fd314bdb64a1e78571d1`  
审计范围：当前克隆仓库中的源码、已提交数据与结果文件。本文不把历史说明中未提交的服务器文件视为当前可用实现。

## 1. 结论摘要

当前仓库已经形成了可复用的 Dense、BM25/RRF Hybrid、Cross-Encoder Rerank、二分类 Sufficiency Critic 和三阶段自适应停止实验链路，但现有 HotpotQA 口径是 **gold document ID coverage**，不是严格的 **supporting fact `[title, sentence_id]` coverage**。

这是当前研究假设与已有实现之间最重要的偏差：数据构建时把每篇 context 的全部句子拼成一个字符串，只保留临时 `hp_xxxxxx` 文档 ID；标题、句子边界和 gold supporting fact 对没有写入数据。因此，已有 Recall、FullCov、Critic label 和 False Early Stop 均只能解释为文档级指标，不能直接作为 sentence-level evidence sufficiency 结论。

审计还确认：

- HotpotQA 当前检索单位是整篇 context 文档，不是滑窗 chunk；没有 chunk size 和 overlap。
- `hotpotqa_all.json` 有 14,810 个文档实例，其中只有 13,783 个唯一文本，存在 1,027 个重复文档实例，最大相同文本重复 4 次。临时 doc ID 会把同内容的另一实例算作未命中。
- 现有 Critic 是二分类，不区分 `Insufficient / Partial / Sufficient`。
- 当前克隆没有 `data/indices/` 和 `models/*.pt`；它们被 `.gitignore` 排除。仅有历史结果和 `models/critic_config.json`。
- 当前源码没有 QA generation 或 FastAPI 服务。Qwen2.5-7B 只出现在 difficulty router 中；Answer EM/F1 暂时不可评估。
- 本地 Python 3.14 环境缺少 `datasets`、`pytest`，且 PyTorch DLL 初始化失败；GPU 实验需在原服务器环境运行。

## 2. 当前目录结构

```text
RAG/
|-- data/
|   |-- hotpotqa.json
|   |-- hotpotqa_all.json
|   |-- nq.json
|   `-- critic_samples/{train,val,test}.jsonl
|-- models/
|   `-- critic_config.json
|-- results/
|   |-- *_dense.json, *_hybrid.json, *_rerank*.json
|   |-- recall_matrix.*, stratify.*, router_eval.*
|   |-- early_stop.*, critic_eval.*, critic_policy.*
|   `-- hotpotqa_all_retrieval.json
|-- scripts/
|   |-- download_data.py, build_index.py, retrieve.py, eval_recall.py
|   |-- stratify.py, router_experiment.py, early_stop.py
|   `-- critic_data.py, train_critic.py, eval_critic.py, policy_sweep.py
|-- EXPERIMENT_PROTOCOL.md
|-- SESSION_CONTEXT.md
`-- run_all.sh
```

忽略且当前不存在的关键服务器产物：`data/indices/`、`models/critic.pt`、`models/critic_final.pt`。

## 3. 当前 Pipeline

实际已实现的检索链路是：

```text
Query
  |-- BAAI/bge-large-en-v1.5 query embedding
  |-- FAISS IndexFlatIP dense retrieval
  |-- BM25Okapi lexical retrieval
  |-- RRF(dense top-200, BM25 top-200, k=60) -> hybrid top-20
  `-- BAAI/bge-reranker-v2-m3 -> reranked top-20
```

自适应实验使用 `Dense@3 -> Hybrid@5 -> Rerank@20`，在 Dense 与 Hybrid 后调用 Critic 判断是否停止。仓库没有检索后的答案生成代码，所以用户描述中的 `Generation` 在当前提交中不是可执行阶段。

## 4. HotpotQA 数据流

1. `scripts/download_data.py` 流式读取 `hotpotqa/hotpot_qa` 的 `distractor/validation` 全部 7,405 道题。
2. 对每道题的 10 篇 context，按扫描顺序分配临时 `hp_xxxxxx`；一篇 context 的全部 sentence 用空格拼接成一个文本。
3. 若 context title 出现在 supporting-fact title 集合中，则对应临时 doc ID 被加入 `gold_docs`。
4. 使用 seed 42 shuffle 问题；前 1,000 道作为历史 test，其余 6,405 道在 Critic 数据脚本中切为 train 5,405 / val 1,000。
5. `build_index.py` 对按 doc ID 排序的 KB 文本建立 FAISS 与 BM25 索引。
6. `retrieve.py` 输出每题 `qid/gold/ranked`，不保存 title、sentence ID、检索分数或逐样本耗时。
7. `eval_recall.py` 用 `set(ranked_doc_ids) & set(gold_doc_ids)` 计算 Hit/Recall。

注意：虽然源码常量写 `HOTPOT_KB_SIZE=10000`，最终 `hotpotqa_all.json` 实际有 14,810 个文档。原因是脚本先保留所有 gold 文档实例；gold 实例本身已超过 10,000，后续不会缩减。

## 5. Chunk 与 Supporting Fact

| 项目 | 当前实现 |
|---|---|
| 检索粒度 | 每个 HotpotQA context article 为一个检索单元 |
| chunk size | 未设置；整篇 context |
| overlap | 0；没有滑窗切分 |
| tokenizer | 建索引时由 BGE tokenizer 内部处理；BM25 使用正则 `[a-z0-9][a-z0-9\\-']*`、小写、停用词过滤 |
| 多 sentence | 是；所有 sentence 被拼接 |
| document title | 未保存 |
| sentence ID | 未保存 |
| chunk ID | 只有临时 doc ID `hp_xxxxxx`，可视为旧版 chunk ID |

因此，当前数据无法直接验证 `[title, sentence_id]` 是否被覆盖。必须从原始 HotpotQA 按相同扫描顺序重建一个不覆盖旧文件的 metadata sidecar：

```text
doc_id -> title, sentence_ids, sentences
qid -> gold_supporting_facts
```

因为当前每个检索单元包含文章的全部 sentence，严格命中规则仍为：title 相同且 gold `sentence_id` 属于该检索单元的 `sentence_ids`。后续若切成真正 chunk，该规则无需改变。

## 6. 检索配置

### Dense

- 模型：`BAAI/bge-large-en-v1.5`
- 文档与 query embedding：`normalize_embeddings=True`
- batch size：64
- 索引：`faiss.IndexFlatIP`
- 文档排序：构建索引前按 doc ID 字典序排序
- 候选：Dense 直接搜索最大 K；Hybrid 为获得全局 dense rank 搜索整个索引

### BM25

- 实现：`rank_bm25.BM25Okapi`
- 参数：库默认参数，源码没有显式设置 `k1/b/epsilon`
- 分词：英文数字正则、小写、删除 34 个硬编码停用词、删除单字符 token
- 候选：按 BM25 score 取前 200

### RRF

- `RRF_K=60`
- Dense top-200 与 BM25 top-200 各贡献 `1 / (60 + rank + 1)`
- 融合后取 top-20
- `rrf_merge` 的 `bm25_idx` 参数未使用，不影响当前结果

### Reranker

- 当前默认模型：`BAAI/bge-reranker-v2-m3`
- 兼容旧模型：`BAAI/bge-reranker-base`
- 输入：`(question, full document text)`
- 候选池：Hybrid top-20
- max length：512
- 输出 K：结果文件保存 20 个排序文档；评估时切 `@3/@5/@10/@20`

## 7. 当前 Cross-Encoder Critic

| 项目 | 当前实现 |
|---|---|
| 输入 | `question + " [SEP] " + concatenated docs` |
| evidence | Dense@3 / Hybrid@5 / Rerank@20；每篇文档先截断到 48 words |
| backbone | `BAAI/bge-reranker-base` |
| head | 二分类 linear head |
| 标签 | 当前 ranked doc IDs 覆盖全部 gold doc IDs 为 1，否则为 0 |
| loss | PyTorch cross entropy |
| optimizer | AdamW, lr 默认 `2e-5` |
| threshold | softmax sufficient probability `>=0.5`；Phase 3b 在 val 上选择 Dense/Hybrid 阈值 |
| split | seed 42；train 5,405 q，val 1,000 q，test 1,000 q；每题 3 个 stage sample |
| checkpoint 选择 | val binary sufficient F1 最优 epoch |
| max length | 512 tokens，tokenizer 静默 truncation |

当前训练只显式固定了数据 shuffle seed；没有统一设置 Python、NumPy、PyTorch、CUDA 和 DataLoader worker seed。也没有保存逐样本 token length、truncation 标志、训练环境、模型 revision、开始/结束时间和完整训练 config。

## 8. 现有指标口径

### Recall@final / FullCov

- `Recall = |retrieved doc IDs ∩ gold doc IDs| / |gold doc IDs|`，再对问题取宏平均。
- `FullCov = 1` 当且仅当当前 ranked doc ID 集合包含全部 gold doc IDs。
- 这两个指标目前都是文档级，不是 supporting-fact sentence 级。

### 平均检索数量

- Fixed 策略直接使用 K。
- 自适应策略按最终停止阶段的 K 计成本，例如 Dense@3 计 3、Hybrid@5 计 5、Rerank@20 计 20。
- 该口径不是累计处理文档数，也没有去重后 article title 数。

### Latency

`measure_latency.py` 对 100 个 query 测量 wall-clock stage latency，历史均值为：embedding 22.12 ms、dense@20 9.37 ms、hybrid@20 71.38 ms、rerank 20 pairs 127.40 ms。Critic E2E 报告使用硬编码累计延迟 Dense 31.5 ms、Hybrid 93.5 ms、Rerank 220.9 ms，再加实时 Critic 平均调用耗时。

局限：没有逐样本 latency；硬编码数来自 2026-08-19 的服务器测量；没有 warm-up/同步细节和硬件元数据，不能直接与新实验混合比较。

## 9. 泄露与划分审计

已验证提交数据：

- train/val/test qid 交集均为 0。
- train/val/test question text 交集均为 0。
- train 16,215、val 3,000、test 3,000 stage samples，分别对应 5,405/1,000/1,000 个问题。
- `policy_sweep.py` 在 val 上选 threshold，再只在 test 评估一次；未发现 test 直接参与阈值搜索。

潜在风险：

1. 所有问题共享同一个 KB，且 KB 包含 train/val/test 的 gold context。这是开放域检索设置允许的 corpus 共享，不等同于 question-label 泄露；但文档实例由全体问题构建，同文章重复 doc ID 会影响检索与指标。
2. 旧 Critic label 和阈值都基于 doc-ID coverage。切换为 sentence-level label 后，旧 checkpoint 和旧 policy threshold 不可直接沿用。
3. 每题三个 stage sample 高度相关。当前按 qid 切分是正确的；后续任何 sample-level random split 都会造成泄露，必须禁止。
4. 旧训练没有完整随机性固定，严格重复训练可能有波动。
5. `test_gold.json` 只保存 doc IDs；不能用于新的 sentence-level评估。

## 10. 标签与数据分布（旧文档级口径）

HotpotQA 7,405 道题均有 2 个 gold document titles；supporting sentence 数量分布为：2: 4,990，3: 1,774，4: 537，5: 80，6: 14，7: 9，8: 1。由此可见，二分类“是否覆盖所有 gold 文档”会丢掉明显的 partial 状态，三分类规则有合理性。

旧二分类 sufficient 比例：

| Split | Dense@3 | Hybrid@5 | Rerank@20 |
|---|---:|---:|---:|
| train | 67.2% | 78.6% | 90.5% |
| val | 68.8% | 81.2% | 91.8% |
| test | 68.4% | 79.2% | 91.4% |

旧标签 trajectory 还存在 `Sufficient -> Insufficient -> Sufficient` 和 `Sufficient -> Insufficient -> Insufficient`。这不是标签代码错误，而是各 stage 的 top-K 集合并非嵌套集合：Hybrid@5 可以移除 Dense@3 的 gold 文档。后续 trajectory 报告必须保留这一现象，不能假设 evidence 随 stage 单调累积。

## 11. 需要最小修改解决的问题

1. 新增 HotpotQA metadata sidecar，保留 title、sentence IDs、gold supporting facts，不覆盖旧数据。
2. 新增纯函数 supporting-fact 映射与去重计数，并用自动测试锁定 Recall `<=1`、Complete Coverage 和 RRF union 行为。
3. 新增统一 baseline evaluator，读取旧 ranking 但输出新的逐样本 schema；缺失的旧 score/逐样本 latency 必须显式标为 unavailable/estimated。
4. 新增三分类 sufficiency dataset builder；split 严格复用现有 qid 划分。
5. 新增共享 Critic 配置与统一训练脚本，通过 `input_mode=query_only|query_evidence` 保证两组实验除输入外一致。
6. 记录 truncation、随机种子、环境、git hash、模型 revision、运行时间和逐样本预测。

## 12. 当前不可直接完成的运行项

当前 Windows 环境是 Python 3.14.4；`datasets` 和 `pytest` 未安装，PyTorch 因 `c10.dll` 初始化失败不可用，且克隆仓库没有 FAISS/BM25 索引和 Critic checkpoint。因此：

- 可在本地完成静态审计、纯 Python 数据/指标测试和基于已提交 ranking 的非模型处理。
- metadata sidecar 需要在具备 Hugging Face `datasets` cache/网络的服务器上从原始 HotpotQA 重建。
- Query-only 与 Query+Evidence Cross-Encoder 训练和真实推理指标需要在 RTX 4090 24GB 服务器环境运行。
- 在这些命令真实完成前，不报告新的三分类 Critic 数字。

## 13. 审计后实施状态

审计完成后，已使用官方 HotpotQA distractor validation Parquet 在本地重建 metadata sidecar。14,810 个 chunk 的文本全部通过与旧 KB 的严格一致性检查，现已可以计算 sentence-level Supporting Fact Recall 和 Complete Evidence Coverage。新增实现和真实 baseline 结果记录于 `docs/experiment_log.md`；本节不回写或替换上述旧实现审计结论。

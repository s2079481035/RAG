# Experiment Log

## 2026-09-02: Sentence-level audit and Phase-1 scaffolding

### 修改

- 完成旧数据、检索、Critic、划分、指标与服务器产物审计。
- 从官方 HotpotQA distractor validation Parquet 重建 sentence-level metadata sidecar，并与旧 KB 逐文档严格核验。
- 实现 `[title, sentence_id]` supporting-fact coverage、Complete Evidence Coverage、三分类标签和 split/threshold 安全检查。
- 生成五阶段 sufficiency trajectory 数据和统一 fixed retrieval baseline。
- 实现共享配置的 Query-only / Query+Evidence 三分类 Cross-Encoder 训练与比较入口。
- 新增 8 个自动测试；全部通过。

### 数据核验

- 官方 Parquet：27,452,575 bytes，SHA256 `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`。
- 旧 `hotpotqa_all.json` SHA256：`0c1b8e61d801bbaa1e90b030caf61af3d9d80a3b6983c057f61879272659d284`。
- metadata：14,810 chunks，7,405 questions；所有旧 KB 文本均匹配。
- split：train 5,405 / dev 1,000 / test 1,000 question IDs，交集为 0。

### Supporting-fact 数量分布

| Gold facts | Questions |
|---:|---:|
| 2 | 4,990 |
| 3 | 1,774 |
| 4 | 537 |
| 5 | 80 |
| 6 | 14 |
| 7 | 9 |
| 8 | 1 |

### Test 三分类标签分布

| Stage | Insufficient | Partial | Sufficient |
|---|---:|---:|---:|
| Dense@5 | 14 | 207 | 779 |
| Dense@10 | 8 | 159 | 833 |
| Hybrid@5 | 5 | 197 | 798 |
| Hybrid@10 | 4 | 133 | 863 |
| Rerank@20 | 2 | 83 | 915 |

异常：Insufficient 类极少，原因是当前 retrieval unit 是包含整篇文章全部句子的 context document；一旦命中文章，通常覆盖该文章内全部 supporting sentences。三分类实验必须报告 Macro F1 和每类指标，不能只看 Accuracy。

另有 72/1,000 条 test trajectory 非单调。原因是 Dense、Hybrid、Rerank 的 top-K 不是累积 evidence set，后续 stage 可能移除前一 stage 已覆盖的 evidence。这是真实策略行为，不应在标签构建时强制改成单调。

五个 stage 合并后，train/test 的 Insufficient 样本分别只有 153/27,025（0.57%）和 33/5,000（0.66%）。此外，train/dev/test 分别有 22.2%/20.0%/20.8% 的问题在不同 stage 具有不同 Sufficiency 标签。Query-only Critic 对同一道题始终接收相同输入，因此无法从输入中识别这部分 stage-dependent label；这正是核心对照要检验的限制。

### Sentence-level Retrieval Baseline（test n=1,000）

| Baseline | SF Recall | Complete Coverage | Avg Chunks | Avg Unique Titles | Latency ms | Reranker Calls |
|---|---:|---:|---:|---:|---:|---:|
| Dense@5 | 0.8831 | 0.779 | 5.0 | 4.536 | 31.5* | 0 |
| Dense@10 | 0.9138 | 0.833 | 10.0 | 8.939 | 31.5* | 0 |
| Hybrid@5 | 0.8976 | 0.798 | 5.0 | 4.603 | 93.5* | 0 |
| Hybrid@10 | 0.9304 | 0.863 | 10.0 | 9.116 | 93.5* | 0 |
| Rerank@20 | 0.9567 | 0.915 | 20.0 | 18.234 | 220.9* | 1 |

`*` Latency 是旧服务器阶段均值估计，不是本轮逐样本实测。当前提交没有 generation pipeline，因此 Answer EM/F1 未评价。

### 当前未运行

Query-only 与 Query+Evidence Critic 尚未在本地训练。本地 Python 3.14 的 PyTorch DLL 无法初始化；按实验设计，应在用户的 RTX 4090 24GB 服务器环境运行。没有填入任何预测性 Critic 数字。

### 复现命令

```bash
python3.12 -m unittest discover -s tests -v
python3.12 scripts/build_hotpotqa_metadata.py --source-parquet /path/to/0000.parquet
python3.12 scripts/build_sufficiency_dataset.py
python3.12 scripts/evaluate_retrieval.py
bash run_phase1_critics.sh
```

# 自适应检索企业知识问答系统

面向企业知识库问答和开放域多跳问答场景的自适应 RAG 项目。项目围绕 HotpotQA / NQ 数据，搭建了从检索召回、混合排序、证据充分性判断到实验评估的完整链路，重点解决固定 Top-K 检索在简单问题上成本偏高、在复杂多跳问题上证据不完整的问题。

这是我投递“大模型应用开发 / RAG 实习生”岗位时的核心项目之一。README 重点展示可落地的工程能力：检索 pipeline 搭建、实验脚本组织、指标评估、bad case 分析和可复现记录。

## 项目概览

### 目标

- 构建 Dense / BM25 / Hybrid / Rerank 多阶段检索 baseline。
- 用 HotpotQA supporting fact 信息评估 sentence-level 证据覆盖，而不只看文档级召回。
- 训练 Sufficiency Critic 判断当前检索证据是否足够，支持自适应停止。
- 保留实验审计、数据划分检查、bad case 和对照指标，降低实验结果不可复现的风险。

### 技术栈

- Python 3.12
- PyTorch / Transformers / sentence-transformers
- FAISS / rank-bm25 / scikit-learn / NumPy
- BAAI/bge-large-en-v1.5
- BAAI/bge-reranker-v2-m3
- HotpotQA / Natural Questions

### 简历对应

- 项目名称：自适应检索企业知识问答系统
- 项目时间：2026.06 - 2026.08
- 求职方向：大模型应用开发 / RAG 实习生
- 主要能力：RAG 检索链路、Embedding 召回、Hybrid Search、RRF、Reranker、Cross-Encoder Critic、实验评估与复现

## 核心工作

### 1. 检索与排序链路

实现并整理了三类检索策略：

- Dense Retrieval：使用 `bge-large-en-v1.5` 编码 query 和文档，FAISS `IndexFlatIP` 做向量召回。
- Hybrid Retrieval：结合 Dense top-200 与 BM25 top-200，使用 RRF 融合排序。
- Cross-Encoder Rerank：对 Hybrid 候选进行二阶段重排序，提升 Top-K evidence 质量。

主要入口：

```bash
python3 scripts/retrieve.py --dataset hotpotqa --phase dense --ks 3,5,10,20
python3 scripts/retrieve.py --dataset hotpotqa --phase hybrid --ks 3,5,10,20
python3 scripts/retrieve.py --dataset hotpotqa --phase rerank --ks 3,5,10,20
```

### 2. Supporting-fact 级评估

早期实验只按 gold document ID 计算覆盖，无法严格说明是否命中 HotpotQA 的 supporting facts。为修正这个问题，项目新增了 sentence-level metadata sidecar：

- 从官方 HotpotQA distractor validation Parquet 重建 `title / sentence_id / sentence` 映射。
- 校验 14,810 个旧 KB chunk 文本与原始数据一致。
- 按 `(document_title, sentence_id)` 精确计算 Supporting Fact Recall 和 Complete Evidence Coverage。
- 保留旧结果，不覆盖历史实验文件。

相关文档：

- `docs/experiment_audit.md`
- `docs/experiment_log.md`
- `docs/sufficiency_label_report.md`

### 3. Retrieval Sufficiency Critic

针对固定 `Rerank@20` 对简单问题过度检索的问题，构造 Retrieval Sufficiency 监督数据，并训练三分类 Cross-Encoder Critic 判断当前 evidence 是否充分。

对照实验包含：

- `query_only`：只看问题本身。
- `query_evidence`：同时输入问题和当前检索到的 evidence。

两组实验使用相同数据划分、backbone、seed、optimizer、训练步数和 threshold 口径，避免因为配置不一致导致对照失真。

评估结果：

| Critic | Accuracy | Macro F1 | Sufficient F1 | False Stop Rate | Sufficient AUROC |
|---|---:|---:|---:|---:|---:|
| query_only | 0.8246 | 0.4241 | 0.8997 | 0.6675 | 0.7786 |
| query_evidence | 0.8782 | 0.7257 | 0.9273 | 0.1601 | 0.9441 |

结论：加入 evidence 后，Critic 能更可靠地判断“是否已经检索充分”，尤其显著降低了 False Stop Rate，可用于决定是否继续升级到更重的检索阶段。

### 4. Chunk 级检索评估

在 dev split 上比较 `sentence_128` 与 `sentence_256` 两种切分粒度，并对 Dense / BM25 / Hybrid / Rerank 统一评估。

部分结果：

| Variant | Method | K | SF Recall | Complete Coverage | Avg Latency ms |
|---|---|---:|---:|---:|---:|
| sentence_256 | dense | 10 | 0.9330 | 0.8700 | 5.54 |
| sentence_256 | hybrid | 10 | 0.9434 | 0.8870 | 40.76 |
| sentence_256 | rerank | 10 | 0.9581 | 0.9180 | 169.33 |
| sentence_256 | rerank | 20 | 0.9619 | 0.9250 | 169.33 |

完整结果见：

- `results/phase2/retrieval_eval/dev/retrieval_summary.md`

## 项目结构

```text
RAG/
├── configs/
│   ├── critic/
│   ├── phase2/
│   └── retrieval/
├── data/
│   ├── critic_samples/
│   └── sufficiency/
├── docs/
│   ├── experiment_audit.md
│   ├── experiment_log.md
│   └── sufficiency_label_report.md
├── results/
│   ├── baselines/
│   ├── critic/
│   └── phase2/
├── scripts/
│   ├── build_hotpotqa_metadata.py
│   ├── build_sufficiency_dataset.py
│   ├── evaluate_retrieval.py
│   ├── retrieve.py
│   ├── train_sufficiency_critic.py
│   └── evaluate_sufficiency_critic.py
└── tests/
```

## 快速开始

### 1. 安装依赖

建议在 Linux / CUDA 环境中运行完整实验。本地只做静态检查和部分纯 Python 测试也可以。

```bash
pip install torch transformers sentence-transformers faiss-cpu rank-bm25 datasets scikit-learn numpy pyarrow matplotlib
```

### 2. 运行测试

```bash
python3.12 -m unittest discover -s tests -v
```

### 3. 重建 HotpotQA sentence-level metadata

```bash
python3.12 scripts/build_hotpotqa_metadata.py \
  --source-parquet /path/to/hotpotqa/distractor/validation/0000.parquet
```

### 4. 构建三分类 Sufficiency 数据

```bash
python3.12 scripts/build_sufficiency_dataset.py
```

### 5. 评估固定检索 baseline

```bash
python3.12 scripts/evaluate_retrieval.py
```

### 6. 训练并比较 Critic

```bash
bash run_phase1_critics.sh
```

## 相关项目

我还做了一个网络配置生成方向的项目：[lora--preconfig](https://github.com/s2079481035/lora--preconfig)。该项目围绕 Cisco IOS / Juniper Junos 配置生成、配置翻译、配置补全和配置分析，比较了 LoRA 微调与 RAG 注入参考配置的效果。

可迁移到本项目的经验：

- LoRA 微调与 RAG 并不总是简单叠加增益，需要按任务类型拆分评估。
- RAG reference 可能带来参数复制问题，需要做 bad case 分析和参数级指标。
- 对照实验必须统一 prompt、token 长度、数据划分和评估脚本，否则指标不可比。

## 实习岗位可迁移能力

这个项目对应小厂日常实习中常见的几类工作：

- 接入 Embedding / Reranker 模型，搭建向量检索和混合检索 baseline。
- 编写离线评估脚本，统计 Recall、覆盖率、延迟和错误样本。
- 根据 bad case 调整 chunk、Top-K、Rerank 和停止策略。
- 整理实验记录、运行脚本和 README，方便团队复现和交接。
- 在有限 GPU / 本地 cache 环境下排查模型加载、数据构建和训练脚本问题。

## 当前边界

- 本仓库重点是检索、证据覆盖评估和 Sufficiency Critic；当前提交没有完整可运行的 QA generation / FastAPI 服务代码。
- 部分模型 checkpoint、FAISS 索引和大体积 JSONL 文件未纳入 Git，需要在服务器环境重建。
- 早期结果存在文档级覆盖口径，已在 `docs/experiment_audit.md` 中标明，后续以 sentence-level supporting-fact 评估为主。

## 面试可讲点

- 为什么只看 Top-K Recall 不够，需要区分 Partial / Sufficient evidence。
- Dense、BM25、RRF、Cross-Encoder Rerank 各自适合什么检索场景。
- Query-only Critic 为什么会在 stage-dependent label 上遇到上限。
- 如何做数据划分、防止 test 泄露和 threshold 调参污染。
- 如何通过 bad case 和审计文档把实验结论写得更可信。

# Adaptive Retrieval for Multi-hop QA

本仓库研究面向 HotpotQA 的证据充分性感知自适应检索。旧版实验实现了 Dense、BM25/RRF Hybrid、Rerank、二分类 Critic 和自适应停止；新增研究链路在不覆盖旧数据和结果的前提下，引入严格的 sentence-level supporting-fact 映射与三分类 Sufficiency Critic。

完整审计见 `docs/experiment_audit.md`，当前实验记录见 `docs/experiment_log.md`。

## Research Experiments

### 1. 环境

服务器建议使用原 Python 3.12 环境和一张 RTX 4090 24GB。核心依赖包括 PyTorch、Transformers、sentence-transformers、FAISS、rank-bm25、datasets、NumPy、scikit-learn、pyarrow 和 matplotlib。

默认模型加载为 `local_files_only=True`，会复用 Hugging Face cache。若模型保存在自定义目录，把 `configs/critic/sufficiency_cross_encoder.json` 中的 `model.name` 改为该目录；两组 Critic 必须使用同一配置。

```bash
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
python3.12 -m unittest discover -s tests -v
```

### 2. 构建 sentence-level 元数据

脚本根据原始 HotpotQA 的 `[title, sentence_id]` 重建 sidecar，并逐文档核验其文本与旧 `hotpotqa_all.json` 完全一致。默认拒绝覆盖已有 sidecar。

使用服务器 datasets cache：

```bash
HF_HUB_OFFLINE=1 python3.12 scripts/build_hotpotqa_metadata.py
```

使用已下载的官方 validation Parquet：

```bash
python3.12 scripts/build_hotpotqa_metadata.py \
  --source-parquet /path/to/hotpotqa/distractor/validation/0000.parquet
```

输出：`data/sufficiency/hotpotqa_metadata.json`。

### 3. 构建三分类 Sufficiency 数据

```bash
python3.12 scripts/build_sufficiency_dataset.py
```

从 Git 同步到服务器后，小型报告已经存在而三份大型 JSONL 被 Git 忽略，此时使用：

```bash
python3.12 scripts/build_sufficiency_dataset.py --jsonl-only
```

输出：

- `data/sufficiency/{train,dev,test}.jsonl`
- `data/sufficiency/label_distribution.json`
- `data/sufficiency/dataset_manifest.json`
- `docs/sufficiency_label_report.md`

每道题保留 Dense@5、Dense@10、Hybrid@5、Hybrid@10、Rerank@20 五个 stage。标签为：0 Insufficient、1 Partial、2 Sufficient。划分固定为 train 5,405 / dev 1,000 / test 1,000 个 question ID；test 不参与训练、checkpoint 或 threshold 选择。

### 4. 评价固定检索 Baseline

```bash
python3.12 scripts/evaluate_retrieval.py
```

输出汇总 `results/baselines/baseline_summary.csv`，逐样本结果位于 `results/baselines/per_sample/`。旧 ranking 没有保存 score 和逐样本 latency，因此输出会将 score 标为 unavailable，latency 标为历史阶段均值估计；不能将它解释为新一轮逐样本实测。

### 5. 训练核心 Critic 对照

两组训练除输入外保持一致。Query-only 只输入 question；Query+Evidence 使用 tokenizer pair 编码，采用 `only_second` 截断，优先完整保留 question，并保存原始 token 长度、截断数量和比例。

```bash
CUDA_VISIBLE_DEVICES=0 python3.12 scripts/train_sufficiency_critic.py \
  --input-mode query_only \
  --run-dir experiments/sufficiency_critic/query_only_seed42

CUDA_VISIBLE_DEVICES=0 python3.12 scripts/train_sufficiency_critic.py \
  --input-mode query_evidence \
  --run-dir experiments/sufficiency_critic/query_evidence_seed42
```

若模型不在本地 cache，可显式加 `--allow-download`。若 batch 16 在服务器环境 OOM，只修改 config 中的 `batch_size`，然后用同一个新配置从头训练两组，不能只调整其中一组。

每个 run 独立保存：resolved config、环境信息、git commit、开始/结束时间、token 长度统计、逐 epoch dev 指标和 best checkpoint。非空 run 目录不会被覆盖。

### 6. 评价和比较 Critic

```bash
CUDA_VISIBLE_DEVICES=0 python3.12 scripts/evaluate_sufficiency_critic.py \
  --query-only-run experiments/sufficiency_critic/query_only_seed42 \
  --query-evidence-run experiments/sufficiency_critic/query_evidence_seed42
```

新建的 `results/critic/comparison_<UTC timestamp>/` 包含：

- `critic_comparison.csv` 和 `critic_comparison.md`
- 两组 Accuracy、Macro F1、各类 P/R/F1、Sufficient AUROC、False Stop Rate
- 两组 confusion matrix CSV/PNG
- 两组完整逐样本概率与预测
- 每组优先按 False Stop 排序的 20 个 bad cases

False Stop Rate 固定定义为：

```text
预测为 Sufficient 且真实标签非 Sufficient 的样本数
--------------------------------------------------
所有真实标签非 Sufficient 的样本数
```

### 7. 一次运行两组 Critic

服务器也可直接执行：

```bash
bash run_phase1_critics.sh
```

该脚本使用固定的 seed-42 run 目录，因此故意不支持覆盖已有 checkpoint。需要重跑时应使用新的 run 目录名称，以保留历史实验。如果三份 sufficiency JSONL 不存在，该脚本会先用 `--jsonl-only` 自动构建，并保留 Git 中已有的报告与 manifest。

## Legacy Pipeline

旧版 Dense/Hybrid/Rerank 入口仍为 `scripts/retrieve.py` 和 `run_all.sh`。新增研究脚本读取旧 ranking，但不改变旧数据格式和旧结果文件。

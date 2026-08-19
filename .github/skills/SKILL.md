---
name: rag-paper-project
description: 用于理解和维护 RAG 检索预算自适应小论文实验项目（NQ 单跳 + HotpotQA 多跳，Retrieval-Only 阶段）。包含技术栈、环境、数据结构、脚本流水线、冻结的实验协议和当前进展。
---

# RAG_paper 项目理解指南

## 项目概述

小论文研究「不同复杂度的查询是否需要不同的检索预算/检索策略」，目标是查询复杂度自适应的 RAG（adaptive retrieval budget）。
当前阶段为 **Retrieval-Only**（不用 LLM 生成），验证研究假设：`oracle 自适应路由在接近固定最优 Recall 的前提下，显著降低平均检索成本`。
当前正在做 Phase 2（LLM Difficulty Router 实验），后续进入 LLM 生成阶段（Qwen2.5-7B 生成 EM/F1）。

## 环境

- 系统 Python 3.12（`/usr/bin/python3.12`）：datasets 5.0.0 / sentence-transformers / faiss / rank_bm25 已装
- 硬件：2× RTX 4090（GPU1 跑 LLM router / 生成，GPU0 跑检索）
- **离线运行**：`HF_HUB_OFFLINE=1`（模型已缓存）；下载数据必须 `HF_ENDPOINT=https://hf-mirror.com`
- 后台任务写法：`(setsid bash -c '...' < /dev/null &)`，输出重定向到 logs_*.txt
- 独立 git 仓库：remote = `git@github.com:s2079481035/RAG.git`，branch `master`
  ⚠️ 外层 `/home/sunjb` 是 drfft 仓库，**勿混提交**

## 数据（data/）

| 文件 | 说明 |
|---|---|
| `nq.json`（56MB） | NQ 单跳：10k 文档 KB + 1k test；文档截断至 1024 token；gold = 问题所在文档（每问 1 个） |
| `hotpotqa.json`（6.6MB） | HotpotQA 多跳：14.8k 文档 KB + 1k test；gold = supporting facts 文档（每问 2-4 个）；含 `type`（bridge/comparison/intersection）、`n_sent` 字段 |
| `indices/`（140MB，gitignore，可重建） | bge-large-en-v1.5 FAISS(IndexFlatIP) + BM25 pkl 索引 |

注意：HotpotQA 的 `context`/`supporting_facts` 是 dict 结构 `{title:[...], sentences:[...]}`。

## 核心脚本（scripts/）

| 脚本 | 作用 | 运行方式 |
|---|---|---|
| `download_data.py` | 下载 NQ + HotpotQA | `HF_ENDPOINT=https://hf-mirror.com python3.12 scripts/download_data.py` |
| `build_index.py` | 建 FAISS + BM25 索引 | `HF_HUB_OFFLINE=1 python3.12 scripts/build_index.py [--dataset nq\|hotpotqa\|all]` |
| `retrieve.py` | 统一检索：dense / hybrid / rerank | `--dataset nq --phase dense --ks 3,5,10,20`；rerank 额外支持 `--reranker base\|v2m3 --max-len` |
| `eval_recall.py` | 评估 Recall@K | 直接运行，输出 `results/recall_matrix.md` + `.json` |
| `stratify.py` | 分层分析（type / n_sent / difficulty / oracle） | 直接运行，输出 `results/stratify.md` + `.json` |
| `router_experiment.py` | Phase 2 Router 对比实验 | `--stage llm`（GPU1，写 `results/router_llm_preds.json`）\| `--stage eval`（CPU） |

完整流水线：`./run_all.sh`（Phase 1-3 dense/hybrid/rerank + 评估）。

## 关键设计约定

- 检索阶段：RRF(k=60) 融合 dense+BM25；rerank 基于 hybrid top-20 候选池，用 `bge-reranker-v2-m3`（v2-m3 替换 base 后 NQ rerank 不再伤害召回）
- 指标口径：NQ 用 Recall@K；HotpotQA 主口径 Hit@K、辅口径 Recall@K
- 结果文件名：`results/{dataset}_{method}_rerank_{variant}.json`，eval 优先读 v2m3
- **实验协议已冻结**（`EXPERIMENT_PROTOCOL.md`，2026-08-18）：difficulty 定义（retrieval-based、oracle-grounded：Easy=所有 gold 排名≤3 可达，Medium≤10，Hard 否则）、8 级策略阶梯（dense@3…rerank@20）、Cost(q)=最终文档数 K。协议冻结后不得改动定义
- Router 只能输入 query 文本，永不见 gold；difficulty 标签只用于校准 oracle 和评估 Router 准确率

## 关键结果（results/）

- **Recall 矩阵**（v2-m3）：NQ rerank@3 0.860 / @10 0.951；HotpotQA rerank@3 0.891 / @10 0.948（rerank 对两数据集均有益）
- **分层发现**：rerank 收益集中在 HotpotQA bridge 型问题（+7.1pp）；Hard 组（NQ 2.7% / HP 8.1%）所有策略 Recall 都差，是 coverage 瓶颈
- **Oracle Adaptive**：NQ 平均成本级 1.30、Recall 0.974；HP 1.51 / 0.960 —— 效率价值被证明，上限由 top-20 候选池决定
- **Router 现状（router_eval.md）**：rule 在 NQ 上 Acc 0.927 但 HP 上仅 0.235（把大部分问题判为 medium/hard）；LLM router 预测全部 easy（退化，无区分度）。→ Next Move 是用 query 特征做在线预测的轻量路由器

## 常见开发任务

- **重跑全检索流程**：`bash run_all.sh`
- **只跑 rerank**：`python3.12 scripts/retrieve.py --dataset nq --phase rerank --ks 3,5,10,20 --reranker v2m3`
- **分层分析**：`python3.12 scripts/stratify.py`
- **跑 Router 实验**：先 GPU1 `--stage llm`，再 CPU `--stage eval`
- **重建索引**：`HF_HUB_OFFLINE=1 python3.12 scripts/build_index.py --dataset all`
- **后台跑长任务**：`(setsid bash -c 'HF_HUB_OFFLINE=1 python3.12 scripts/retrieve.py ... > logs_rerank.txt 2>&1' < /dev/null &)`
- **学习入口**：先读 `SESSION_CONTEXT.md`（最新进展 + Next Move），再读 `EXPERIMENT_PROTOCOL.md`（冻结协议），对照 `results/*.md` 看结果

# Session Context — RAG Paper (Retrieval-Budget Pilot)

> 本文件是 RAG_paper 项目的独立 session 上下文。新 session 从 `Next Move` 开始继续。

## Objective
小论文：以 NQ（单跳）+ HotpotQA（多跳）研究「不同复杂度的查询是否需要不同的检索预算/检索策略」，最终做查询复杂度自适应的 RAG（adaptive retrieval budget）。当前阶段只做 Retrieval-Only（不用 LLM），验证研究假设。

## Git / 环境
- 工作目录：`/home/sunjb/RAG/RAG_paper`（独立 git 仓库，remote = `git@github.com:s2079481035/RAG.git`，branch `master`）
- 外层 `/home/sunjb` 是 drfft 仓库，勿混提交
- 系统 Python 3.12：datasets 5.0.0 / sentence-transformers / faiss / rank_bm25 齐备
- 2× RTX 4090；`HF_HUB_OFFLINE=1` 离线跑；下载必须 `HF_ENDPOINT=https://hf-mirror.com`
- 后台任务：`(setsid bash -c '...' < /dev/null &)`

## 数据（重新下载，已入库）
- `data/nq.json`（56MB，已提交）：10k 文档 KB + 1k test；文档截断至 1024 token；gold = 问题所在文档（每问 1 个）
- `data/hotpotqa.json`（6.6MB，已提交）：14.8k 文档 KB（全部 gold 文档 + 填充）+ 1k test；gold = supporting facts 文档（每问 2-4 个）
- ⚠️ **HotpotQA 下载脚本未保留 `type` 字段（bridge/comparison/intersection）——复杂度细分分析需要重新生成**
- `data/indices/`（140MB，gitignore，可重建）：bge-large-en-v1.5 FAISS + BM25 pkl

## 已完成
- `scripts/download_data.py`：NQ + HotpotQA 下载（HotpotQA 的 supporting_facts/context 是 dict 结构 `{title:[...], sentences:[...]}`）
- `scripts/build_index.py`：FAISS(IndexFlatIP) + BM25 索引
- `scripts/retrieve.py`：`--phase dense|hybrid|rerank --ks 3,5,10,20`；RRF(k=60) 融合，rerank 基于 hybrid top-20 → bge-reranker-base
- `scripts/eval_recall.py`：NQ Recall@K；HotpotQA Hit@K + Recall@K
- `results/recall_matrix.md` + 6 个结果 JSON（已入库）
- 首次推送：`git push -u origin master` 成功（commit: retrieval-budget pilot）

## 结果矩阵（Recall@K，已修复 RRF bug 后）
| 数据集 | 检索器 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| NQ | dense | 0.834 | 0.886 | 0.925 | 0.962 |
| NQ | hybrid | 0.851 | 0.905 | 0.938 | 0.970 |
| NQ | rerank | 0.772 | 0.843 | 0.909 | 0.970 |
| HotpotQA | dense | 0.833 | 0.881 | 0.912 | 0.934 |
| HotpotQA | hybrid | 0.846 | 0.893 | 0.929 | 0.956 |
| HotpotQA | rerank | 0.872 | 0.919 | 0.943 | 0.956 |

HotpotQA Hit@3 = 0.981（几乎全部问题 top-3 命中首个证据）。

## 关键发现
1. **Hybrid > Dense**：两数据集一致（+1~3pp）
2. **Reranker 收益方向相反**：NQ @3 0.772 < dense 0.834（-6pp，有害）；HotpotQA @3 0.872 > dense（+4pp，有益）——「不同复杂度问题需要不同检索策略」的初步证据
3. NQ/HotpotQA Recall@K 曲线斜率接近（@3→@10 增量 ~0.08-0.09），但 HotpotQA 覆盖全部证据需更大 K

## 已修 Bug
- Hybrid RRF dense 权重错位（scores 是排名序非位置序，须用 `idxs[0]`）——修复后 hybrid @3 0.49→0.85
- NQ 文档未截断导致 489MB（截断 1024 token → 56MB）
- HotpotQA context/supporting_facts 为 dict 结构

## Next Move
1. 复杂度细分分析：重新生成 HotpotQA 数据（保留 `type` 字段），NQ 用难度代理（问题-文档重叠度 / 答案在文档中的位置），按复杂度分组画 Recall@K 曲线，验证「rerank 简单问题有害、复杂问题有益」是否由复杂度驱动
2. 若现象成立 → 进入 LLM 阶段（Qwen2.5-7B 生成，需下载）+ adaptive budget 路由
3. 阶段性结果推送至 s2079481035/RAG
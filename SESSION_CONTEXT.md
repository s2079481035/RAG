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

## 结果矩阵（Recall@K，reranker 已升级为 bge-reranker-v2-m3）
| 数据集 | 检索器 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| NQ | dense | 0.834 | 0.886 | 0.925 | 0.962 |
| NQ | hybrid | 0.851 | 0.905 | 0.938 | 0.970 |
| NQ | rerank | 0.860 | 0.914 | 0.951 | 0.970 |
| HotpotQA | dense | 0.833 | 0.881 | 0.912 | 0.934 |
| HotpotQA | hybrid | 0.846 | 0.893 | 0.929 | 0.956 |
| HotpotQA | rerank | 0.891 | 0.931 | 0.948 | 0.956 |

HotpotQA Hit@3 = 0.981（几乎全部问题 top-3 命中首个证据）。

## 关键发现（分层分析，results/stratify.md）
1. **rerank 对两个数据集均有益**：v2-m3 替换 base 后，NQ rerank@3 0.860 > hybrid 0.851、@10 0.951 > 0.938；HotpotQA @3 0.891 > 0.846。原「rerank 伤害 NQ」是 bge-reranker-base（280M，弱）的假象——base 曾把 160 个 hybrid@1 的 gold 降级。
2. **rerank 收益集中在 bridge 型多跳问题**（HotpotQA）：bridge 上 rerank@3 0.871 vs dense 0.800（+7.1pp）；comparison 本身近饱和（dense@3 0.973，rerank 0.976），且 hybrid 对 comparison 有害（0.944 < dense 0.973）。
3. **按难度分层（retrieval-based）**：Easy(93.4% NQ / 84.8% HP) rerank 仍稳增；Medium 上 rerank 大胜（NQ @5 0.462 vs dense 0.308；HP @10 0.915 vs 0.796）；Hard 组（NQ 2.7% / HP 8.1%）所有策略 Recall 都差（HP @5 仅 ~0.49），是真正的 coverage 瓶颈，且 NQ hard 上 dense@20 (0.370) > hybrid@20 (0.259)——RRF 融合会压掉词面不相似的 gold。
4. **Oracle Adaptive**：若按「覆盖全部 gold 的最便宜策略」路由，NQ 平均成本级 1.30（vs 固定 dense@3 成本 1 但 Recall 0.834；rerank@20 成本 7 才到 0.970），oracle 达 0.974；HotpotQA oracle 成本 1.51 达 0.960。→ 覆盖率几乎打平固定最优，但预算平均省 ~4-5 个成本级。这证明 adaptive budget 的「效率」价值；覆盖率上限由候选池（top-20）决定。
5. 复杂度分组：HotpotQA bridge 813 / comparison 187；n_sent 2(703) / 3+(297)，n_sent=3+ 时 rerank@3 0.914 > 2 的 0.881。

## 已修 Bug
- Hybrid RRF dense 权重错位（scores 是排名序非位置序，须用 `idxs[0]`）——修复后 hybrid @3 0.49→0.85
- NQ 文档未截断导致 489MB（截断 1024 token → 56MB）
- HotpotQA context/supporting_facts 为 dict 结构
- HotpotQA 下载脚本未保留 type 字段 → 已重建 data/hotpotqa.json（kb 与旧版逐字节一致，索引复用），新增 type/n_supporting/n_sent
- retrieve.py 支持 `--reranker base|v2m3`、`--max-len`；rerank 结果存 `*_rerank_{variant}.json`
- eval_recall.py 优先读取 v2m3 结果

## Next Move
1. LLM 阶段（Qwen2.5-7B 生成）：以 rerank@10 为主检索 → 生成 EM/F1，验证 bridge/medium 组答案质量提升
2. Adaptive router：用 query 特征（type 标签不可用，需在线预测）做规则/轻量分类器，目标 = oracle 的预算分配（≈ rerank@3~5 主用 + hard 加大 pool）；难点：hard 组召回天花板（需扩大候选池或混合深度检索）
3. 若需要：2WikiMultihopQA 扩展实验 + RAGAS 指标
4. 阶段性结果推送至 s2079481035/RAG（本 session 已推一版）

---

# Phase 2 — LLM Router 实验（已冻结协议 EXPERIMENT_PROTOCOL.md）

## 冻结的策略映射（由 Oracle 分布确定）
easy→dense@3（成本1）、medium→dense@5（成本2）、hard→hybrid@20（成本6）；成本级 1-8（dense@10=3 入表）。

## 结果（results/router_eval.md）
| 系统 | NQ R@10 | HP R@10 | NQ成本 | HP成本 | NQ Acc | HP Acc |
|---|---|---|---|---|---|---|
| dense@10 | 0.925 | 0.912 | 10.0 | 10.0 | - | - |
| hybrid@10 | 0.938 | 0.929 | 10.0 | 10.0 | - | - |
| rerank@10 | 0.951 | 0.948 | 10.0 | 10.0 | - | - |
| rerank@20 | 0.951 | 0.948 | 20.0 | 20.0 | - | - |
| rule | 0.834 | 0.886 | 3.03 | 9.55 | 0.927 | 0.235 |
| llm | 0.844 | 0.882 | 3.52 | 7.73 | 0.806 | 0.310 |
| oracle | 0.973 | 0.959 | 3.71 | 4.17 | - | - |

结论：**LLM Router 目前不能接近 Oracle Recall**。NQ：成本与 oracle 相当（3.52 vs 3.71）但 Recall 掉 ~13pp（@10 0.844 vs 0.973）；HotpotQA：Recall 与固定 rerank@10 差 ~6.6pp（0.882 vs 0.948），成本还高（7.73）。根因是**难度判断与 retrieval-based oracle 定义不一致**：LLM 对 HotpotQA 过度判 medium/hard（50.4% vs oracle 7.1%），对 NQ 误把真正 medium/hard 判 easy。按协议不迭代调 prompt，如实记录。
- LLM 曾因未加 chat template 导致 43% 输出不可解析（已修复：`apply_chat_template`，unparsed=0）
- 运行细节：Qwen2.5-7B-Instruct fp16（14.2GB）device_map=auto 于 GPU1，batch 1（GPU1 与 yangcc 共享，OOM 过 3 次：4bit+acc1.14 不兼容 → fp16；chunk 切片 bug → 改 1）
- 原始输出在 results/router_llm_raw.json
# Phase 2b — Oracle Early Stopping（决策门实验，已冻结协议第 9 节）

## 结果（results/early_stop.md + latency.md）
| 系统 | HP Recall@final | HP FullCov | HP 平均文档成本 | HP 平均延迟(ms) |
|---|---|---|---|---|
| rerank@10 (fixed) | 0.948 | 0.897 | 10.0 | 220.9 |
| rerank@20 (fixed) | 0.956 | 0.914 | 20.0 | 220.9 |
| oracleES-A (dense@3→hybrid@5→hybrid@10→rerank@10→rerank@20) | **0.959** | 0.919 | **5.39** | **66.9** |
| oracleES-B (dense@10→hybrid@10→rerank@10→rerank@20) | 0.962 | 0.926 | 10.82 | 52.3 |

NQ: ES-A 0.972@4.07docs；ES-B 0.980@10.27docs（fixed rerank@20 = 0.970@20docs）。
HotpotQA ES-A 停止分布：S1 68.4% / S2 14.6% / S3 4.6% / S4 3.3% / S5 9.1%（S5 ≈ 不可充分满足的 hard 查询 81 个）。
False Early Stop（oracle 口径）= 0（按构造）。

## 决策门结论 ✅
**Oracle Early Stopping 明显优于固定策略：Recall ≥ fixed（0.959 vs 0.956）且成本 3.7x 更低（5.39 vs 20 docs）、延迟 3.3x 更低。方向成立 → 值得训练 Retrieval Sufficiency Critic。**

## 下一阶段（Critic，未开始，等待确认）
1. 训练集构造：用 train 划分的 HotpotQA，gold-grounded 充分性标签（当前 top-K 是否覆盖全部 supporting facts）。⚠️ 严格 train/validation/test 划分，test 的 gold 只用于评估 oracle/标签验证，不得进入训练
2. Critic 输入：query + 当前 retrieved docs（text），输出 Sufficient/Insufficient；轻量（DeBERTa/roberta-base 或复用 embedding）
3. 评估：在 ES ladder 上接 Critic 早停 → Recall/Cost/Latency/False Early Stop（Critic 判 sufficient 但 gold 未覆盖）
4. LogicRAG 的 reasoning depth / subquestion count 仅保留为后续 error analysis 变量，不做主路由器
5. 不做 RAFT/DAG Cache/Web Search/HyDE/Query Decomposition

# Phase 3 — Learned Retrieval Sufficiency Critic（已完成）

## 结果（results/critic_eval.md）
- Critic（bge-reranker-base 二分类，train 5405q×3阶段，val 选优 F1=0.9386@ep2）
- 离线 F1：dense@3 0.887 / hybrid@5 0.923 / rerank@20 0.958
- E2E：Fixed rerank@20 (0.956, 20.0docs, 220.9ms) vs Oracle ES-3 (0.959, 6.18, 72.8) vs **Critic (0.915, 6.46, 82.9, FES 9.6%)**
- 结论：Critic 成本/延迟逼近 Oracle（6.46 vs 6.18 docs），但 False Early Stop 9.6%（96 查询，误停点平均覆盖 0.495）导致 Recall 掉 4.3pp
- 待探索：提高阈值牺牲少量成本换 Recall；或 stage 3 前强制再检索

## 关键经验
- 数据集离线模式（HF_HUB_OFFLINE=1）下 datasets 库无法解析数据集，须 HF_ENDPOINT=https://hf-mirror.com
- bge-reranker-base checkpoint 是 num_labels=1，加载 2 类头须先加载再替换 classifier.out_proj（ignore_mismatched_sizes 对该模型无效）
- test 1000 与已有实验完全一致（KB 逐字节相同、shuffle 需复刻 extra 消耗的 RNG 状态）

# Phase 3b — Validation-set 阈值策略优化（已完成）

## 方法
- 只在 val(1000q) 上 sweep 与选择；test 仅报告选定配置一次（无 test 泄漏）
- 全局 t=0.30~0.95 (step .05) + 分阶段 grid (d∈{.55...95}, h∈{.35...75})

## 结果（results/critic_policy.md, 曲线 recall_cost_curve.png）
- val 约束 Recall≥0.94 下最小成本配置：**d0.95/h0.65**（val R=0.941, cost 7.01）
- test @d0.95/h0.65: R=0.929, FullCov=0.861, cost 7.23, lat 96.4ms, FES 6.4%（vs t=0.5: R=0.915/FES 9.6%）
- **val→test 有 ~1.2pp 泛化差**；阈值调优只能部分修复 FES

## FES 诊断（results/fes_queries_test_t05.json，96 条 test@t=0.5 误停查询）
- 停止阶段：dense=70 / hybrid=26；99% 停止时只覆盖 1/2 gold docs
- 停止点 critic 概率 mean=0.845、max=0.999 → 大部分是"高置信误判"，非低置信边界样本
- 仅 18/96 在 t=0.8 下会继续升级 → 单纯提阈值上限有限
- 这些查询 rerank@20 覆盖率均值 0.938 → 升级后大多可恢复

# 实验协议 — LLM Difficulty Router 验证（Phase 2）

> 本协议在实现 Router 之前冻结。所有定义、映射、规则、Prompt 一经确定不再调整，
> 防止「为了让 Router 看起来有效而悄悄修改 difficulty 定义」。

## 1. 目标（单一且明确）

验证假设：**LLM Difficulty Router 能在接近 Oracle Recall 的前提下，显著降低平均 Retrieval Cost。**

- 不是追求最高 Recall
- 不引入 HyDE / Query Decomposition / 其他检索模块
- 对照：Fixed / Rule-based / LLM Router / Oracle 四种策略

## 2. 冻结的 Difficulty 定义（retrieval-based，oracle-grounded）

对每个 query，用其 gold 文档（测试集 gold 信息）计算，算法与 `scripts/stratify.py` 完全一致：

```
best(m, g)   = gold 文档 g 在方法 m (dense/hybrid/rerank) 的 ranked 列表中的最小排名（1-based），
               不在 top-20 则视为不可达
easy   = 每个 gold 文档 g 都存在某方法使 best(m, g) <= 3
medium = 非 easy，且每个 gold 文档 g 都存在某方法使 best(m, g) <= 10
hard   = 否则（至少一个 gold 文档无法进入任何方法的 top-10）
```

NQ 与 HotpotQA 同一套定义（NQ 每问 1 个 gold，HotpotQA 2 个）。

**用途约束**：该标签只用于 (a) 训练/校准 Oracle，(b) 评估 Router 的 accuracy。Router 本身只输入 query 文本，永不见 gold。

## 3. 冻结的策略阶梯与 Cost 定义

策略阶梯（成本递增，候选池为 hybrid top-20，rerank 在池内重排）：

| # | 策略 | 最终列表 |
|---|---|---|
| 1 | dense@3 | dense top-3 |
| 2 | dense@5 | dense top-5 |
| 3 | dense@10 | dense top-10 |
| 4 | hybrid@5 | RRF top-5 |
| 5 | hybrid@10 | RRF top-10 |
| 6 | rerank@10 | rerank(v2-m3) top-10 |
| 7 | hybrid@20 | RRF top-20 |
| 8 | rerank@20 | rerank(v2-m3) top-20 |

**Cost(q) = 该策略的最终文档数 K**（即送入生成器的上下文文档数，是检索+生成阶段的主导成本代理）。
额外记录 cost level（1–7）便于比较。

## 4. 冻结的 difficulty → 策略映射（由 Oracle 分布确定，已计算）

Oracle 定义为：对每个 query 取「覆盖全部 gold 的最便宜策略」，无覆盖则取覆盖率最高（并列取最便宜）。
各难度组内 Oracle 的选择分布（2026-08-18 计算，数据已入 git）：

| 组 | NQ Oracle 选择 | HotpotQA Oracle 选择 | 冻结映射 |
|---|---|---|---|
| easy | dense@3 占 89% | dense@3 占 81% | **dense@3**（成本 1） |
| medium | dense@5 12 / hybrid@10 10 / rerank@10 8（分散） | dense@5 30 / rerank@10 16 / hybrid@10 15 | **dense@5**（成本 2，众数） |
| hard | 20 例不可检索（tie-break 兜底 dense@3，覆盖率恒 0）+ hybrid@20 7 例 | 65 例不可检索 + hybrid@20 8 例 | **hybrid@20**（成本 6，唯一有覆盖率贡献的选择） |

说明：hard 组 Oracle 的 dense@3 票是「任何策略覆盖率为 0 时按成本并列取最便宜」的产物，对 Recall 无贡献；
hard 组唯一有意义的策略是深池 hybrid@20。故冻结映射 hard → hybrid@20。

**Oracle 上界**（用于对照）：逐 query 最优选择，NQ 覆盖率 0.974 / 平均成本 1.30；HotpotQA 0.960 / 1.51。

## 5. 冻结的评估指标

对每种系统（Fixed 4 个固定点 + Rule-based + LLM Router + Oracle）严格记录：

1. **Recall@K（K=3/5/10/20）**：系统最终列表 top-K 与 gold 的交集比例。
   若列表长度 < K，则取整个列表（不做 padding 幻觉）——低成本路由在 @10/@20 上被如实惩罚。
2. **平均 Retrieval Cost** = mean(Cost(q))，附 cost level 均值。
3. **Router Accuracy** = 路由预测（easy/medium/hard）与 oracle 标签的一致率（rule / LLM 两种都报告）。
4. **Difficulty 分布**：预测分布 vs oracle 分布。

## 6. 冻结的 Rule-based Router（启发式，先验设定，不调参）

```
tokens = 小写分词（去标点）后的词数
特征：
  - 词数 >= 18                        -> hard
  - 包含 {compare, comparison, both, differ, similar, versus, vs., unlike}  -> medium
  - 词数 >= 10 且包含 {and, or, which, whom} 且大写实体 >= 2  -> medium
  - 否则                              -> easy
```
（规则为 a-priori 启发式；accuracy 如实报告，允许表现差，不允许事后调参。）

## 7. 冻结的 LLM Router

- 模型：Qwen2.5-7B-Instruct（本地，bitsandbytes 4-bit），`CUDA_VISIBLE_DEVICES=1`
- 输入：仅 query 文本。输出：仅 `easy` / `medium` / `hard` 之一
- temperature=0，max_new_tokens=8，解析首行输出；无法解析则回退 `easy` 并计数
- Prompt（冻结）：

```
Classify the following question by the difficulty of retrieving its answer
documents from a large corpus.
- easy: the answer documents are found within the top-3 results of a standard
  dense retriever.
- medium: they are not in top-3, but within top-10 of some stronger retriever.
- hard: the answer documents cannot be found even in top-10.
Answer with exactly one word: easy, medium, or hard.

Question: {query}
```

## 8. 防偏保障

- 难度标签仅由 gold-grounded oracle 产生；Router 只见 query
- 映射表在 Router 运行前冻结（本文档）
- 本阶段不迭代调 Router；如 LLM 表现差，如实记录，进入下一阶段再分析原因
- 所有结果输出 `results/router_eval.md` + `results/router_eval.json`，入 git
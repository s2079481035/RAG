# 候选 Claim—Evidence 矩阵

## 判定规则

- `SUPPORTED`：存在冻结实验直接支持，且措辞不超出数据集、协议和统计范围。
- `SUPPORTED WITH QUALIFICATION`：方向有证据，但必须同时写出适用范围或反例。
- `UNSUPPORTED`：现有证据不足、与冻结结果冲突，或比较协议不允许。

## C1. Evidence-aware critic 优于 query/stage-based routing

**状态：SUPPORTED**

- **Supporting experiments**：Phase 2 matched Controller baselines；Phase 3A 三 seed Test；seed-42 paired question bootstrap；2Wiki Train-derived Dev 三 seed；2Wiki frozen heldout classification。
- **Dataset**：HotpotQA、2WikiMultiHopQA。
- **Exact metrics**：Hotpot 三 seed score-aware vs Query+Stage：Macro F1 `0.7970763875423774` vs `0.5773192401362102`，AUROC `0.9246307529424743` vs `0.8207468483339321`，FSR `0.4514991181657848` vs `0.8465608465608465`。2Wiki Dev sequential Macro F1 `0.9872524228858328` vs `0.9299401096178403`，FSR `0.004651147133451258` vs `0.028131942983359897`。
- **Statistical evidence**：Hotpot seed-42 paired bootstrap：Macro F1 delta `+0.18596545414551247`，95% CI `[0.14818097525341, 0.2198967643819512]`；AUROC delta CI `[0.0730343680380726, 0.12981707553065983]`；FSR delta CI `[-0.3874353007860861, -0.2541887866909472]`。另有三 seed 均值/标准差。
- **Limitations**：只支持本研究 backbone、数据构造和固定 ladder；不能外推为所有 query router 或官方 Adaptive-RAG 均较差。

## C2. Score-aware packing 是测试表示中最强的稳定方案

**状态：SUPPORTED WITH QUALIFICATION**

- **Supporting experiments**：Phase 2 packing comparison；Phase 3A 三 seed Query+Evidence concat vs score-aware。
- **Dataset**：HotpotQA。
- **Exact metrics**：三 seed score-aware vs concat：Macro F1 `0.7970763875423774` vs `0.7523233785363042`，AUROC `0.9246307529424743` vs `0.9081029196716967`，FSR `0.4514991181657848` vs `0.5361552028218695`。
- **Statistical evidence**：三 seed sample std 分别为 Macro F1 `0.013309348361898274` vs `0.016917681271698157`；未保存 score-aware−concat 的 paired bootstrap CI。
- **Limitations**：只能称“在测试表示中最好/更稳定”，不能声称理论最优或对任意长度分布最优。

## C3. False Stop 主要集中于 Hard Partial evidence

**状态：SUPPORTED WITH QUALIFICATION**

- **Supporting experiments**：Phase 3A hard-partial bucket analysis。
- **Dataset**：HotpotQA Test。
- **Exact metrics**：Continue decisions 为 Easy `10`、Medium `45`、Hard `323`；FinalController 的 Hard false stops 为 `137/159`，Hard FSR `0.4241486068111455`，Medium FSR `0.4666666666666667`。
- **Statistical evidence**：按 question 的 bootstrap 未针对 bucket prevalence 单独提供 CI。
- **Limitations**：只支持“按绝对数量集中，且由 Hard Partial 高占比驱动”；不支持“Hard Partial 条件错误率最高”。

## C4. Coverage Auxiliary 能稳定改善 Hard Partial discrimination

**状态：UNSUPPORTED**

- **Supporting experiments**：Phase 3A auxiliary ablation。
- **Dataset**：HotpotQA Test，3 seeds。
- **Exact metrics**：baseline vs auxiliary：Macro F1 `0.7970763875423774` vs `0.7964713902302248`；AUROC `0.9246307529424743` vs `0.9146921248370523`；FSR 均为 `0.4514991181657848`；Hard Partial FSR 均为 `0.45717234262125905`。
- **Statistical evidence**：seed-42 fixed-model bootstrap 对 AUROC/FSR 有利，但 Macro F1 CI `[-0.002543580220234376, 0.04500820403965164]` 包含 0；三 seed 结果不复现稳定改善。
- **Limitations**：可以写 coverage signal 可学习（MAE/Spearman），不能写分类稳定获益。

## C5. Hard-partial-aware sampling 降低 False Stop

**状态：UNSUPPORTED**

- **Supporting experiments**：Phase 3A sampling ablation。
- **Dataset**：HotpotQA Test，3 seeds。
- **Exact metrics**：natural vs hard-partial-aware：FSR `0.4514991181657848` vs `0.45943562610229277`；Hard Partial FSR `0.45717234262125905` vs `0.4726522187822497`；AUROC `0.9246307529424743` vs `0.9057730544826909`。
- **Statistical evidence**：三 seed mean/std；无支持改善的 paired CI。
- **Limitations**：该失败不等于所有重采样/代价敏感方法无效。

## C6. Risk10 在 HotpotQA 上保持 QA 质量并降低成本

**状态：SUPPORTED WITH QUALIFICATION**

- **Supporting experiments**：Phase 3B frozen Test end-to-end evaluation。
- **Dataset**：HotpotQA Test，1,000 questions。
- **Exact metrics**：Risk10 vs FixedHeavy F1 `0.5104395222321538` vs `0.510839232130099`；chunks `9.38` vs `20.646`；reranker `0.207` vs `1.0`；latency `347.66887517040595` vs `587.7891366463155 ms`；FSR `0.08445945945945946`。
- **Statistical evidence**：F1 delta CI `[-0.019256642605760238, 0.018541145847616475]`，支持预注册 -0.02 margin 下非劣；chunks 与 latency delta CI 均不含 0。
- **Limitations**：FSR CI `[0.05638965371174415, 0.11765174222555261]` 未完全低于 10%；Test 在 Phase 3A 已被检查，不能描述为新的独立 confirmatory set。

## C7. Risk10 在跨数据集上提供 10% 风险保证

**状态：UNSUPPORTED**

- **Supporting experiments**：Phase 3B HotpotQA Test；Phase 4 2Wiki frozen heldout。
- **Dataset**：HotpotQA、2Wiki。
- **Exact metrics**：Hotpot observed FSR `0.08445945945945946`；2Wiki observed FSR `0.12310521976279572`。
- **Statistical evidence**：2Wiki 95% CI `[0.11810921353389348, 0.12837358336460408]` 完全高于 10%。
- **Limitations**：只能写“跨数据集出现 risk generalization gap”；不能写 distribution-free 或 guaranteed control。

## C8. Risk10 在 2Wiki 上减少成本且不损害 QA 质量

**状态：SUPPORTED WITH QUALIFICATION**

- **Supporting experiments**：Phase 4 one-time frozen heldout、5,000 次 question bootstrap。
- **Dataset**：2Wiki heldout，12,576 questions。
- **Exact metrics**：Risk10 vs FixedHeavy Answer F1 `0.20256212239279647` vs `0.19215672075640125`；chunks `13.268686386768447` vs `20.293256997455472`；reranker `0.5010337150127226` vs `1.0`；latency `1124.3924006067339` vs `1676.3052861272201 ms`。
- **Statistical evidence**：F1 delta `+0.010405401636395223`，CI `[0.005712640327771366, 0.015261824576448622]`；chunks、reranker、latency delta CI 均不含 0。
- **Limitations**：同时必须报告 FSR `0.12310521976279572` 未达目标，以及 complete coverage 从 `0.39782124681933845` 降至 `0.35520038167938933`。

## C9. Temperature Scaling 是性能提升的关键原因

**状态：UNSUPPORTED**

- **Supporting experiments**：Phase 3B calibration metrics 与 raw/temperature operating points。
- **Dataset**：HotpotQA Dev/Test；2Wiki Train-derived Dev。
- **Exact metrics**：Hotpot dev_policy raw NLL/ECE `0.16135968177160073/0.02012202241830531`，temperature `0.16019405647318355/0.02103199002872667`，未同时改善；Hotpot raw conservative Risk10 与 temperature conservative Risk10 的 Test FSR/F1 数值相同到报告精度。2Wiki Dev calibration 则改善 NLL/Brier/ECE。
- **Statistical evidence**：无温度缩放独立效应的 paired causal comparison。
- **Limitations**：可描述为冻结 calibration protocol 的组成，不可宣称普遍提升 discrimination 或部署收益。

## C10. 简单 Score Threshold 足以进行安全停止

**状态：UNSUPPORTED**

- **Supporting experiments**：Phase 4 score-threshold Dev sweep 与 heldout baseline。
- **Dataset**：2Wiki。
- **Exact metrics**：ScoreThresholdRisk10 heldout FSR `0.0`、UER `1.0`、average final stage `3.0`、chunks `20.293256997455472`。
- **Statistical evidence**：冻结 Dev selection 只能选 always-final sentinel；没有非退化工作点。
- **Limitations**：不代表其他 score calibration/learned score methods 不可能有效。

## C11. 轻量 critic 在本协议下优于 LLM Judge 的风险—成本折中

**状态：SUPPORTED WITH QUALIFICATION**

- **Supporting experiments**：Phase 4 LLM Judge Dev gate 与 frozen heldout。
- **Dataset**：2Wiki。
- **Exact metrics**：heldout EvidenceAwareClassification FSR/UER/latency `0.017046833292890074 / 0.06103589082312486 / 1241.5772767273606 ms`；Judge-512 为 `0.03152073732718894 / 0.7382474080386309 / 1788.9903085523201 ms`；Judge-Full 为 `0.019734832284474858 / 0.767405951712521 / 1890.598273004002 ms`。
- **Statistical evidence**：无 Judge vs critic paired bootstrap；结论为冻结工作点的描述性比较。
- **Limitations**：Judge 是 hard-label baseline，不能比较 AUROC；使用相同 generator，不是独立 oracle；不能外推到其他 LLM/prompt。

## C12. Evidence-conditioned stopping 优于本研究的 Adaptive-RAG-style Router

**状态：SUPPORTED WITH QUALIFICATION**

- **Supporting experiments**：Phase 4 Router 三 seed heldout；seed42 paired bootstrap。
- **Dataset**：2Wiki heldout。
- **Exact metrics**：Router mean F1/chunks `0.19576064593012324 / 15.487171331636981`；Risk10 `0.20256212239279647 / 13.268686386768447`。Router Medium route rate 为 `0.0`。
- **Statistical evidence**：seed42 Router−Risk10 F1 delta CI `[-0.00919986490927534, -0.004277691741995629]`；chunks delta CI `[2.0531130725190825, 2.2430065203562353]`。
- **Limitations**：只适用于 adapted query-only Router；禁止写“优于官方 Adaptive-RAG”。

## C13. 2Wiki 的主要瓶颈之一是 retrieval ceiling

**状态：SUPPORTED WITH QUALIFICATION**

- **Supporting experiments**：Phase 4 heldout retrieval table；Train-derived stage rescue analysis。
- **Dataset**：2Wiki。
- **Exact metrics**：heldout final coverage `0.39782124681933845`、TRFR `0.6021787531806615`；heldout Dense→Final rescue `0.11757166161733862`，Hybrid→Final rescue `0.05159674389480275`。Train-derived Dev 对应 rates 为 `0.06489675516224189` 与 `0.02386451116243264`。
- **Statistical evidence**：heldout n=12,576 的描述性比例；rescue rate 未提供独立 CI，但 Train-derived Dev 方向一致。
- **Limitations**：ceiling 属于冻结 corpus/retriever/ladder，不是 2Wiki 固有不可回答率，也不排除 query reformulation 或更强 retriever 的改善。

## C14. Automatic supporting-fact sufficiency label 等价于人类 answerability

**状态：UNSUPPORTED（强形式）**

- **Supporting experiments**：180-record stratified Dev manual audit。
- **Dataset**：HotpotQA Dev audit。
- **Exact metrics**：binary 与 three-class Accuracy/Macro F1 均为 `1.0`；disagreement `0`。
- **Statistical evidence**：无独立 reviewer、一致性系数或随机总体抽样 CI。
- **Limitations**：只可写“在该分层单 reviewer 审核中完全一致，支持其作为本协议 proxy”；不能写普遍等价或 population accuracy=100%。

## C15. Ours 数值优于 S2G-RAG / Stop-RAG / SURE-RAG

**状态：UNSUPPORTED**

- **Supporting experiments**：无协议一致的外部数值实验。
- **Dataset**：N/A。
- **Exact metrics**：S2G official-pipeline executed N=`0`；Stop-RAG experiment skipped；SURE-RAG method audit only。
- **Statistical evidence**：无。
- **Limitations**：只能做 method-level novelty boundary 与 reproducibility status，对不同 retriever/generator/corpus 禁止显著性或速度比较。

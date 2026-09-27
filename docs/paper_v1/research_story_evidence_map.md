# 研究叙事—证据映射

本文档只组织已冻结的 Phase 1–4 与 external baseline 证据，不新增实验，不改变指标。每个主题均按同一条证据链书写。

## 1. Fixed retrieval

**研究问题**  固定检索深度能否同时兼顾回答质量、证据完整性与计算成本？

**→ 对应实验现象**  HotpotQA 上 FixedHeavy 的 Answer F1 为 `0.510839232130099`、平均 chunks 为 `20.646`、总延迟为 `587.7891366463155 ms`；Risk10 的 F1 为 `0.5104395222321538`、chunks 为 `9.38`、延迟为 `347.66887517040595 ms`。2Wiki heldout 上 FixedHeavy 的 F1 为 `0.19215672075640125`、chunks 为 `20.293256997455472`、延迟为 `1676.3052861272201 ms`。

**→ 为什么现有简单方案不够**  固定浅检索成本低但容易缺证据；固定深检索避免提前停止，却对所有问题支付最高检索与重排成本，而且在 2Wiki 上仍有 `0.6021787531806615` 的 terminal retrieval failure rate。

**→ 我们采取的设计**  使用 Dense@5 → Hybrid@10 → Rerank@20 的冻结检索阶梯，仅在前两个 actionable stage 由 Controller 判断 Stop/Continue，最终阶段强制终止。

**→ 支持该设计的实验**  HotpotQA Phase 3B 固定策略与 Risk10 对比；2Wiki Phase 4 heldout 固定策略、Controller、Router 与 Judge 对比；question-level bootstrap。

**→ 可以支持的结论**  在已评估的冻结阶梯中，按 evidence state 自适应停止可以形成可测量的质量—成本边界，并显著减少平均检索量和延迟。

**→ 不能支持的结论**  不能声称固定检索普遍劣于任何自适应方法，也不能把不同语料、硬件或生成栈的延迟差解释为纯算法优势。

## 2. Query-only / Query+Stage

**研究问题**  仅凭问题或问题加阶段标识，能否可靠判断当前证据是否足够？

**→ 对应实验现象**  HotpotQA 三 seed Test 上 Query+Stage Macro F1 为 `0.5773192401362102 ± 0.035058973661818135`、AUROC 为 `0.8207468483339321 ± 0.008139434985352037`、FSR 为 `0.8465608465608465 ± 0.0642045560820719`；score-aware Query+Evidence 分别为 `0.7970763875423774 ± 0.013309348361898274`、`0.9246307529424743 ± 0.007180620389277922`、`0.4514991181657848 ± 0.04152119000703193`。

**→ 为什么现有简单方案不够**  Query-only/Stage 容易学习问题难度与阶段先验，但无法直接观察当前检索结果是否覆盖关键事实。类别不平衡还会使高 Stop F1 掩盖 Continue 样本的错误停止。

**→ 我们采取的设计**  将 question 与 model-visible packed evidence 联合输入 evidence-aware critic，并单独报告 Macro F1、AUROC、FSR、Hard Partial FSR 和 UER。

**→ 支持该设计的实验**  Phase 1 Query-only vs Query+Evidence；Phase 2 matched baselines；Phase 3A 三 seed 与 2,000 次 question-level paired bootstrap；2Wiki Train-derived Dev 三 seed gate。

**→ 可以支持的结论**  在匹配训练与评估协议下，使用当前 evidence 的 critic 明显优于只使用 query/stage shortcut。

**→ 不能支持的结论**  不能声称任何 query router 都无效，也不能把本研究的 Adaptive-RAG-style Router 当成官方 Adaptive-RAG 的完整复现。

## 3. Score Threshold

**研究问题**  直接对检索分数设阈值，能否替代学习式证据充分性判断？

**→ 对应实验现象**  2Wiki Dev 上所有风险目标最终只能选择 always-final sentinel；heldout 的 ScoreThresholdRisk10 等价于 FixedHeavy：FSR `0.0`、UER `1.0`、chunks `20.293256997455472`、Answer F1 `0.19215672075640125`。

**→ 为什么现有简单方案不够**  Dense 与 RRF 分数跨 stage 不同尺度，top-1 score 不直接表示多跳 supporting facts 是否完整；安全阈值会退化为总是继续。

**→ 我们采取的设计**  使用 question+evidence critic 学习覆盖状态，并把 score threshold 保留为冻结 baseline，而不是修改其阈值以追求更好结果。

**→ 支持该设计的实验**  Phase 4 score-threshold Dev sweep、冻结 selection manifest 与一次性 2Wiki heldout 结果。

**→ 可以支持的结论**  在本研究冻结的 2Wiki 检索分数与阈值协议下，简单 score threshold 未形成非退化的风险—效率工作点。

**→ 不能支持的结论**  不能声称所有检索分数校准或学习排序分数方法都必然失败。

## 4. LLM Judge

**研究问题**  通用生成式 LLM 对证据充分性的硬判断，能否替代轻量 critic？

**→ 对应实验现象**  2Wiki heldout 上 Judge-512 FSR `0.03152073732718894`，但 UER `0.7382474080386309`、总延迟 `1788.9903085523201 ms`；Judge-Full FSR `0.019734832284474858`，UER `0.767405951712521`、延迟 `1890.598273004002 ms`。EvidenceAwareClassification 的 FSR 为 `0.017046833292890074`、UER 为 `0.06103589082312486`、延迟为 `1241.5772767273606 ms`。

**→ 为什么现有简单方案不够**  硬标签 Judge 可以通过大量预测 Continue 降低 FSR，但会显著增加不必要升级、token 与推理延迟；硬标签也不提供可排序概率供同口径 AUROC 比较。

**→ 我们采取的设计**  使用轻量监督 critic 输出连续 stop probability；将 Judge-512 与 Judge-Full 作为冻结 hard-label baseline，并对 invalid output 采用安全 Continue。

**→ 支持该设计的实验**  2Wiki 2,000-question Dev gate、12,576-question frozen heldout、strict parse audit 与 token/latency 统计。

**→ 可以支持的结论**  在本研究实现与冻结阶梯下，LLM Judge 的低 FSR 伴随很高 UER 和额外计算，轻量 evidence-aware critic 提供更平衡的工作点。

**→ 不能支持的结论**  不能声称轻量 critic 普遍优于所有 LLM-as-a-Judge，也不能把同一 generator 产生答案和 Judge 标签视为独立人工 oracle。

## 5. Evidence-aware Critic

**研究问题**  当前 evidence state 是否是停止决策的必要信息？

**→ 对应实验现象**  Phase 1 中 Query+Evidence Macro F1 `0.7256850737817263`、FSR `0.16009852216748768`，Query-only 分别为 `0.4240929411333591`、`0.6674876847290641`。Phase 3A paired bootstrap 中 ScoreAwareBaseline−QueryStage 的 Macro F1 差为 `+0.18596545414551247`，95% CI `[0.14818097525341, 0.2198967643819512]`；FSR 差为 `-0.32275132275132273`，CI `[-0.3874353007860861, -0.2541887866909472]`。

**→ 为什么现有简单方案不够**  问题难度和 stage 只能表示先验，不能区分同一问题在不同检索结果下的真实覆盖状态。

**→ 我们采取的设计**  以 question + cumulative packed evidence 为输入，显式训练 Stop/Continue critic，并把 terminal retrieval failure 与 actionable-stage false stop 分开。

**→ 支持该设计的实验**  Phase 1 输入消融、Phase 2 表示比较、Phase 3A 多 seed/paired bootstrap、Phase 4 2Wiki Dev gate 和 heldout classification。

**→ 可以支持的结论**  当前 evidence 对充分性判别具有超出 query/stage shortcut 的实证价值。

**→ 不能支持的结论**  不能据此证明模型进行了正确的多跳推理，也不能把高 AUROC等同于阈值风险保证。

## 6. Hard Partial

**研究问题**  False Stop 是否主要发生在“高度相关但尚不完整”的 evidence 上？

**→ 对应实验现象**  HotpotQA Test 的 378 个 Continue decisions 中，Hard Partial 有 323 个，占 `85.4%`；FinalController 的 Hard Partial FSR 为 `0.4241486068111455`，137/159 个 false stops 来自 Hard Partial。Medium Partial FSR 更高，为 `0.4666666666666667`，但仅 45 个样本。

**→ 为什么现有简单方案不够**  只看总体 Stop F1 会被多数类主导；简单相关性线索容易把部分覆盖误判为充分。

**→ 我们采取的设计**  按 supporting-fact coverage 划分 Easy/Medium/Hard Continue，报告 bucket FSR、stop probability 与 false-stop 构成；另测试 hard-partial-aware sampling。

**→ 支持该设计的实验**  Phase 3A hard-partial analysis、三 seed sampling ablation、2Wiki recoverability diagnostics。

**→ 可以支持的结论**  False Stop 按绝对数量集中在 Hard Partial，主要因为 Hard Partial 在 Continue 类中占比最高。

**→ 不能支持的结论**  不能声称 Hard Partial 的条件错误率必然最高；本实验中 Medium Partial 的条件 FSR 更高但样本较少。

## 7. Coverage Auxiliary

**研究问题**  显式预测 evidence coverage 是否能稳定改善 Stop/Continue 判别？

**→ 对应实验现象**  三 seed Test 上 coverage MAE 为 `0.06741900107926792 ± 0.009521439052141537`，Spearman 为 `0.4758139512814535 ± 0.018309191531812943`；但 Macro F1 从 baseline `0.7970763875423774` 变为 `0.7964713902302248`，AUROC 从 `0.9246307529424743` 变为 `0.9146921248370523`，FSR 均为 `0.4514991181657848`。

**→ 为什么现有简单方案不够**  可学习 coverage 不等于 coverage signal 会转化成更好的分类边界；固定 seed 的改善也可能不跨训练随机性稳定。

**→ 我们采取的设计**  共享 backbone，增加 SmoothL1 coverage regression head，以 Dev 选择 λ=0.1，并同时报告 coverage 与 Controller 指标。

**→ 支持该设计的实验**  Phase 3A λ selection、三 seed ablation、固定 seed-42 question bootstrap。

**→ 可以支持的结论**  coverage 是可学习的辅助信号；seed-42 上 AUROC/FSR 有有利变化，但多 seed 未显示稳定分类改善。

**→ 不能支持的结论**  `UNSUPPORTED`：Coverage Auxiliary 稳健提升了 Controller，或稳定降低了 Hard Partial FSR。

## 8. Risk-constrained stopping

**研究问题**  能否在 Dev 上按预设 false-stop risk 选择保守阈值，并在冻结 Test/heldout 上保持质量同时降低成本？

**→ 对应实验现象**  HotpotQA Risk10 Test FSR `0.08445945945945946`，95% CI `[0.05638965371174415, 0.11765174222555261]`；与 FixedHeavy 的 Answer F1 差 `-0.00039970989794524403`，CI `[-0.019256642605760238, 0.018541145847616475]`，chunks 差 `-11.266`，延迟差 `-240.1202614759095 ms`。2Wiki Risk10 heldout FSR `0.12310521976279572`，CI `[0.11810921353389348, 0.12837358336460408]`，完整高于 10% 目标。

**→ 为什么现有简单方案不够**  默认 0.5 阈值或只优化 Stop F1 不控制错误停止；点估计达标也不等于 CI 或跨数据集风险保证。

**→ 我们采取的设计**  分离 dev_calibration 与 dev_policy，在 Dev 上冻结 temperature、risk target 与 conservative threshold，再一次性评估 Test/heldout，并用 question-level bootstrap。

**→ 支持该设计的实验**  Phase 3B HotpotQA frozen Test；Phase 4 2Wiki frozen heldout；2,000/5,000 次 question bootstrap。

**→ 可以支持的结论**  风险阈值提供了可解释的风险—效率工作点；HotpotQA 上观察 FSR 接近目标并显著降成本；2Wiki 上质量非劣且降成本，但风险目标未泛化。

**→ 不能支持的结论**  `UNSUPPORTED`：方法提供分布无关风险保证、跨数据集始终满足 α=10%，或 Temperature Scaling 本身带来独立部署收益。

## 9. HotpotQA

**研究问题**  在 retrieval-sufficient 程度较高的数据上，动态停止能否减少冗余检索而基本保持 QA 质量？

**→ 对应实验现象**  FixedHeavy complete coverage `0.922`，Risk10 coverage `0.906`；F1 分别为 `0.510839232130099` 和 `0.5104395222321538`；chunks 从 `20.646` 降至 `9.38`，reranker calls 从 `1.0` 降至 `0.207`。

**→ 为什么现有简单方案不够**  FixedHeavy 对容易问题仍执行完整 Rerank；浅层 fixed 策略则 FSR 高、质量下降。

**→ 我们采取的设计**  Evidence-aware score-aware critic + Dev 冻结保守 Risk10 阈值 + 固定生成评估协议。

**→ 支持该设计的实验**  Phase 3A 判别实验、Phase 3B end-to-end Test 与 paired bootstrap。

**→ 可以支持的结论**  在该冻结 HotpotQA 设置中，Risk10 在声明的 -0.02 F1 非劣界限内减少了检索量与延迟。

**→ 不能支持的结论**  不能声称 95% CI 完全低于 10% FSR，也不能将该 Test 描述为 Phase 3B 的全新独立样本。

## 10. 2Wiki retrieval ceiling

**研究问题**  当检索器本身经常无法取回完整证据时，sufficiency 与继续检索价值如何分离？

**→ 对应实验现象**  2Wiki heldout Dense/Hybrid/Final complete coverage 为 `0.317589058524173`/`0.36506043256997456`/`0.39782124681933845`，最终 TRFR 为 `0.6021787531806615`；heldout Dense→Final rescue 为 1009/8582 (`0.11757166161733862`)，Hybrid→Final rescue 为 412/7985 (`0.05159674389480275`)。Train-derived dev_policy 的相应 rescue rates 为 `0.06489675516224189` 与 `0.02386451116243264`。

**→ 为什么现有简单方案不够**  “当前不足”不等价于“下一阶段可恢复”；总是 Continue 可能持续支付成本却无法补齐证据。

**→ 我们采取的设计**  分离 Controller FSR、Recoverable FSR、Unrecoverable Early Stop 与 Terminal Retrieval Failure，并构建 analysis-only retrieval oracle。

**→ 支持该设计的实验**  Phase 4 retrieval ceiling、stage rescue、heldout retrieval table 和 recoverability analysis。

**→ 可以支持的结论**  在冻结 2Wiki ladder 中，主要瓶颈包含显著 retrieval ceiling；证据充分性与未来检索效用是相关但不同的目标。

**→ 不能支持的结论**  不能声称 2Wiki 本身有 60% 不可回答，也不能声称更强的 query reformulation/retriever 无法突破该 ceiling。

## 11. Adaptive-RAG-style Router

**研究问题**  检索前基于 query complexity 的路由，能否达到 evidence-conditioned sequential stopping 的质量—成本边界？

**→ 对应实验现象**  2Wiki heldout 三 seed Router route Macro F1 为 `0.5935941014885876 ± 0.0024099543864248073`，Medium route rate 为 `0.0`；mean Answer F1 `0.19576064593012324`、chunks `15.487171331636981`。Risk10 F1 `0.20256212239279647`、chunks `13.268686386768447`。seed42−Risk10 F1 差 CI 为 `[-0.00919986490927534, -0.004277691741995629]`，chunks 差 CI 为 `[2.0531130725190825, 2.2430065203562353]`。

**→ 为什么现有简单方案不够**  Pre-retrieval router 看不到实际 retrieval outcome，并且三分类准确率会掩盖完全不预测 Medium 的失败模式。

**→ 我们采取的设计**  将 Adapted query-only Router 作为同 ladder baseline，与 evidence-aware policy 在相同 heldout 和成本口径下比较。

**→ 支持该设计的实验**  Phase 4 三 seed Dev gate、三 seed heldout、seed42 paired question bootstrap。

**→ 可以支持的结论**  对本研究实现的 Adaptive-RAG-style Router，evidence-conditioned Risk10 得到更好的冻结质量—成本工作点。

**→ 不能支持的结论**  不能声称复现或击败官方 Adaptive-RAG；该 baseline 是 adapted router，不是官方系统。

## 12. S2G-RAG

**研究问题**  与同样读取 question+evidence 的 gap-aware iterative RAG 相比，我们的任务边界在哪里？

**→ 对应实验现象**  官方代码与协议审计完成，但 mandatory trained S2G-Judge LoRA 未公开；smoke execution 为 `not_started_stop_condition`，N=`0`，无 EM/F1。

**→ 为什么现有简单方案不够**  仅比较“是否读取 evidence”会忽略 S2G-RAG 的 gap generation、new query generation 与 evidence memory construction。

**→ 我们采取的设计**  仅进行方法级对位与可复现性审计，不自行重训或用不同模型替代官方 checkpoint。

**→ 支持该设计的实验**  external baseline protocol audit 与 reproduction status；没有数值实验。

**→ 可以支持的结论**  两者共享 evidence sufficiency state，但 S2G-RAG重点在缺口诊断和证据获取，我们重点在 partial/sufficient 判别、保守阈值和风险—效率边界。

**→ 不能支持的结论**  `UNSUPPORTED`：Ours 数值优于 S2G-RAG，或 S2G-RAG 方法失败。

## 13. Stop-RAG

**研究问题**  当前 evidence sufficiency 与 expected future retrieval utility 是否是同一个学习目标？

**→ 对应实验现象**  Stop-RAG 官方流程需要完整 trajectory、重复 answer sampling、TD(λ) value training，并报告使用 4×H100；未识别到可直接推理的官方 value checkpoint，因此实验按规则跳过。

**→ 为什么现有简单方案不够**  在 retrieval-limited regime，evidence 不足并不保证继续当前 ladder 有用；仅预测 sufficiency 不等于估计 continuation value。

**→ 我们采取的设计**  保持 Ours 的 sufficiency 定义，同时通过 rescue/TRFR/RFSR 诊断两者边界，不把 Ours 标签改造成 Stop-RAG value target。

**→ 支持该设计的实验**  Stop-RAG protocol audit；2Wiki retrieval ceiling 与 recoverability diagnostics。

**→ 可以支持的结论**  Sufficiency 与 future utility 是两个可分离的研究问题，2Wiki 冻结结果为这种分离提供实证动机。

**→ 不能支持的结论**  `UNSUPPORTED`：Ours 优于 Stop-RAG，或 Stop-RAG 能/不能解决本研究 2Wiki ceiling。

## 14. SURE-RAG

**研究问题**  Retrieval stopping 与 candidate-answer verification 是否应视为同一任务？

**→ 对应实验现象**  方法审计显示 SURE-RAG 输入为 question+candidate answer+evidence，输出 Supported/Refuted/Insufficient，再决定 answer/abstain；本阶段预注册为不运行实验。

**→ 为什么现有简单方案不够**  二者都谈“证据可靠性”，但决策时机、输入、动作和 risk denominator 不同，强行数值比较会混淆任务。

**→ 我们采取的设计**  将 SURE-RAG 放在 answer verification/selective answering 相关工作中，与 retrieval stopping 明确分节。

**→ 支持该设计的实验**  SURE-RAG method audit；无数值实验。

**→ 可以支持的结论**  SURE-RAG 与 Ours 共享 evidence reliability 关注点，但前者控制回答/弃答，后者控制检索停止/继续。

**→ 不能支持的结论**  `UNSUPPORTED`：两者在同一风险指标上可直接比较，或 Ours 在 answer verification 上优于 SURE-RAG。

# 论文 V1 最终数字清单

## 使用规则

- 本清单定义 V1 准备进入正文、主表或紧邻正文补充表的数字；没有执行任何重算。
- `Frozen heldout` 的“是”仅表示该数字来自已经冻结且未调参的 Test/heldout 执行；Dev、审计样本与早期 evaluation split 标为“否”。
- “正文可用=限定”表示可以写入正文，但必须连同该行限制条件一起表述。
- 显示精度可在排版阶段统一，但原始值以 `source file` 为准。

## A. 输入状态与表示选择

| Dataset | Split | Experiment | Source file | Metric | Value | Frozen heldout | 正文可用 |
|---|---|---|---|---|---:|---|---|
| HotpotQA | Phase 1 evaluation | Query-only | `results/critic/comparison_20260902T125642Z/critic_comparison.csv` | Macro F1 | 0.4240929411333591 | 否 | 限定：早期输入消融 |
| HotpotQA | Phase 1 evaluation | Query-only | 同上 | False Stop Rate | 0.6674876847290641 | 否 | 限定：早期输入消融 |
| HotpotQA | Phase 1 evaluation | Query-only | 同上 | AUROC | 0.7785796328708343 | 否 | 限定：早期输入消融 |
| HotpotQA | Phase 1 evaluation | Query+Evidence | 同上 | Macro F1 | 0.7256850737817263 | 否 | 限定：早期输入消融 |
| HotpotQA | Phase 1 evaluation | Query+Evidence | 同上 | False Stop Rate | 0.16009852216748768 | 否 | 限定：早期输入消融 |
| HotpotQA | Phase 1 evaluation | Query+Evidence | 同上 | AUROC | 0.9440719672910167 | 否 | 限定：早期输入消融 |
| HotpotQA | Test, seed 42 | Query-only cumulative | `results/phase2/controller_baselines.csv` | Macro F1 | 0.5483483483483483 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query-only cumulative | 同上 | FSR | 0.9021164021164021 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query+Stage | 同上 | Macro F1 | 0.6012119866632133 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query+Stage | 同上 | FSR | 0.8068783068783069 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query+Evidence concat-truncate | 同上 | Macro F1 | 0.7648037920865522 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query+Evidence concat-truncate | 同上 | FSR | 0.5 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query+Evidence uniform packing | 同上 | Macro F1 | 0.702467381609243 | 是 | 可放补充表 |
| HotpotQA | Test, seed 42 | Query+Evidence uniform packing | 同上 | FSR | 0.6534391534391535 | 是 | 可放补充表 |
| HotpotQA | Test, seed 42 | Query+Evidence score-aware | 同上 | Macro F1 | 0.7871774408087258 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query+Evidence score-aware | 同上 | AUROC | 0.9169572481929462 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query+Evidence score-aware | 同上 | FSR | 0.48412698412698413 | 是 | 是 |

## B. 多 seed 稳定性与 Hard Partial

| Dataset | Split | Experiment | Source file | Metric | Value | Frozen heldout | 正文可用 |
|---|---|---|---|---|---:|---|---|
| HotpotQA | Test, 3 seeds | Query+Stage | `results/phase3/multi_seed_summary.csv` | Macro F1 mean ± sample std | 0.5773192401362102 ± 0.035058973661818135 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Query+Stage | 同上 | AUROC mean ± sample std | 0.8207468483339321 ± 0.008139434985352037 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Query+Stage | 同上 | FSR mean ± sample std | 0.8465608465608465 ± 0.0642045560820719 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Query+Evidence concat | 同上 | Macro F1 mean ± sample std | 0.7523233785363042 ± 0.016917681271698157 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Query+Evidence concat | 同上 | AUROC mean ± sample std | 0.9081029196716967 ± 0.010805204216729652 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Query+Evidence concat | 同上 | FSR mean ± sample std | 0.5361552028218695 ± 0.05589048281301882 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Score-aware baseline | 同上 | Macro F1 mean ± sample std | 0.7970763875423774 ± 0.013309348361898274 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Score-aware baseline | 同上 | AUROC mean ± sample std | 0.9246307529424743 ± 0.007180620389277922 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Score-aware baseline | 同上 | FSR mean ± sample std | 0.4514991181657848 ± 0.04152119000703193 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Score-aware baseline | 同上 | Hard Partial FSR mean ± sample std | 0.45717234262125905 ± 0.03667565664379124 | 是 | 是 |
| HotpotQA | Test, seed 42 | Continue distribution | `results/phase3/hard_partial/hard_partial_summary.csv` | Easy / Medium / Hard samples | 10 / 45 / 323 | 是 | 是 |
| HotpotQA | Test, seed 42 | Query+Stage | 同上 | Hard Partial FSR | 0.804953560371517 | 是 | 是 |
| HotpotQA | Test, seed 42 | Score-aware baseline | 同上 | Hard Partial FSR | 0.47678018575851394 | 是 | 是 |
| HotpotQA | Test, seed 42 | FinalController | 同上 | Hard Partial FSR | 0.4241486068111455 | 是 | 限定：固定 seed |
| HotpotQA | Test, seed 42 | FinalController | 同上 | Medium Partial FSR | 0.4666666666666667 | 是 | 限定：n=45 |

## C. Coverage Auxiliary 与 Sampling Ablation

| Dataset | Split | Experiment | Source file | Metric | Value | Frozen heldout | 正文可用 |
|---|---|---|---|---|---:|---|---|
| HotpotQA | Test, 3 seeds | Coverage auxiliary λ=0.1 | `results/phase3/ablation_summary.csv` | Coverage MAE mean ± std | 0.06741900107926792 ± 0.009521439052141537 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Coverage auxiliary λ=0.1 | 同上 | Coverage Spearman mean ± std | 0.4758139512814535 ± 0.018309191531812943 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Coverage auxiliary λ=0.1 | 同上 | Macro F1 mean ± std | 0.7964713902302248 ± 0.010093408925214108 | 是 | 是：用于否定稳定提升 |
| HotpotQA | Test, 3 seeds | Coverage auxiliary λ=0.1 | 同上 | AUROC mean ± std | 0.9146921248370523 ± 0.02232909648359574 | 是 | 是：用于否定稳定提升 |
| HotpotQA | Test, 3 seeds | Coverage auxiliary λ=0.1 | 同上 | FSR mean ± std | 0.4514991181657848 ± 0.026761888722210782 | 是 | 是：用于否定稳定提升 |
| HotpotQA | Test, 3 seeds | Hard-partial-aware sampling | `results/phase3/sampling_summary.csv` | Macro F1 mean ± std | 0.7914340623221654 ± 0.010600938473960968 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Hard-partial-aware sampling | 同上 | AUROC mean ± std | 0.9057730544826909 ± 0.0076721139910822375 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Hard-partial-aware sampling | 同上 | Hard Partial FSR mean ± std | 0.4726522187822497 ± 0.04739303406895783 | 是 | 是 |
| HotpotQA | Test, 3 seeds | Balanced Stop/Continue | 同上 | Macro F1 mean ± std | 0.7833329476560728 ± 0.013702444686430771 | 是 | 可放补充表 |
| HotpotQA | Test, 3 seeds | Balanced Stop/Continue | 同上 | FSR mean ± std | 0.4832451499118166 ± 0.04219003073688073 | 是 | 可放补充表 |

## D. 标签审计与 fixed-model bootstrap

| Dataset | Split | Experiment | Source file | Metric | Value | Frozen heldout | 正文可用 |
|---|---|---|---|---|---:|---|---|
| HotpotQA | Dev audit | Manual sufficiency audit | `docs/manual_audit_analysis.md` | Audited records | 180 | 否 | 是，需声明 stratified/single reviewer |
| HotpotQA | Dev audit | Automatic Stop vs Human Stop | 同上 | Accuracy / Macro F1 | 1.0 / 1.0 | 否 | 限定 |
| HotpotQA | Dev audit | Automatic 3-class vs Human 3-class | 同上 | Accuracy / Macro F1 | 1.0 / 1.0 | 否 | 限定 |
| HotpotQA | Dev audit | Full packed visibility | 同上 | all-gold-visible / some-not-visible | 22 / 158 | 否 | 是 |
| HotpotQA | Test, seed 42 | ScoreAwareBaseline−QueryStage | `results/phase3/bootstrap/paired_comparison_summary.csv` | Macro F1 delta, 95% CI | 0.18596545414551247; [0.14818097525341, 0.2198967643819512] | 是 | 是 |
| HotpotQA | Test, seed 42 | ScoreAwareBaseline−QueryStage | 同上 | AUROC delta, 95% CI | 0.10215605438717579; [0.0730343680380726, 0.12981707553065983] | 是 | 是 |
| HotpotQA | Test, seed 42 | ScoreAwareBaseline−QueryStage | 同上 | FSR delta, 95% CI | -0.32275132275132273; [-0.3874353007860861, -0.2541887866909472] | 是 | 是 |
| HotpotQA | Test, seed 42 | FinalController−ScoreAwareBaseline | 同上 | Macro F1 delta, 95% CI | 0.02087126577497811; [-0.002543580220234376, 0.04500820403965164] | 是 | 限定：CI 含 0 |
| HotpotQA | Test, seed 42 | FinalController−ScoreAwareBaseline | 同上 | AUROC delta, 95% CI | 0.015398298483729311; [0.002802029370247394, 0.028245429007750764] | 是 | 限定：固定 seed |
| HotpotQA | Test, seed 42 | FinalController−ScoreAwareBaseline | 同上 | FSR delta, 95% CI | -0.06349206349206349; [-0.11363636363636365, -0.01550327597379224] | 是 | 限定：固定 seed |

## E. HotpotQA frozen end-to-end risk boundary

| Dataset | Split | Experiment | Source file | Metric | Value | Frozen heldout | 正文可用 |
|---|---|---|---|---|---:|---|---|
| HotpotQA | Test, 1,000 questions | FixedDense | `results/phase3b/test_risk_policy.csv` | Answer F1 / coverage / chunks | 0.46591004060267216 / 0.8 / 5.0 | 是 | 是 |
| HotpotQA | Test, 1,000 questions | FixedHybrid | 同上 | Answer F1 / coverage / chunks | 0.5074312016638333 / 0.9 / 11.504 | 是 | 是 |
| HotpotQA | Test, 1,000 questions | FixedHeavy | 同上 | Answer EM / F1 | 0.407 / 0.510839232130099 | 是 | 是 |
| HotpotQA | Test, 1,000 questions | FixedHeavy | 同上 | Coverage / chunks / reranker / latency ms | 0.922 / 20.646 / 1.0 / 587.7891366463155 | 是 | 是 |
| HotpotQA | Test, 1,000 questions | Temperature conservative Risk10 | 同上 | FSR / Hard Partial FSR | 0.08445945945945946 / 0.08433734939759036 | 是 | 是 |
| HotpotQA | Test, 1,000 questions | Temperature conservative Risk10 | 同上 | Answer EM / F1 | 0.409 / 0.5104395222321538 | 是 | 是 |
| HotpotQA | Test, 1,000 questions | Temperature conservative Risk10 | 同上 | Coverage / chunks / reranker / latency ms | 0.906 / 9.38 / 0.207 / 347.66887517040595 | 是 | 是 |
| HotpotQA | Test bootstrap, 2,000 | Risk10−FixedHeavy | `results/phase3b/bootstrap_ci.csv` | Answer F1 delta, 95% CI | -0.00039970989794524403; [-0.019256642605760238, 0.018541145847616475] | 是 | 是 |
| HotpotQA | Test bootstrap, 2,000 | Risk10−FixedHeavy | 同上 | Chunks delta, 95% CI | -11.266; [-11.661025, -10.850975000000002] | 是 | 是 |
| HotpotQA | Test bootstrap, 2,000 | Risk10−FixedHeavy | 同上 | Latency delta ms, 95% CI | -240.1202614759095; [-250.08015591200675, -229.57471854693722] | 是 | 是 |
| HotpotQA | Test bootstrap, 2,000 | Risk10 | 同上 | FSR 95% CI | [0.05638965371174415, 0.11765174222555261] | 是 | 是：明确 CI 未全低于 10% |

## F. 2Wiki retrieval ceiling 与 evidence-aware Controller

| Dataset | Split | Experiment | Source file | Metric | Value | Frozen heldout | 正文可用 |
|---|---|---|---|---|---:|---|---|
| 2Wiki | heldout, 12,576 | Dense@5 | `results/phase4/final/table_retrieval.csv` | SF recall / complete coverage | 0.6526479007633587 / 0.317589058524173 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Hybrid@10 | 同上 | SF recall / complete coverage | 0.6872269932145886 / 0.36506043256997456 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Rerank@20 | 同上 | SF recall / complete coverage | 0.7054760390161152 / 0.39782124681933845 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Rerank@20 | 同上 | Terminal Retrieval Failure Rate | 0.6021787531806615 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Gold facts=2, Rerank@20 | 同上 | Questions / complete coverage | 9805 / 0.5068842427332994 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Gold facts=4+, Rerank@20 | 同上 | Questions / complete coverage | 2751 / 0.0054525627044711015 | 是 | 限定：描述性难度分层 |
| 2Wiki | heldout, 12,576 | Dense→Final rescue | `results/phase4/2wiki/heldout/stage_rescue_analysis.csv` | Rescued / eligible / rate | 1009 / 8582 / 0.11757166161733862 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Hybrid→Final rescue | 同上 | Rescued / eligible / rate | 412 / 7985 / 0.05159674389480275 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Retrieval oracle | `results/phase4/2wiki/heldout/retrieval_ceiling.json` | Avg chunks / reranker calls / retrieval latency ms | 14.994115776081426 / 0.6349395674300254 / 916.2910005067052 | 是 | 限定：analysis-only oracle |
| 2Wiki | Train-derived dev_policy | Dense→Final rescue | `results/phase4/2wiki/stage_rescue_analysis.csv` | Rescued / eligible / rate | 88 / 1356 / 0.06489675516224189 | 否 | 是：机制分析 |
| 2Wiki | Train-derived dev_policy | Hybrid→Final rescue | 同上 | Rescued / eligible / rate | 31 / 1299 / 0.02386451116243264 | 否 | 是：机制分析 |
| 2Wiki | Train-derived dev_policy, 3 seeds | Final evidence-aware | `results/phase4/2wiki/controller_dev/dev_multiseed_summary.csv` | Sequential Macro F1 mean ± std | 0.9872524228858328 ± 0.0005155331299172127 | 否 | 是：Dev 稳定性 |
| 2Wiki | Train-derived dev_policy, 3 seeds | Final evidence-aware | 同上 | Sequential FSR mean ± std | 0.004651147133451258 ± 0.0002172126698360667 | 否 | 是：Dev 稳定性 |
| 2Wiki | Train-derived dev_policy, 3 seeds | Query+Stage | 同上 | Sequential Macro F1 mean ± std | 0.9299401096178403 ± 0.0029213218786967913 | 否 | 是：Dev 对比 |
| 2Wiki | Train-derived dev_policy, 3 seeds | Query+Stage | 同上 | Sequential FSR mean ± std | 0.028131942983359897 ± 0.006620203270093875 | 否 | 是：Dev 对比 |
| 2Wiki | heldout, 12,576 | EvidenceAwareClassification | `results/phase4/final/table_controller.csv` | Macro F1 / AUROC / FSR | 0.9610653159404154 / 0.9950041820049874 / 0.017046833292890074 | 是 | 是 |
| 2Wiki | heldout, 12,576 | EvidenceAwareClassification | 同上 | UER / Hard Partial FSR | 0.06103589082312486 / 0.017456668944523825 | 是 | 是 |

## G. 2Wiki frozen risk-efficiency and external baselines

| Dataset | Split | Experiment | Source file | Metric | Value | Frozen heldout | 正文可用 |
|---|---|---|---|---|---:|---|---|
| 2Wiki | heldout, 12,576 | FixedHeavy | `results/phase4/final/table_end_to_end.csv` | Answer EM / F1 | 0.16777989821882952 / 0.19215672075640125 | 是 | 是 |
| 2Wiki | heldout, 12,576 | FixedHeavy | 同上 | Coverage / chunks / reranker / latency ms | 0.39782124681933845 / 20.293256997455472 / 1.0 / 1676.3052861272201 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Risk10 | `results/phase4/final/table_end_to_end.csv`（点估计）；`results/phase4/final/bootstrap_ci.csv`（区间，comparison=temperature_conservative_risk_10, metric=false_stop_rate） | FSR / 95% CI | 0.12310521976279572 / [0.11810921353389348, 0.12837358336460408] | 是 | 是：明确未达 10% |
| 2Wiki | heldout, 12,576 | Risk10 | `results/phase4/final/table_end_to_end.csv` | Answer EM / F1 | 0.17692430025445294 / 0.20256212239279647 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Risk10 | 同上 | Coverage / chunks / reranker / latency ms | 0.35520038167938933 / 13.268686386768447 / 0.5010337150127226 / 1124.3924006067339 | 是 | 是 |
| 2Wiki | heldout bootstrap, 5,000 | Risk10−FixedHeavy | `results/phase4/final/bootstrap_ci.csv` | Answer F1 delta, 95% CI | 0.010405401636395223; [0.005712640327771366, 0.015261824576448622] | 是 | 是 |
| 2Wiki | heldout bootstrap, 5,000 | Risk10−FixedHeavy | 同上 | Chunks delta, 95% CI | -7.024570610687025; [-7.153627942111961, -6.897662213740459] | 是 | 是 |
| 2Wiki | heldout bootstrap, 5,000 | Risk10−FixedHeavy | 同上 | Reranker delta, 95% CI | -0.49896628498727735; [-0.5078741253180662, -0.49029898218829515] | 是 | 是 |
| 2Wiki | heldout bootstrap, 5,000 | Risk10−FixedHeavy | 同上 | Latency delta ms, 95% CI | -551.912885520486; [-564.2714045058193, -539.5956550606721] | 是 | 是 |
| 2Wiki | heldout, 12,576 | ScoreThresholdRisk10 | `results/phase4/final/table_external_baselines.csv` | FSR / UER / chunks | 0.0 / 1.0 / 20.293256997455472 | 是 | 是：退化 baseline |
| 2Wiki | heldout, 12,576 | Judge-512 | 同上 | FSR / UER | 0.03152073732718894 / 0.7382474080386309 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Judge-512 | 同上 | Answer F1 / chunks / latency ms | 0.19794994770897936 / 17.685114503816795 / 1788.9903085523201 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Judge-Full | 同上 | FSR / UER | 0.019734832284474858 / 0.767405951712521 | 是 | 是 |
| 2Wiki | heldout, 12,576 | Judge-Full | 同上 | Answer F1 / chunks / latency ms | 0.19844733253365426 / 18.041825699745548 / 1890.598273004002 | 是 | 是 |
| 2Wiki | heldout, 3 seeds | Adaptive-RAG-style Router | 同上 | Route Macro F1 mean ± std | 0.5935941014885876 ± 0.0024099543864248073 | 是 | 是：adapted baseline |
| 2Wiki | heldout, 3 seeds | Adaptive-RAG-style Router | 同上 | Medium route rate mean ± std | 0.0 ± 0.0 | 是 | 是：失败模式 |
| 2Wiki | heldout, 3 seeds | Adaptive-RAG-style Router | 同上 | Answer F1 mean ± std | 0.19576064593012324 ± 0.000500160184901776 | 是 | 是：描述性 |
| 2Wiki | heldout, 3 seeds | Adaptive-RAG-style Router | 同上 | Chunks mean ± std | 15.487171331636981 ± 0.1937703426694323 | 是 | 是：描述性 |
| 2Wiki | heldout bootstrap, seed42 | Router−Risk10 | `results/phase4/final/bootstrap_ci.csv` | Answer F1 delta, 95% CI | -0.006747500054927719; [-0.00919986490927534, -0.004277691741995629] | 是 | 限定：seed42 adapted baseline |
| 2Wiki | heldout bootstrap, seed42 | Router−Risk10 | 同上 | Chunks delta, 95% CI | 2.148616412213741; [2.0531130725190825, 2.2430065203562353] | 是 | 限定：seed42 adapted baseline |

## H. External recent methods

| Dataset | Split | Experiment | Source file | Metric | Value | Frozen heldout | 正文可用 |
|---|---|---|---|---|---:|---|---|
| HotpotQA | External reference | S2G-RAG official pipeline | `results/external_baselines/s2g_reference_summary.csv` | Executed N | 0 | 否 | 否：只能写复现状态，不能写性能 |
| 2Wiki | External reference | S2G-RAG official pipeline | 同上 | Executed N | 0 | 否 | 否：只能写复现状态，不能写性能 |
| HotpotQA/2Wiki/MuSiQue | Protocol audit | Stop-RAG | `results/external_baselines/external_baseline_manifest.json` | Experiment skipped | true | 否 | 可用于方法/复现性说明 |
| N/A | Method audit | SURE-RAG | 同上 | Experiment run | false | 否 | 可用于任务边界说明 |

## 不进入正文主张的数字

- S2G-RAG 的 `N=0` 不是零性能，禁止放入性能比较表。
- Phase 1 truncation ratio、单次 latency 等可作为审计背景，不与 Phase 2–4 正式结果混表。
- 2Wiki Dev 上 Risk10 FSR `0.05206738131699847` 只能说明阈值选择依据；heldout 结论必须使用 `0.12310521976279572`。
- Coverage Auxiliary 的 seed-42 favorable bootstrap 不得替代三 seed 无稳定改善的结论。
- LLM Judge hard labels 没有合法 AUROC，禁止人为构造概率或 threshold sweep。

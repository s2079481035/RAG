# Oracle Early Stopping（Phase 2b）
> 协议: EXPERIMENT_PROTOCOL.md 第 9 节（2026-08-19 冻结）

## nq（n=1000）

### Oracle Early Stopping vs Fixed（Recall / 平均成本）
| 系统 | Recall@final | FullCoverage | 平均文档成本 | 平均阶梯级 | 不可充分满足 |
|---|---|---|---|---|---|
| rerank@10(fixed) | 0.951 | 0.951 | 10 | - | - |
| rerank@20(fixed) | 0.970 | 0.970 | 20 | - | - |
| oracleES-A | 0.972 | 0.972 | 4.07 | 1.33 | 28 |
| oracleES-B | 0.980 | 0.980 | 10.27 | 1.14 | 20 |

### 各阶段停止比例（Oracle Early Stopping）
| 阶梯 | 阶段分布（level 1..n 停止的查询数） |
|---|---|
| A | S1:834(83.4%) S2:89(8.9%) S3:22(2.2%) S4:20(2.0%) S5:35(3.5%) |
| B | S1:925(92.5%) S2:36(3.6%) S3:12(1.2%) S4:27(2.7%) |

> False Early Stop（oracle 口径）：按构造为 0（充分性由 gold 判定，不会误停）。False Early Stop 只在 Critic 阶段有定义：Critic 判 sufficient 但 gold 未被覆盖的查询比例。Oracle 阶段同时报告不可充分满足数作为 recall 天花板信息。

## hotpotqa（n=1000）

### Oracle Early Stopping vs Fixed（Recall / 平均成本）
| 系统 | Recall@final | FullCoverage | 平均文档成本 | 平均阶梯级 | 不可充分满足 |
|---|---|---|---|---|---|
| rerank@10(fixed) | 0.948 | 0.897 | 10 | - | - |
| rerank@20(fixed) | 0.956 | 0.914 | 20 | - | - |
| oracleES-A | 0.959 | 0.919 | 5.39 | 1.70 | 81 |
| oracleES-B | 0.962 | 0.926 | 10.82 | 1.35 | 74 |

### 各阶段停止比例（Oracle Early Stopping）
| 阶梯 | 阶段分布（level 1..n 停止的查询数） |
|---|---|
| A | S1:684(68.4%) S2:146(14.6%) S3:46(4.6%) S4:33(3.3%) S5:91(9.1%) |
| B | S1:833(83.3%) S2:61(6.1%) S3:24(2.4%) S4:82(8.2%) |

> False Early Stop（oracle 口径）：按构造为 0（充分性由 gold 判定，不会误停）。False Early Stop 只在 Critic 阶段有定义：Critic 判 sufficient 但 gold 未被覆盖的查询比例。Oracle 阶段同时报告不可充分满足数作为 recall 天花板信息。


## 延迟（实测，GPU1, n=200, 增量模型）

| 阶段累计 | 延迟(ms) |
|---|---|
| S1 embed+dense | 31.5 |
| S2/S3 +BM25+RRF | 93.5 |
| S4/S5 +rerank(20 pairs) | 220.9 |

| 系统 | 平均延迟(ms) | vs fixed |
|---|---|---|
| fixed rerank@10/20 | 220.9 | 1.0x |
| NQ ES-A | 48.8 | 4.5x |
| NQ ES-B | 39.6 | 5.6x |
| HotpotQA ES-A | 66.9 | 3.3x |
| HotpotQA ES-B | 52.3 | 4.2x |

> 增量模型：embed 与 dense 全库搜索只付一次，hybrid 与 rerank 只付一次（rerank 恒对 top-20 取分，@10/@20 同价）。

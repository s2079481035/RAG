# 分层分析 (Stratified Recall)

## nq — 查询难度分布
easy=934(93.4%) | medium=39(3.9%) | hard=27(2.7%)

### nq — 按难度分组 Recall@K
| 数据集 | 分组 | 检索器 | R@3 | R@5 | R@10 | R@20 |
| --- | --- | --- | --- | --- | --- | --- |
| nq | easy | dense | 0.893 | 0.936 | 0.960 | 0.984 |
| nq | easy | hybrid | 0.911 | 0.960 | 0.982 | 0.998 |
| nq | easy | rerank | 0.921 | 0.959 | 0.988 | 0.998 |
| nq | hard | dense | 0.000 | 0.000 | 0.000 | 0.370 |
| nq | hard | hybrid | 0.000 | 0.000 | 0.000 | 0.259 |
| nq | hard | rerank | 0.000 | 0.000 | 0.000 | 0.259 |
| nq | medium | dense | 0.000 | 0.308 | 0.718 | 0.846 |
| nq | medium | hybrid | 0.000 | 0.205 | 0.538 | 0.795 |
| nq | medium | rerank | 0.000 | 0.462 | 0.718 | 0.795 |

### nq — Oracle Adaptive vs Fixed
| 策略 | Recall(全部gold覆盖) | 平均成本级(1-7) |
|---|---|---|
| dense@3 | 0.834 | - |
| dense@5 | 0.886 | - |
| hybrid@5 | 0.905 | - |
| hybrid@10 | 0.938 | - |
| rerank@10 | 0.951 | - |
| hybrid@20 | 0.970 | - |
| rerank@20 | 0.970 | - |
| ORACLE(全命中) | 0.974 | 1.30 |
| ORACLE(含未命中) | 0.974 | 1.30 |

## hotpotqa — 查询难度分布
easy=848(84.8%) | medium=71(7.1%) | hard=81(8.1%)

### hotpotqa — 按难度分组 Recall@K
| 数据集 | 分组 | 检索器 | R@3 | R@5 | R@10 | R@20 |
| --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | easy | dense | 0.898 | 0.937 | 0.965 | 0.978 |
| hotpotqa | easy | hybrid | 0.910 | 0.952 | 0.981 | 0.996 |
| hotpotqa | easy | rerank | 0.963 | 0.987 | 0.994 | 0.996 |
| hotpotqa | hard | dense | 0.444 | 0.457 | 0.469 | 0.519 |
| hotpotqa | hard | hybrid | 0.481 | 0.488 | 0.488 | 0.537 |
| hotpotqa | hard | rerank | 0.488 | 0.488 | 0.488 | 0.537 |
| hotpotqa | medium | dense | 0.493 | 0.704 | 0.796 | 0.887 |
| hotpotqa | medium | hybrid | 0.486 | 0.655 | 0.810 | 0.951 |
| hotpotqa | medium | rerank | 0.486 | 0.775 | 0.915 | 0.951 |

### HotpotQA — 按 question type 分组 Recall@K
| 数据集 | 分组 | 检索器 | R@3 | R@5 | R@10 | R@20 |
| --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | bridge | dense | 0.800 | 0.855 | 0.892 | 0.919 |
| hotpotqa | bridge | hybrid | 0.823 | 0.873 | 0.913 | 0.946 |
| hotpotqa | bridge | rerank | 0.871 | 0.916 | 0.935 | 0.946 |
| hotpotqa | comparison | dense | 0.973 | 0.997 | 1.000 | 1.000 |
| hotpotqa | comparison | hybrid | 0.944 | 0.984 | 1.000 | 1.000 |
| hotpotqa | comparison | rerank | 0.976 | 0.997 | 1.000 | 1.000 |

### HotpotQA — 按 supporting sentences 数量分组 Recall@K
| 数据集 | 分组 | 检索器 | R@3 | R@5 | R@10 | R@20 |
| --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | 2 | dense | 0.833 | 0.883 | 0.908 | 0.926 |
| hotpotqa | 2 | hybrid | 0.841 | 0.887 | 0.922 | 0.951 |
| hotpotqa | 2 | rerank | 0.881 | 0.926 | 0.943 | 0.951 |
| hotpotqa | 3+ | dense | 0.832 | 0.879 | 0.924 | 0.953 |
| hotpotqa | 3+ | hybrid | 0.855 | 0.909 | 0.946 | 0.968 |
| hotpotqa | 3+ | rerank | 0.914 | 0.944 | 0.958 | 0.968 |

### hotpotqa — Oracle Adaptive vs Fixed
| 策略 | Recall(全部gold覆盖) | 平均成本级(1-7) |
|---|---|---|
| dense@3 | 0.833 | - |
| dense@5 | 0.881 | - |
| hybrid@5 | 0.893 | - |
| hybrid@10 | 0.929 | - |
| rerank@10 | 0.948 | - |
| hybrid@20 | 0.956 | - |
| rerank@20 | 0.956 | - |
| ORACLE(全命中) | 0.960 | 1.51 |
| ORACLE(含未命中) | 0.960 | 1.51 |


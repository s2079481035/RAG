# Sufficiency 标签分布

标签按严格 supporting fact `[title, sentence_id]` coverage 构造：0=Insufficient，1=Partial，2=Sufficient。

## Supporting Fact 数量

| Gold facts / question | Questions |
|---:|---:|
| 2 | 4990 |
| 3 | 1774 |
| 4 | 537 |
| 5 | 80 |
| 6 | 14 |
| 7 | 9 |
| 8 | 1 |

## train

| Stage | Insufficient | Partial | Sufficient |
|---|---:|---:|---:|
| dense@5 | 65 | 1209 | 4131 |
| dense@10 | 32 | 882 | 4491 |
| hybrid@5 | 37 | 1093 | 4275 |
| hybrid@10 | 15 | 745 | 4645 |
| rerank@20 | 4 | 506 | 4895 |

Non-monotonic trajectories: 456 / 5405

Most common trajectories:

- `sufficient -> sufficient -> sufficient -> sufficient -> sufficient`: 3808
- `partial -> partial -> partial -> partial -> partial`: 394
- `partial -> sufficient -> sufficient -> sufficient -> sufficient`: 243
- `partial -> partial -> sufficient -> sufficient -> sufficient`: 208
- `sufficient -> sufficient -> partial -> sufficient -> sufficient`: 173
- `partial -> partial -> partial -> sufficient -> sufficient`: 136
- `sufficient -> sufficient -> partial -> partial -> sufficient`: 109
- `partial -> partial -> partial -> partial -> sufficient`: 100
- `partial -> sufficient -> partial -> sufficient -> sufficient`: 53
- `sufficient -> sufficient -> partial -> partial -> partial`: 38

## dev

| Stage | Insufficient | Partial | Sufficient |
|---|---:|---:|---:|
| dense@5 | 13 | 202 | 785 |
| dense@10 | 9 | 139 | 852 |
| hybrid@5 | 5 | 177 | 818 |
| hybrid@10 | 4 | 121 | 875 |
| rerank@20 | 2 | 79 | 919 |

Non-monotonic trajectories: 67 / 1000

Most common trajectories:

- `sufficient -> sufficient -> sufficient -> sufficient -> sufficient`: 737
- `partial -> partial -> partial -> partial -> partial`: 61
- `partial -> sufficient -> sufficient -> sufficient -> sufficient`: 46
- `partial -> partial -> sufficient -> sufficient -> sufficient`: 30
- `partial -> partial -> partial -> sufficient -> sufficient`: 27
- `sufficient -> sufficient -> partial -> sufficient -> sufficient`: 23
- `partial -> partial -> partial -> partial -> sufficient`: 19
- `sufficient -> sufficient -> partial -> partial -> sufficient`: 17
- `sufficient -> sufficient -> partial -> partial -> partial`: 7
- `partial -> sufficient -> partial -> sufficient -> sufficient`: 7

## test

| Stage | Insufficient | Partial | Sufficient |
|---|---:|---:|---:|
| dense@5 | 14 | 207 | 779 |
| dense@10 | 8 | 159 | 833 |
| hybrid@5 | 5 | 197 | 798 |
| hybrid@10 | 4 | 133 | 863 |
| rerank@20 | 2 | 83 | 915 |

Non-monotonic trajectories: 72 / 1000

Most common trajectories:

- `sufficient -> sufficient -> sufficient -> sufficient -> sufficient`: 726
- `partial -> partial -> partial -> partial -> partial`: 65
- `partial -> sufficient -> sufficient -> sufficient -> sufficient`: 37
- `partial -> partial -> sufficient -> sufficient -> sufficient`: 33
- `sufficient -> sufficient -> partial -> sufficient -> sufficient`: 30
- `partial -> partial -> partial -> partial -> sufficient`: 29
- `partial -> partial -> partial -> sufficient -> sufficient`: 24
- `sufficient -> sufficient -> partial -> partial -> sufficient`: 16
- `partial -> sufficient -> partial -> sufficient -> sufficient`: 7
- `sufficient -> sufficient -> partial -> partial -> partial`: 6

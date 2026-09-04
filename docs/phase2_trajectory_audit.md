# Phase 2 Evidence Trajectory Audit

The Controller ladder is evaluated under both raw-stage evidence and cumulative evidence memory. Cumulative coverage is asserted to be non-decreasing during data construction.

## train

| Evidence definition | Questions | Non-monotonic | Ratio |
|---|---:|---:|---:|
| raw | 5404 | 138 | 0.025537 |
| cumulative | 5404 | 0 | 0.000000 |

| Mode | Stage | Insufficient | Partial | Sufficient | Continue | Stop |
|---|---|---:|---:|---:|---:|---:|
| raw | dense@5 | 34 | 1094 | 4276 | 1128 | 4276 |
| raw | hybrid@10 | 12 | 674 | 4718 | 686 | 4718 |
| raw | rerank@20 | 1 | 484 | 4919 | 485 | 4919 |
| cumulative | dense@5 | 34 | 1094 | 4276 | 1128 | 4276 |
| cumulative | hybrid@10 | 8 | 535 | 4861 | 543 | 4861 |
| cumulative | rerank@20 | 1 | 432 | 4971 | 433 | 4971 |

## dev

| Evidence definition | Questions | Non-monotonic | Ratio |
|---|---:|---:|---:|
| raw | 1000 | 22 | 0.022000 |
| cumulative | 1000 | 0 | 0.000000 |

| Mode | Stage | Insufficient | Partial | Sufficient | Continue | Stop |
|---|---|---:|---:|---:|---:|---:|
| raw | dense@5 | 8 | 173 | 819 | 181 | 819 |
| raw | hybrid@10 | 2 | 111 | 887 | 113 | 887 |
| raw | rerank@20 | 2 | 73 | 925 | 75 | 925 |
| cumulative | dense@5 | 8 | 173 | 819 | 181 | 819 |
| cumulative | hybrid@10 | 2 | 88 | 910 | 90 | 910 |
| cumulative | rerank@20 | 2 | 68 | 930 | 70 | 930 |

## test

| Evidence definition | Questions | Non-monotonic | Ratio |
|---|---:|---:|---:|
| raw | 1000 | 23 | 0.023000 |
| cumulative | 1000 | 0 | 0.000000 |

| Mode | Stage | Insufficient | Partial | Sufficient | Continue | Stop |
|---|---|---:|---:|---:|---:|---:|
| raw | dense@5 | 9 | 191 | 800 | 200 | 800 |
| raw | hybrid@10 | 2 | 121 | 877 | 123 | 877 |
| raw | rerank@20 | 0 | 88 | 912 | 88 | 912 |
| cumulative | dense@5 | 9 | 191 | 800 | 200 | 800 |
| cumulative | hybrid@10 | 1 | 99 | 900 | 100 | 900 |
| cumulative | rerank@20 | 0 | 78 | 922 | 78 | 922 |

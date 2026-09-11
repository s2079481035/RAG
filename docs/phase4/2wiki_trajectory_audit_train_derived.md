# Phase 4 Evidence Trajectory Audit

The Controller ladder is evaluated under both raw-stage evidence and cumulative evidence memory. Cumulative coverage is asserted to be non-decreasing during data construction.

## train_core

| Evidence definition | Questions | Non-monotonic | Ratio |
|---|---:|---:|---:|
| raw | 163454 | 4731 | 0.028944 |
| cumulative | 163454 | 0 | 0.000000 |

| Mode | Stage | Insufficient | Partial | Sufficient | Continue | Stop |
|---|---|---:|---:|---:|---:|---:|
| raw | dense@5 | 500 | 108902 | 54052 | 109402 | 54052 |
| raw | hybrid@10 | 922 | 107210 | 55322 | 108132 | 55322 |
| raw | rerank@20 | 187 | 103436 | 59831 | 103623 | 59831 |
| cumulative | dense@5 | 500 | 108902 | 54052 | 109402 | 54052 |
| cumulative | hybrid@10 | 85 | 104775 | 58594 | 104860 | 58594 |
| cumulative | rerank@20 | 53 | 102633 | 60768 | 102686 | 60768 |

## dev_calibration

| Evidence definition | Questions | Non-monotonic | Ratio |
|---|---:|---:|---:|
| raw | 2000 | 38 | 0.019000 |
| cumulative | 2000 | 0 | 0.000000 |

| Mode | Stage | Insufficient | Partial | Sufficient | Continue | Stop |
|---|---|---:|---:|---:|---:|---:|
| raw | dense@5 | 6 | 1316 | 678 | 1322 | 678 |
| raw | hybrid@10 | 5 | 1269 | 726 | 1274 | 726 |
| raw | rerank@20 | 2 | 1231 | 767 | 1233 | 767 |
| cumulative | dense@5 | 6 | 1316 | 678 | 1322 | 678 |
| cumulative | hybrid@10 | 0 | 1247 | 753 | 1247 | 753 |
| cumulative | rerank@20 | 0 | 1221 | 779 | 1221 | 779 |

## dev_policy

| Evidence definition | Questions | Non-monotonic | Ratio |
|---|---:|---:|---:|
| raw | 2000 | 67 | 0.033500 |
| cumulative | 2000 | 0 | 0.000000 |

| Mode | Stage | Insufficient | Partial | Sufficient | Continue | Stop |
|---|---|---:|---:|---:|---:|---:|
| raw | dense@5 | 7 | 1349 | 644 | 1356 | 644 |
| raw | hybrid@10 | 16 | 1323 | 661 | 1339 | 661 |
| raw | rerank@20 | 4 | 1276 | 720 | 1280 | 720 |
| cumulative | dense@5 | 7 | 1349 | 644 | 1356 | 644 |
| cumulative | hybrid@10 | 0 | 1299 | 701 | 1299 | 701 |
| cumulative | rerank@20 | 0 | 1268 | 732 | 1268 | 732 |

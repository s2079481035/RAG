# Phase 4 Evidence Trajectory Audit

The Controller ladder is evaluated under both raw-stage evidence and cumulative evidence memory. Cumulative coverage is asserted to be non-decreasing during data construction.

## heldout

| Evidence definition | Questions | Non-monotonic | Ratio |
|---|---:|---:|---:|
| raw | 12576 | 525 | 0.041746 |
| cumulative | 12576 | 0 | 0.000000 |

| Mode | Stage | Insufficient | Partial | Sufficient | Continue | Stop |
|---|---|---:|---:|---:|---:|---:|
| raw | dense@5 | 49 | 8533 | 3994 | 8582 | 3994 |
| raw | hybrid@10 | 138 | 8212 | 4226 | 8350 | 4226 |
| raw | rerank@20 | 27 | 7637 | 4912 | 7664 | 4912 |
| cumulative | dense@5 | 49 | 8533 | 3994 | 8582 | 3994 |
| cumulative | hybrid@10 | 11 | 7974 | 4591 | 7985 | 4591 |
| cumulative | rerank@20 | 5 | 7568 | 5003 | 7573 | 5003 |

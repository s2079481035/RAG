# Phase 2 Analysis

Main Controller, packing, length, and supporting-fact results below are read from completed `test` prediction files. Counterfactual diagnostics are Dev-only. Thresholds were selected on Dev and were not refit on Test.

## RQ1: Evidence beyond stage shortcuts

- Best Query+Evidence Stop F1=0.9553 vs Query+Stage=0.9290 (delta=+0.0263); this descriptively supports added evidence value. No significance test is implied.
- Best Query+Evidence Stop F1=0.9553 vs Query+Stage+Stats=0.9319 (delta=+0.0234); this descriptively supports added evidence value. No significance test is implied.

## RQ2: 512-token truncation

Concat-Truncate overall False Stop Rate=0.5000. Length-bucket results are reported in `results/phase2/length_bucket_analysis.csv`; they establish association, not causal attribution.
- `<=512`: n=141, Stop F1=0.9562, FSR=0.4118.
- `513-1024`: n=884, Stop F1=0.9362, FSR=0.3017.
- `1025-2048`: n=975, Stop F1=0.9515, FSR=0.6421.
- `>2048`: n=1000, Stop F1=0.9552, FSR=0.7701.

## RQ3: Chunk-level label distribution

- raw: dense@5: I=9, P=191, S=800; hybrid@10: I=2, P=121, S=877; rerank@20: I=0, P=88, S=912.
- cumulative: dense@5: I=9, P=191, S=800; hybrid@10: I=1, P=99, S=900; rerank@20: I=0, P=78, S=922.

## RQ4: Representation

Best non-concat representation is `score_aware_packing`. Its FSR=0.4841 vs Concat=0.5000 (delta=-0.0159); Partial-to-Stop false counts are in the packing table.

## RQ5: Cumulative evidence

- raw: 23/1000 non-monotonic trajectories (0.023000).
- cumulative: 0/1000 non-monotonic trajectories (0.000000).

## RQ6: Number of gold facts

- `2` facts: n=2109, SF Recall=0.9327, complete=0.8687, Stop F1=0.9541, FSR=0.4621.
- `3` facts: n=684, SF Recall=0.9342, complete=0.8713, Stop F1=0.9511, FSR=0.5455.
- `4+` facts: n=207, SF Recall=0.9713, complete=0.9372, Stop F1=0.9797, FSR=0.5385.

## Go / No-Go

The tables support only descriptive Go/No-Go decisions. Confirm practical effect sizes across Query+Stage, Query+Stage+Stats, counterfactual swap, long-evidence buckets, and manual label audit before choosing a Phase 3 method. Risk calibration remains intentionally out of scope.

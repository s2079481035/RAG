# Query-only vs Query + Evidence Critic

Test split is evaluation-only. Both critics use the same data split, backbone, seed, optimizer, training steps, and sufficient threshold.

| Critic | Accuracy | Macro F1 | Sufficient P | Sufficient R | Sufficient F1 | False Stop Rate | Sufficient AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| query_only | 0.8246 | 0.4241 | 0.8771 | 0.9236 | 0.8997 | 0.6675 | 0.7786 |
| query_evidence | 0.8782 | 0.7257 | 0.9663 | 0.8914 | 0.9273 | 0.1601 | 0.9441 |

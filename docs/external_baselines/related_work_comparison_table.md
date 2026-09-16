# Related-Work Comparison Matrix

This is a method-level comparison. It is not a unified fair numerical benchmark.

| Method | Decision Timing | Input | Uses Current Evidence? | Uses Candidate Answer? | Decision Target | Continuation Action | Generates New Query? | Training Supervision | Risk / Calibration | Explicit Retrieval Cost? | Datasets | Main Limitation for This Study |
|---|---|---|---:|---:|---|---|---:|---|---|---:|---|---|
| Adaptive-RAG | Before retrieval | Question | No | No | Choose retrieval strategy/complexity | Execute routed strategy | Method-dependent | Strategy labels / routing supervision | No matched empirical FSR control | Indirect | Multi-hop QA benchmarks | Cannot detect evidence sufficiency after seeing retrieved content |
| S2G-RAG | After each retrieval turn | Question + accumulated selected evidence | Yes | No | Sufficiency plus missing-information gaps | Retrieve using a gap-derived query | Yes | Teacher process labels distilled into a LoRA judge | No matched frozen risk target in audited pipeline | Retrieval budget/turns | HotpotQA, 2Wiki, TriviaQA | Official trained judge adapter is not distributed; different retrieval/generation stack |
| Stop-RAG | At each trajectory prefix | Question + trajectory/document state | Yes | Indirectly through reward generation | Relative value of STOP versus CONTINUE | Run another iterative retrieval step | Yes | Offline trajectories, repeated answer-quality estimates, TD(lambda) value targets | Checkpoint/threshold selected on dev; not sufficiency FSR | Yes, through future utility | HotpotQA, 2Wiki, MuSiQue | Requires expensive trajectory/value training and is not aligned with the frozen three-stage ladder |
| SURE-RAG | After an answer candidate exists | Question + candidate answer + evidence | Yes | Yes | Supported / Refuted / Insufficient | Abstain rather than prescribe retrieval | No | Claim-evidence verification labels and aggregate classifier | Selective risk-coverage calibration | No retrieval-action cost model | Controlled RAG verification benchmarks | Answer verification is a different decision problem from retrieval stopping |
| Ours | After Dense@5 and Hybrid@10 | Question + packed cumulative evidence | Yes | No | Current evidence sufficiency / stop probability | Escalate through a fixed stage ladder | No | Gold supporting-fact sufficiency, coverage auxiliary supervision, hard-partial analysis | Conservative dev threshold; frozen FSR and risk-efficiency evaluation | Yes, measured chunks/reranker/latency | HotpotQA, 2Wiki | Current sufficiency does not estimate whether later retrieval can actually rescue an incomplete state |

## Positioning Summary

- Relative to Adaptive-RAG, Ours makes a post-retrieval evidence-aware decision.
- Relative to S2G-RAG, Ours emphasizes partial-versus-sufficient discrimination and empirical risk/cost analysis; S2G-RAG emphasizes gap diagnosis and acquisition of missing evidence.
- Relative to Stop-RAG, Ours predicts present sufficiency; Stop-RAG predicts future continuation utility.
- Relative to SURE-RAG, Ours controls retrieval; SURE-RAG verifies a candidate answer and controls answering/abstention.

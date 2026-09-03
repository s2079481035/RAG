# Phase 2 Chunking Audit

Tokenizer: `BAAI/bge-large-en-v1.5` without special tokens in the chunk budget.
Sentence-aligned greedy packing, overlap=0. A sentence is never split; a sentence exceeding the budget forms one oversized chunk.

Natural paragraph boundaries are unavailable: HotpotQA exposes each context article as a flat sentence list and does not preserve paragraph markers. The legacy whole-article unit remains the paragraph/article-scale reference and is not mislabeled as a recovered natural paragraph.

| Variant | Chunks | Mean tokens | P50 | P95 | Max | Mean chunks/article | Oversized sentences | Mean underfill | Mapping rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_128 | 16900 | 76.36 | 77 | 123 | 259 | 1.23 | 44 | 51.84 | 0.999944 |
| sentence_256 | 13891 | 92.90 | 84 | 186 | 259 | 1.01 | 1 | 163.11 | 0.999944 |

Sentence-boundary deviation is reported as unused budget for non-oversized chunks. Dev/test supporting-fact mapping must be 1.0 before retrieval experiments are allowed to run.

| Variant | Train mapping | Dev mapping | Test mapping |
|---|---:|---:|---:|
| sentence_128 | 0.999924 | 1.000000 | 1.000000 |
| sentence_256 | 0.999924 | 1.000000 | 1.000000 |

## Granularity check

| Variant | Single-chunk articles | Multi-chunk articles | Single-chunk ratio |
|---|---:|---:|---:|
| sentence_128 | 10896 | 2887 | 0.790539 |
| sentence_256 | 13676 | 107 | 0.992237 |

A high single-chunk ratio means the target budget often leaves the original short HotpotQA context article intact. This limits how strongly that variant repairs the Phase 1 article-granularity problem and must be considered together with dev retrieval results.

## Invalid source annotations

Original gold annotations are preserved and never auto-corrected. Questions with impossible train annotations are retained with `gold_annotation_valid=false` and excluded from supervised Controller training.

| Question | Split | Title | Sentence ID |
|---|---|---|---:|
| hotpot_005060 | train | Jimmy Butler (basketball) | 902 |

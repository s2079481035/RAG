# 2Wiki Shared Benchmark-Context Corpus Audit

This corpus pools context documents from official Train and labeled Dev, then deduplicates exact title-plus-text instances. It is not Full Wikipedia retrieval.

- Raw document instances: 1800300
- Unique titles: 384857
- Unique exact texts: 414664
- Deduplicated title-plus-text documents: 415892
- Exact duplicate instances removed: 1385557
- Same-title/different-text titles: 29734
- Different-title/identical-text groups: 49
- Sentence-segmentation conflicts: 1148
- Frozen sentence-aligned 256-token chunks: 446798
- Oversized single-sentence chunks: 19

No 2Wiki chunk-size sweep is performed; the HotpotQA-selected approximately 256-token setting is transferred unchanged.

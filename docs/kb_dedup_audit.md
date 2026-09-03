# Phase 2 KB Dedup Audit

Canonicalization only merges exact `document_title + joined article text` duplicates.
The corpus membership is held fixed to the Phase 1 shared gold-context pool; this is not the full HotpotQA distractor-context pool.

| Category | Groups/Titles | Instances/Variants |
|---|---:|---:|
| duplicate title + identical text | 936 | 1963 |
| same title + different text | 0 | 0 |
| different title + identical text | 0 | 0 |

- Legacy document instances: 14810
- Official source context instances scanned: 73700
- Source contexts outside frozen Phase 1 KB: 58890
- Every frozen KB instance is a gold document for at least one question: True
- Canonical articles: 13783
- Removed duplicate instances: 1027
- Maximum duplicate group size: 4
- Sentence-segmentation conflicts among exact duplicates: 0

Different titles are never merged, even if their text matches. Same-title text variants would also remain distinct. Supporting facts are deduplicated as `(title, sentence_id)` before coverage is computed.

# 2WikiMultiHopQA Dataset Audit

The input is validated against the official JSON structure. Format differences cause an explicit failure; no field is silently remapped.

## Sources and Roles

| Official split | Experiment role | Questions | Source | SHA-256 |
|---|---|---|---|---|
| Train | train_core/dev_calibration/dev_policy | 167454 | data/2wiki/source/train.json | b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa |
| Dev | heldout evaluation only | 12576 | data/2wiki/source/dev.json | 48b9bdc69654dc580fda5f935a48b88cb89f11887587310af60d406c8d0111a6 |

- Question-level split digest: `c7828fe62192601a57804a3067c277d7b77d8d852e4c8994123419ff85749e8e`
- Official unlabeled Test is not used.
- Supporting facts were observed and required as exact `[title, sentence_id]` pairs.

## Experiment Splits

| Split | Questions |
|---|---|
| dev_calibration | 2000 |
| dev_policy | 2000 |
| heldout | 12576 |
| train_core | 163454 |

## Official Train

- Raw context document instances: 1674540
- Unique document titles: 369378
- Repeated-title groups: 152196
- Repeated-title instances beyond first: 1305162
- Unique exact document texts: 396978
- Duplicate-text groups: 155140
- Duplicate-text instances beyond first: 1277562
- Contexts per question: `{"min": 10, "mean": 10, "median": 10.0, "p95": 10, "max": 10}`
- Supporting-fact counts: `{"2": 132512, "3": 297, "4": 34256, "5": 389}`
- Answer types: `{"date": 30629, "number": 36, "span": 78799, "year": 431, "yes_no": 57559}`
- Question types: `{"bridge_comparison": 34631, "comparison": 51963, "compositional": 76481, "inference": 4379}`
- Observed source key sets: `[["_id", "answer", "context", "evidences", "question", "supporting_facts", "type"]]`

## Official Dev

- Raw context document instances: 125760
- Unique document titles: 54957
- Repeated-title groups: 12274
- Repeated-title instances beyond first: 70803
- Unique exact document texts: 56680
- Duplicate-text groups: 11794
- Duplicate-text instances beyond first: 69080
- Contexts per question: `{"min": 10, "mean": 10, "median": 10.0, "p95": 10, "max": 10}`
- Supporting-fact counts: `{"2": 9805, "3": 20, "4": 2739, "5": 11, "6": 1}`
- Answer types: `{"date": 561, "number": 27, "span": 10602, "year": 91, "yes_no": 1295}`
- Question types: `{"bridge_comparison": 2751, "comparison": 3040, "compositional": 5236, "inference": 1549}`
- Observed source key sets: `[["_id", "answer", "context", "evidences", "question", "supporting_facts", "type"]]`

## Gold-to-Chunk Mapping

- Gold supporting facts: 435571
- Mapped supporting facts: 435571
- Mapping rate: 1.000000
- Missing facts: 0

Mapping requires exact `chunk.document_title == gold.title` and membership of `gold.sentence_id` in `chunk.sentence_ids`.

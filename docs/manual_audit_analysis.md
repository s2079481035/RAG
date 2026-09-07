# Manual Sufficiency Audit Analysis

This report evaluates whether automatic gold supporting-fact coverage is a reasonable proxy for human answerability. It does not modify any dev or test label.

## Audit Scope

- Completed dev audit records: 180
- Human stop label is treated as reference; the automatic coverage-derived label is treated as prediction.
- `retrieved_evidence` contains full retrieved chunks. The Controller saw packed evidence, which may be shorter.

## Agreement Metrics

| Comparison | Accuracy | Macro F1 | Sufficient precision | Sufficient recall |
|---|---:|---:|---:|---:|
| Automatic Stop vs Human Stop | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Automatic three-class vs Human three-class | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Priority Disagreements

- Automatic Sufficient -> Human Not Sufficient: 0
- Automatic Continue -> Human Sufficient: 0

| Automatic label | Human label | Count |
|---|---|---:|

## Full Evidence vs Model-visible Evidence

- `all_gold_visible`: n=22, binary agreement=1.0000
- `some_gold_not_visible`: n=158, binary agreement=1.0000
- `unknown`: n=0, binary agreement=N/A

## Human Disagreement Notes

The following notes are evidence for the researcher's qualitative synthesis; they are not generated label corrections.

| Audit ID | Automatic -> Human | Human note |
|---:|---|---|

## Interpretation Boundary

These results validate or challenge the answerability proxy on dev only. They must not be used to relabel test, tune a test threshold, or claim that the model saw every token shown in the audit sheet.

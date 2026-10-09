"""Read frozen results without changing them; keep provenance and missingness explicit."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [
        ('results/phase4/final/table_end_to_end.csv', '2Wiki', 'heldout'),
        ('results/phase4/final/table_external_baselines.csv', '2Wiki', 'heldout'),
        ('results/phase4/2wiki/adaptive_rag_dev_summary.csv', '2Wiki', 'dev_policy'),
        ('results/phase3b/test_risk_policy.csv', 'HotpotQA', 'test'),
    ]
    rows, hashes, seen = [], {}, set()
    for source, dataset, split in sources:
        path = ROOT / source
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        for original in csv.DictReader(path.open(encoding='utf-8-sig')):
            system = original.get('system', original.get('policy', ''))
            if split != 'dev_policy' and not any(t in system.lower() for t in ('heavy', 'risk_10', 'risk10', 'judge', 'adaptive')):
                continue
            key = (dataset, split, system, original.get('seed', ''), original.get('role', ''))
            if key in seen:
                continue
            seen.add(key)
            row = dict(original)
            row.update(dataset=dataset, split=split, source_type='our_existing_experiment', source_file=source,
                       comparison_group=f'{dataset}_{split}_frozen', metric_scale='fraction',
                       latency_scope='frozen component-sum estimate; not new wall-clock measurement',
                       risk_definition='NR', total_llm_calls='NR', total_llm_tokens='NR')
            if original.get('false_stop_rate'):
                row['risk_definition'] = 'false stops / actual continue labels at reached actionable decisions; excludes forced terminal'
            if original.get('average_llm_total_tokens') and original.get('seed') != 'std':
                total = float(original['average_llm_total_tokens'])
                if 'judge' in system.lower():
                    total += float(original['judge_input_tokens_per_question']) + float(original['judge_output_tokens_per_question'])
                    row['total_llm_calls'] = 1 + float(original['average_controller_calls'])
                else:
                    row['total_llm_calls'] = 1
                row['total_llm_tokens'] = total
            rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / 'unified_baseline_comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, '') or 'NR' for k in fields} for row in rows)
    (OUT / 'existing_result_provenance.json').write_text(json.dumps({'source_sha256': hashes, 'rows': len(rows),
        'note': 'Original numeric strings retained. Judge total tokens add separately stored Judge tokens. Dev and heldout are separate groups; std rows do not sum marginal standard deviations.'}, indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} rows from {len(hashes)} frozen files')


if __name__ == '__main__':
    main()

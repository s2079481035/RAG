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

    smoke_specs = [
        ('ircot_inspired_dev32_v2', 'IRCoT-inspired'),
        ('sim_rag_dev32', 'SIM-RAG adapted'),
    ]
    smoke_rows, smoke_hashes, shared_ids = [], {}, None
    for directory, label in smoke_specs:
        result_dir = OUT / directory
        manifest_path = result_dir / 'manifest.json'
        predictions_path = result_dir / 'predictions.jsonl'
        summary_path = result_dir / 'summary.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        predictions = [json.loads(line) for line in predictions_path.open(encoding='utf-8')]
        ids = [row['question_id'] for row in predictions]
        if ids != manifest['ids'] or len(ids) != len(set(ids)) or len(ids) != summary['questions']:
            raise ValueError(f'Invalid or reordered smoke result IDs: {directory}')
        if shared_ids is None:
            shared_ids = ids
        elif ids != shared_ids:
            raise ValueError('Adapted smoke methods must use the identical frozen Dev32 IDs')
        source_paths = [manifest_path, predictions_path, summary_path]
        checkpoint_path = result_dir / 'checkpoint_provenance.json'
        if checkpoint_path.exists():
            source_paths.append(checkpoint_path)
        for path in source_paths:
            smoke_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        verdicts = {}
        if label == 'SIM-RAG adapted':
            for row in predictions:
                for turn in row['trace']:
                    key = turn['critic_raw'] or 'blank_reject_prefix'
                    verdicts[key] = verdicts.get(key, 0) + 1
        smoke_rows.append({
            'method': label,
            'dataset': '2Wiki',
            'split': manifest['split'],
            'sample_count': len(ids),
            'source_type': manifest['source_type'],
            'protocol_label': 'ADAPTED DEV32 SMOKE; SAME IDS; NOT ORIGINAL-PAPER REPRODUCTION',
            'answer_em': summary['answer_em'],
            'answer_f1': summary['answer_f1'],
            'supporting_fact_recall': summary['supporting_fact_recall'],
            'complete_evidence_coverage': summary['complete_evidence_coverage'],
            'avg_retrieval_rounds': summary['retrieval_rounds'],
            'avg_retrieved_chunks': summary['retrieved_chunks'],
            'avg_reasoner_calls': summary.get('reasoner_calls', 'NR'),
            'avg_critic_calls': summary.get('critic_calls', 'NR'),
            'avg_total_llm_calls': summary['total_llm_calls'],
            'avg_total_tokens': summary['total_tokens'],
            'avg_latency_ms_diagnostic_only': summary['total_latency_ms'],
            'changed_query_rounds': summary.get('changed_query_rounds', sum(r['retrieval_rounds'] for r in predictions)),
            'questions_with_retrieval': sum(r['retrieval_rounds'] > 0 for r in predictions),
            'stop_reason_counts': json.dumps(summary['stop_reason_counts'], sort_keys=True),
            'critic_verdict_counts': json.dumps(verdicts, sort_keys=True) if verdicts else 'NR',
            'source_directory': result_dir.relative_to(ROOT).as_posix(),
            'source_git_commit': manifest['git_commit'],
            'comparison_limitations': summary['latency_limitation'],
        })
    smoke_fields = list(smoke_rows[0])
    with (OUT / 'adapted_smoke_comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=smoke_fields)
        writer.writeheader()
        writer.writerows(smoke_rows)
    (OUT / 'adapted_smoke_provenance.json').write_text(json.dumps({
        'source_sha256': smoke_hashes,
        'rows': len(smoke_rows),
        'shared_dev32_ids_sha256': hashlib.sha256('\n'.join(shared_ids).encode()).hexdigest(),
        'note': 'Same frozen Dev32 IDs and corpus, but adapted method implementations. Latency is diagnostic and these rows must not be merged with heldout or original-paper results.',
    }, indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} frozen rows and {len(smoke_rows)} adapted smoke rows')


if __name__ == '__main__':
    main()

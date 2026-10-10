"""Transcribe explicitly located original-paper cells; never impute missing metrics."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    fields = ['method','paper_title','year','venue','dataset','task','retriever','reranker','generator',
              'decision_model','corpus','corpus_size','evaluation_split','sample_count','max_rounds','top_k',
              'answer_em','answer_f1','answer_metric_scale','verification_macro_f1','verification_macro_f1_std',
              'evidence_recall','evidence_recall_definition','complete_coverage','retrieval_calls','tokens',
              'latency','cost','risk_metric','risk_value','risk_definition','ece','training_cost',
              'trained_inference_release','source_type','table','pdf_page','source_url','protocol_label']
    rows = []
    def add(method, title, year, venue, url, table, page, dataset, em='NR', f1='NR', **kw):
        row = dict.fromkeys(fields, 'NR')
        row.update(method=method,paper_title=title,year=year,venue=venue,source_url=url,table=table,pdf_page=page,
                   dataset=dataset,answer_em=em,answer_f1=f1,answer_metric_scale='percent',task='open-domain QA',
                   source_type='original_paper',protocol_label='LITERATURE-REPORTED, NON-COMPARABLE PROTOCOLS')
        row.update(kw)
        rows.append(row)
    for dataset, em, f1, steps, relative in [('HotpotQA',42,53.82,3.55,5.99),('2WikiMultiHopQA',40.60,49.75,2.63,4.68)]:
        add('Adaptive-RAG','Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity',2024,'NAACL',
            'https://aclanthology.org/2024.naacl-long.389.pdf','Table 2; setup section 4.4',7,dataset,em,f1,
            retriever='BM25',generator='FLAN-T5-XL',decision_model='T5-Large',sample_count=500,
            corpus='IRCoT preprocessed dataset-specific corpus',evaluation_split='paper test subset following IRCoT',
            cost=f'{steps} retrieval-generation steps; {relative} relative time (single-step=1)',
            training_cost='silver outcome labels + dataset bias; T5-Large best validation checkpoint within 100 training iterations; GPU hours NR',
            trained_inference_release='PARTIAL: code/data/predictions released; README requires training T5-Large classifier; trained classifier checkpoint not identified')
    for dataset, em, f1, size in [('HotpotQA',49.3,60.7,5233329),('2WikiMultiHopQA',57.7,68.0,430225)]:
        add('IRCoT','Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions',2023,'ACL',
            'https://aclanthology.org/2023.acl-long.557.pdf','Table 3; Appendix A; sections 3-4',14,dataset,em,f1,
            retriever='Elasticsearch BM25',generator='GPT3 code-davinci-002',decision_model='CoT answer marker/horizon',
            corpus='Hotpot Wikipedia / 2Wiki contexts pooled across splits',corpus_size=size,
            evaluation_split='500 sampled official dev; disjoint 100 tuning',sample_count=500,max_rounds=8,
            top_k='dev-selected from 2/4/6/8; max 15 accumulated paragraphs',training_cost='20 manually written CoTs; 3 demonstration sets; no finetuning')
        rows[-1]['trained_inference_release']='code released; no task-trained checkpoint needed for prompting method; proprietary GPT3 or local FLAN-T5 base model still required'
    for generator, values in [('Llama3-8B',[(32.7,43.3),(34.1,40.2)]),('GPT-4',[(39.8,52.2),(46.1,54.6)])]:
        for dataset,(em,f1) in zip(['HotpotQA','2WikiMultiHopQA'],values):
            add('SIM-RAG full',"Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing",2025,'SIGIR',
                'https://arxiv.org/pdf/2505.02811v2','Table 1; section 4.2',7,dataset,em,f1,retriever='BM25',generator=generator,
                decision_model='Flan-T5-2.85B',corpus='dataset Wikimedia dumps',training_cost='self-practice trajectories + critic fine-tuning; pipeline 2x3090; GPU hours NR',
                trained_inference_release='yes; author-linked checkpoints; general-purpose Critic runtime verified in adapted Dev32 smoke; original Table 1 protocol not reproduced')
    for retriever, values in [('BM25',[(43.3,56.5),(41.7,48.6)]),('E5-base-v2',[(42.0,53.5),(39.0,45.3)])]:
        for dataset,(em,f1) in zip(['HotpotQA','2WikiMultiHopQA'],values):
            add('S2G-RAG','S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA',2026,'ACL',
                'https://aclanthology.org/2026.acl-long.1185.pdf','Table 1; section 4.2',6,dataset,em,f1,retriever=retriever,
                generator='Llama-3-8B-Instruct',decision_model='Llama-3.2-3B-Instruct + trained LoRA',corpus='dataset Wikimedia dumps',
                evaluation_split='official dev; code caps first 1000 unique IDs (not evidence of full-dev evaluation)',max_rounds=4,top_k=6,
                training_cost='GPT-4o-mini teacher + LoRA; pipeline 2x48GB; GPU hours NR',trained_inference_release='PARTIAL: trained adapter unavailable in existing audit')
    for dataset,em,f1 in [('HotpotQA',52.4,66.1),('2WikiMultiHopQA',68.2,75.7)]:
        kw = dict(retriever='Contriever-MSMARCO',reranker='BGE',generator='Llama-3.1-8B-Instruct',decision_model='DeBERTa-v3-large dual Q heads',
                  corpus='Hotpot Wikipedia / 2Wiki pooled contexts across train-dev-test',evaluation_split='1000 official dev for test; disjoint 1000 validation',
                  sample_count=1000,max_rounds=10,top_k='10 candidates; retain 1 reranked passage',training_cost='4 H100; trajectories and 8 answers/prefix; GPU hours NR',
                  trained_inference_release='no trained value checkpoint identified in existing audit')
        if dataset == '2WikiMultiHopQA':
            kw.update(evidence_recall=82.6,evidence_recall_definition='document retrieval recall percent; not sentence SF recall',cost='5.1 average iterations (Table 2 p8)')
        add('Stop-RAG (authors pipeline)','Stop-RAG: Value-Based Retrieval Control for Iterative RAG',2025,'arXiv',
            'https://arxiv.org/pdf/2510.14337','Table 1; Table 2',7,dataset,em,f1,**kw)
    for method,macro,std,ece,risk in [('SURE-RAG raw',0.8951,0.0069,0.0304,0.1642),('SURE-RAG calibrated',0.9075,0.0060,0.0198,0.1670),('Concat cross-encoder',0.8888,0.0109,'NR','NR')]:
        add(method,'SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation',2026,'arXiv',
            'https://arxiv.org/pdf/2605.03534','Tables III/V/VI',6,'HotpotQA-RAG v3',task='answer-conditioned evidence verification',
            answer_metric_scale='NA',verification_macro_f1=macro,verification_macro_f1_std=std,ece=ece,
            decision_model='DeBERTa-v3-base; set aggregator except concat baseline',corpus='constructed full/partial/refuted/insufficient evidence',
            corpus_size='4026 examples / 900 question groups / 20130 claim-evidence pairs (all splits)',evaluation_split='question-group-disjoint test',
            risk_metric='Risk@30' if risk != 'NR' else 'NR',risk_value=risk,risk_definition='unsafe selected answers / selected answers at 30% selection coverage',
            trained_inference_release='NOT REPRODUCIBLE AS TRAINED MODEL FROM CURRENT RELEASE')
    for method, score in [('SURE-RAG matched',0.8951),('GPT-4o Judge',0.7284)]:
        add(method,'SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation',2026,'arXiv',
            'https://arxiv.org/pdf/2605.03534','Table VIII; section VI-G',7,'HotpotQA-RAG v3',task='answer-conditioned evidence verification',
            answer_metric_scale='NA',verification_macro_f1=score,sample_count=438,evaluation_split='matched sampled test IDs')
    out = ROOT / 'results/paper_v1/literature_metrics.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(rows)} literature rows')


if __name__ == '__main__':
    main()

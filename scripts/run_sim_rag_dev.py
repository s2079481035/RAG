"""Adapt the released SIM-RAG critic to the frozen 2Wiki Dev corpus.

Preserves answer/rationale-conditioned Accept/Reject stopping. Uses the shared
BM25 corpus and Qwen reasoner, so results are an adapted reproduction.
"""
import argparse
import hashlib
import json
import pickle
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from evidence_utils import supporting_fact_metrics
from generate_phase3a import chat_prompt
from phase2_retrieval import bm25_tokenize, deterministic_top_indices
from phase3a_generation_utils import exact_match_score, token_f1_score

ROOT = Path(__file__).resolve().parents[1]
CRITIC_INSTRUCTION = ('Instruction: Predict if the following answer to the question and context '
                      'should be accepted, 1, or rejected, 0, based on the rationale.')
ANSWER_PROMPT = ("You are a knowledgeable question-answering assistant. Based on the context provided (if any), "
                 "answer the following multihop question. Provide a brief multihop explanation. If you do not know "
                 "part of what is referenced in the question, do not try to make it up. If unsure, respond with "
                 "'unsure.' Respond only as:\nAnswer: <answer>\nRationale: <brief rationale>")
QUERY_PROMPT = ("Generate one concise Wikipedia search query that addresses the next missing fact for this "
                "multihop question. Do not repeat an earlier query or a retrieved title. Return only the query.")


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def qkey(text):
    return ' '.join(text.lower().split()).strip(' .?!"\'')


def parse_answer(text):
    answer = re.search(r'(?im)^\s*Answer\s*:\s*(.+?)\s*$', text)
    rationale = re.search(r'(?ims)^\s*Rationale\s*:\s*(.+?)\s*$', text)
    return (answer.group(1).strip() if answer else 'unsure',
            rationale.group(1).strip() if rationale else text.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reasoner', required=True)
    p.add_argument('--critic', required=True)
    p.add_argument('--n', type=int, choices=[32, 100], default=32)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reasoner-gpu-memory-gib', type=int, default=10)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    questions_path = ROOT / 'data/2wiki/questions.json'
    chunks_path = ROOT / 'data/2wiki/chunks/sentence_256.jsonl'
    index_path = ROOT / 'data/2wiki/indices/sentence_256/bm25.pkl'
    split_path = ROOT / 'data/2wiki/splits/dev_policy_ids.txt'
    for path in [questions_path, chunks_path, index_path, split_path]:
        if not path.exists():
            raise FileNotFoundError(f'Run on server with frozen data mounted: {path}')
    ids = sorted(split_path.read_text().split(), key=lambda q: hashlib.sha256(('recent_rag_v1:' + q).encode()).hexdigest())
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'results/paper_v1'):
        raise ValueError('Output must be under results/paper_v1')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(method='SIM-RAG adapted reproduction', source_type='adapted_reproduction', split='dev_policy',
        ids=ids[:args.n], pre_frozen_100_ids=ids[:100], n=args.n, git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), reasoner=args.reasoner, critic=args.critic,
        max_retrieval_rounds=4, top_k=2, max_chunks=8, critic_decision='decoded output exactly equals 1',
        critic_input='question + accumulated query/doc context + candidate answer + rationale',
        adaptations=['Qwen2.5-7B-Instruct reasoner/query generator', 'shared frozen BM25 sentence-chunk corpus',
                     'official public general-purpose critic', 'official answer/rationale-conditioned binary decision',
                     'zero-shot concise query prompt; official few-shot 2Wiki query prompt omitted'],
        source_sha256={str(x.relative_to(ROOT)):sha(x) for x in [questions_path,chunks_path,index_path,split_path]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if args.preflight:
        print('PREFLIGHT_OK; no model inference',flush=True); return
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    torch.manual_seed(42); torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for reasoner')
    free,_=torch.cuda.mem_get_info()
    if free < (args.reasoner_gpu_memory_gib+2)*2**30:
        raise RuntimeError('Insufficient GPU headroom; leave other jobs untouched')
    rtok=AutoTokenizer.from_pretrained(args.reasoner,local_files_only=True)
    reasoner=AutoModelForCausalLM.from_pretrained(args.reasoner,local_files_only=True,torch_dtype=torch.float16,
        device_map='auto',max_memory={0:f'{args.reasoner_gpu_memory_gib}GiB','cpu':'48GiB'}).eval()
    ctok=AutoTokenizer.from_pretrained(args.critic)
    critic=AutoModelForSeq2SeqLM.from_pretrained(args.critic,torch_dtype=torch.float32,device_map='cpu').eval()
    questions=json.loads(questions_path.read_text())
    chunks={r['chunk_id']:r for r in (json.loads(line) for line in chunks_path.open())}
    with index_path.open('rb') as f: payload=pickle.load(f)  # trusted frozen project index
    bm25,doc_ids=payload['bm25'],payload['doc_ids']
    records=[]
    for qid in ids[:args.n]:
        q=questions[qid]; start=time.perf_counter(); selected=[]; task=f"Question: {q['question']}\nContext:\n"
        trace=[]; reasoner_calls=critic_calls=reasoner_in=reasoner_out=critic_in=critic_out=0
        queries=[]; stop_reason='forced_horizon'
        def rgen(system,content,limit):
            nonlocal reasoner_calls,reasoner_in,reasoner_out
            prompt=chat_prompt(rtok,system+'\n\n'+content)
            enc=rtok(prompt,return_tensors='pt',add_special_tokens=False).to(reasoner.get_input_embeddings().weight.device)
            if enc.input_ids.shape[1]>4096: raise ValueError('Reasoner prompt exceeds 4096 tokens')
            with torch.inference_mode(): outids=reasoner.generate(**enc,max_new_tokens=limit,do_sample=False,num_beams=1,pad_token_id=rtok.eos_token_id)
            reasoner_calls+=1; reasoner_in+=enc.input_ids.shape[1]; reasoner_out+=outids.shape[1]-enc.input_ids.shape[1]
            return rtok.decode(outids[0,enc.input_ids.shape[1]:],skip_special_tokens=True).strip()
        for turn in range(5):
            raw=rgen(ANSWER_PROMPT,task,128); answer,rationale=parse_answer(raw)
            gate=f"{CRITIC_INSTRUCTION}\n{task} \nAnswer: {answer}\nRationale: {rationale}"
            cenc=ctok(gate,return_tensors='pt',truncation=True)
            with torch.inference_mode(): cids=critic.generate(**cenc,max_new_tokens=1)
            verdict_text=ctok.batch_decode(cids,skip_special_tokens=True)[0].strip()
            critic_calls+=1; critic_in+=cenc.input_ids.shape[1]; critic_out+=cids.shape[1]-1
            accepted=verdict_text=='1'
            entry=dict(turn=turn,answer=answer,rationale=rationale,critic_raw=verdict_text,accepted=accepted)
            trace.append(entry)
            if accepted: stop_reason='critic_accept'; break
            if turn==4: break
            query=rgen(QUERY_PROMPT,task+'\nCandidate answer: '+answer+'\nRationale: '+rationale,48).splitlines()[0].strip()
            if not query or qkey(query) in {qkey(x) for x in queries}:
                stop_reason='invalid_query'; entry['invalid_query']=query; break
            queries.append(query)
            found=[doc_ids[int(i)] for i in deterministic_top_indices(bm25.get_scores(bm25_tokenize(query)),2)]
            selected=list(dict.fromkeys(selected+found))[:8]
            entry.update(query=query,retrieved_ids=found)
            docs='\n'.join(f"Title: {chunks[c]['document_title']} Content: {chunks[c]['chunk_text']}" for c in found)
            task+=f"Query: {query}\nRetrieved Document: {docs}\n"
        torch.cuda.synchronize()
        metrics=supporting_fact_metrics(selected,chunks,q['gold_supporting_facts'])
        row=dict(question_id=qid,answer=answer,answer_em=exact_match_score(answer,q['answer']),
            answer_f1=token_f1_score(answer,q['answer']),supporting_fact_recall=metrics['supporting_fact_recall'],
            complete_evidence_coverage=metrics['complete_evidence_coverage'],retrieval_rounds=len(queries),
            retrieved_chunks=len(selected),reasoner_calls=reasoner_calls,critic_calls=critic_calls,
            total_llm_calls=reasoner_calls+critic_calls,reasoner_input_tokens=reasoner_in,
            reasoner_output_tokens=reasoner_out,critic_input_tokens=critic_in,critic_output_tokens=critic_out,
            total_tokens=reasoner_in+reasoner_out+critic_in+critic_out,total_latency_ms=(time.perf_counter()-start)*1000,
            stop_reason=stop_reason,trace=trace)
        records.append(row)
        with (out/'predictions.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        print(f'{len(records)}/{args.n} completed',flush=True)
    means={k:sum(r[k] for r in records)/len(records) for k in ['answer_em','answer_f1','supporting_fact_recall',
        'complete_evidence_coverage','retrieval_rounds','retrieved_chunks','reasoner_calls','critic_calls','total_llm_calls','total_tokens','total_latency_ms']}
    means.update(status='SMOKE_COMPLETE_NOT_FORMAL_COMPARISON',questions=len(records),
        stop_reason_counts=dict(Counter(r['stop_reason'] for r in records)),
        latency_limitation='shared GPU, CPU critic, and possible reasoner CPU offload; diagnostic only')
    (out/'summary.json').write_text(json.dumps(means,indent=2),encoding='utf-8')


if __name__=='__main__': main()

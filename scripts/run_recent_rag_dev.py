"""Dev-only IRCoT-inspired smoke runner on the existing 2Wiki shared BM25 index.

No training and no writes to frozen artifacts. Run only on the experiment server.
"""
import argparse
import hashlib
import json
import pickle
import re
import subprocess
import time
from pathlib import Path

from evidence_utils import supporting_fact_metrics
from generate_phase3a import chat_prompt, fit_prompt
from phase2_retrieval import bm25_tokenize, deterministic_top_indices
from phase3a_generation_utils import exact_match_score, extract_short_answer, token_f1_score

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def first_sentence(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())[0].strip()


def query_key(text):
    return ' '.join(text.lower().split()).strip(' .?!')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--n', type=int, choices=[32, 100], default=32)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-memory-gib', type=int, default=12)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    if not (ROOT / 'data/2wiki/questions.json').exists():
        raise FileNotFoundError('Run on server with the frozen data mounted')
    questions_path = ROOT / 'data/2wiki/questions.json'
    chunks_path = ROOT / 'data/2wiki/chunks/sentence_256.jsonl'
    index_path = ROOT / 'data/2wiki/indices/sentence_256/bm25.pkl'
    split_path = ROOT / 'data/2wiki/splits/dev_policy_ids.txt'
    config_path = ROOT / 'configs/phase3a/generation.json'
    ids = sorted(split_path.read_text().split(), key=lambda q: hashlib.sha256(('recent_rag_v1:' + q).encode()).hexdigest())
    manifest = dict(method='IRCoT-inspired Iterative Retrieval', source_type='adapted_reproduction',
        split='dev_policy', ids=ids[:args.n], pre_frozen_100_ids=ids[:100],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        model=args.model, n=args.n, top_k=2, max_retrieval_rounds=8, max_chunks=15,
        adaptations=['Qwen2.5-7B-Instruct', 'frozen shared sentence chunks and BM25 tokenizer',
                     'zero-shot one-sentence reasoning', 'frozen final-answer prompt/evaluator',
                     'stop on answer-is marker or horizon; fail on empty or repeated continuation query'],
        source_sha256={str(x.relative_to(ROOT)): sha(x) for x in [questions_path, chunks_path, index_path, split_path, config_path]})
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'results/paper_v1'):
        raise ValueError('Output must be under results/paper_v1')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.preflight:
        print('PREFLIGHT_OK; no model inference', flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not silently launch CPU generation')
    free, _ = torch.cuda.mem_get_info()
    if free < (args.gpu_memory_gib + 2) * 2**30:
        raise RuntimeError('Insufficient GPU headroom; leave other jobs untouched')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float16, device_map='auto',
        max_memory={0: f'{args.gpu_memory_gib}GiB', 'cpu': '48GiB'})
    model.eval()
    config = json.loads(config_path.read_text())
    questions = json.loads(questions_path.read_text())
    chunks = {r['chunk_id']: r for r in (json.loads(line) for line in chunks_path.open())}
    with index_path.open('rb') as f:
        payload = pickle.load(f)  # trusted, frozen project-generated index only
    bm25, doc_ids = payload['bm25'], payload['doc_ids']
    records = []
    for qid in ids[:args.n]:
        q = questions[qid]
        assert q['split'] == 'dev_policy'
        start = time.perf_counter()
        calls, input_tokens, output_tokens = 0, 0, 0
        def generate(prompt, limit):
            nonlocal calls, input_tokens, output_tokens
            enc = tokenizer(chat_prompt(tokenizer, prompt), return_tensors='pt', add_special_tokens=False).to(model.get_input_embeddings().weight.device)
            if enc.input_ids.shape[1] > 4096:
                raise ValueError('Reasoning prompt exceeds budget; do not silently drop question/history')
            with torch.inference_mode():
                answer = model.generate(**enc, max_new_tokens=limit, do_sample=False, num_beams=1, pad_token_id=tokenizer.eos_token_id)
            calls += 1
            input_tokens += enc.input_ids.shape[1]
            output_tokens += answer.shape[1] - enc.input_ids.shape[1]
            return tokenizer.decode(answer[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        selected, history, trace = [], [], []
        query = q['question']
        stop_reason = 'horizon'
        invalid_query = None
        for step in range(8):
            if step and query_key(query) in {query_key(t['query']) for t in trace}:
                stop_reason = 'invalid_repeated_query'
                invalid_query = query
                break
            found = [doc_ids[int(i)] for i in deterministic_top_indices(bm25.get_scores(bm25_tokenize(query)), 2)]
            selected = list(dict.fromkeys(selected + found))[:15]
            context = '\n'.join(f"TITLE: {chunks[c]['document_title']}\nTEXT: {chunks[c]['chunk_text']}" for c in selected)
            prompt = ('Use the evidence and previous reasoning to produce exactly one next reasoning sentence. '
                      'If the answer is established, write: The answer is: <answer>. Otherwise state the next factual link to investigate.\n'
                      f"Question: {q['question']}\nEvidence:\n{context}\nPrevious reasoning:\n" + '\n'.join(history))
            raw = generate(prompt, 96)
            thought = first_sentence(raw)
            if not thought:
                stop_reason = 'invalid_empty_reasoning'
                invalid_query = ''
                break
            trace.append(dict(round=step + 1, query=query, retrieved_ids=found, reasoning=thought, raw_output=raw))
            history.append(thought)
            if 'answer is:' in thought.lower():
                stop_reason = 'answer_marker'
                break
            query = thought
        prompt, visible, truncated = fit_prompt(tokenizer, config, q['question'], [chunks[c] for c in selected])
        answer = extract_short_answer(generate(prompt, config['max_new_tokens']))
        torch.cuda.synchronize()
        row = dict(question_id=qid, answer=answer, answer_em=exact_match_score(answer, q['answer']),
            answer_f1=token_f1_score(answer, q['answer']), retrieval_rounds=len(trace), retrieved_chunks=len(selected),
            total_llm_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens, total_latency_ms=(time.perf_counter()-start)*1000,
            stop_reason=stop_reason, invalid_query=invalid_query,
            valid_iterative_trace=not stop_reason.startswith('invalid_'), trace=trace, final_prompt_truncated=truncated,
            visible_ids=visible, **supporting_fact_metrics(selected, chunks, q['gold_supporting_facts']))
        records.append(row)
        with (out / 'predictions.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(f'{len(records)}/{args.n} completed', flush=True)
    keys = ['answer_em','answer_f1','supporting_fact_recall','complete_evidence_coverage','retrieval_rounds',
            'retrieved_chunks','total_llm_calls','total_tokens','total_latency_ms']
    summary = {k: sum(r[k] for r in records)/len(records) for k in keys}
    summary.update(status='SMOKE_COMPLETE_NOT_FORMAL_COMPARISON', questions=len(records),
                   changed_query_rounds=sum(len(r['trace'])-1 for r in records),
                   valid_iterative_questions=sum(r['valid_iterative_trace'] for r in records),
                   valid_iterative_rate=sum(r['valid_iterative_trace'] for r in records)/len(records),
                   stop_reason_counts={reason: sum(r['stop_reason'] == reason for r in records)
                                       for reason in sorted({r['stop_reason'] for r in records})},
                   latency_limitation='shared GPU and possible CPU offload; not comparable to frozen latency')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

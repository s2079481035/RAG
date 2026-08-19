"""
Critic data construction + full retrieval
==========================================
Stage 1 of the Learned Retrieval Sufficiency Critic (protocol: Phase 3).

- Rebuilds ALL 7405 HotpotQA questions (KB identical to data/hotpotqa.json,
  verified byte-for-byte) -> data/hotpotqa_all.json
- Runs dense / hybrid(RRF) / rerank(v2-m3) retrieval for all questions
  -> results/hotpotqa_all_retrieval.json
- Builds (query, docs, label) samples for stages dense@3 / hybrid@5 / rerank@20
  -> data/critic_samples/{train,val,test}.jsonl

Leakage: the existing 1000 test questions are untouched (same qids/gold);
they only appear in test samples. Train/val come from the other 6405 questions.

Usage (GPU1):
  CUDA_VISIBLE_DEVICES=1 python3 scripts/critic_data.py
"""

import json, logging, pickle, random, re, time
from pathlib import Path

import numpy as np
import faiss
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
INDEX_DIR = DATA / "indices"
RESULTS = ROOT / "results"
SAMPLES = DATA / "critic_samples"

HOTPOT_MAX_SCAN = 7405
SEED = 42
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RRF_K = 60
STAGES = [("dense", 3), ("hybrid", 5), ("rerank", 20)]
DOC_MAX_WORDS = 48  # per-doc truncation for critic input


def scan_hotpotqa():
    gold_docs = {}
    all_docs = {}
    questions = []
    scanned = 0
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    for item in ds:
        scanned += 1
        if scanned > HOTPOT_MAX_SCAN:
            break
        gold_titles = set(item["supporting_facts"]["title"])
        gold_ids = []
        titles = item["context"]["title"]
        sents = item["context"]["sentences"]
        for i, t in enumerate(titles):
            ss = sents[i] if i < len(sents) else []
            doc_id = f"hp_{len(all_docs):06d}"
            all_docs[doc_id] = " ".join(ss) if isinstance(ss, list) else str(ss)
            if t in gold_titles:
                gold_docs[doc_id] = all_docs[doc_id]
                gold_ids.append(doc_id)
        if not gold_ids:
            continue
        sf = item["supporting_facts"]
        questions.append({"qid": f"hotpot_{scanned:06d}", "question": item["question"],
                          "answer": item["answer"], "gold_docs": gold_ids,
                          "type": item["type"], "n_supporting": len(gold_ids),
                          "n_sent": len(sf["sent_id"])})
    return all_docs, gold_docs, questions, scanned


def build_kb(all_docs, gold_docs):
    rng = random.Random(SEED)
    extra = [d for d in all_docs if d not in gold_docs]
    rng.shuffle(extra)
    kb = dict(gold_docs)
    for d in extra:
        if len(kb) >= 10000:
            break
        kb[d] = all_docs[d]
    for d in list(gold_docs):
        if d not in kb:
            kb[d] = gold_docs[d]
    return kb


def trunc_words(text, n):
    return " ".join(text.split()[:n])


def main():
    SAMPLES.mkdir(exist_ok=True)

    # ---- 1. rebuild all questions, verify KB + test identical ----
    all_docs, gold_docs, all_questions, scanned = scan_hotpotqa()
    rng = random.Random(SEED)
    extra = [d for d in all_docs if d not in gold_docs]
    rng.shuffle(extra)
    kb = dict(gold_docs)
    for d in extra:
        if len(kb) >= 10000:
            break
        kb[d] = all_docs[d]
    for d in list(gold_docs):
        if d not in kb:
            kb[d] = gold_docs[d]
    questions = [q for q in all_questions if all(g in kb for g in q["gold_docs"])]
    rng.shuffle(questions)
    all_questions = questions
    test_qs = all_questions[:1000]
    rest = all_questions[1000:]
    old = json.load(open(DATA / "hotpotqa.json", encoding="utf-8"))
    assert list(kb.keys()) == list(old["kb"].keys()) and all(kb[k] == old["kb"][k] for k in kb), "KB mismatch!"
    logger.info(f"KB identical to existing ({len(kb)} docs)")
    old_test = {q["qid"]: q for q in old["questions"]}
    core = lambda q: (q["qid"], q["question"], q["answer"], tuple(q["gold_docs"]))
    assert all(core(q) == core(old_test[q["qid"]]) for q in test_qs), "test set mismatch!"
    logger.info("test 1000 byte-identical (qid/question/answer/gold)")
    json.dump({"name": "HotpotQA", "kb": kb, "questions": all_questions},
              open(DATA / "hotpotqa_all.json", "w", encoding="utf-8"), ensure_ascii=False)
    train_qs, val_qs = rest[:5405], rest[5405:6405]
    logger.info(f"split: test={len(test_qs)} train={len(train_qs)} val={len(val_qs)}")

    # ---- 2. retrieval for all questions ----
    index = faiss.read_index(str(INDEX_DIR / "hotpotqa_dense.faiss"))
    doc_ids = np.load(INDEX_DIR / "hotpotqa_doc_ids.npy", allow_pickle=True).tolist()
    with open(INDEX_DIR / "hotpotqa_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)["bm25"]
    q_model = SentenceTransformer(DENSE_MODEL)
    qs = [q["question"] for q in all_questions]
    q_embs = q_model.encode(qs, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    reranker = None

    dense_lists, hybrid_lists, rerank_lists = {}, {}, {}
    for qi, emb in enumerate(q_embs):
        qid = all_questions[qi]["qid"]
        scores, idxs = index.search(np.array([emb], dtype="float32"), k=20)
        dense_lists[qid] = [doc_ids[i] for i in idxs[0]]
        scores_full, idxs_full = index.search(np.array([emb], dtype="float32"), k=len(doc_ids))
        bm = bm25.get_scores(re.findall(r"[a-z0-9][a-z0-9\-']*", qs[qi].lower()))
        n = len(doc_ids)
        rrf = np.zeros(n)
        for rank, i in enumerate(idxs_full[0][:200]):
            rrf[i] += 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(np.argsort(-bm)[:200]):
            rrf[i] += 1.0 / (RRF_K + rank + 1)
        top = np.argsort(-rrf)[:20]
        hybrid_lists[qid] = [doc_ids[i] for i in top]
        if qi % 500 == 0:
            logger.info(f"retrieve {qi}/{len(q_embs)}")

    if reranker is None:
        reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    for qi, qid in enumerate(hybrid_lists):
        pairs = [(qs[qi], kb[d]) for d in hybrid_lists[qid]]
        sc = reranker.predict(pairs)
        order = np.argsort(-np.array(sc))
        rerank_lists[qid] = [hybrid_lists[qid][i] for i in order]
        if qi % 500 == 0:
            logger.info(f"rerank {qi}/{len(hybrid_lists)}")

    json.dump({"dense": dense_lists, "hybrid": hybrid_lists, "rerank": rerank_lists},
              open(RESULTS / "hotpotqa_all_retrieval.json", "w"), ensure_ascii=False)
    logger.info("retrieval saved -> results/hotpotqa_all_retrieval.json")

    # ---- 3. build samples ----
    def emit(questions, split, f):
        n_suff = {s: 0 for _, s in STAGES}
        for q in questions:
            gold = set(q["gold_docs"])
            for phase, k in STAGES:
                lst = {"dense": dense_lists, "hybrid": hybrid_lists, "rerank": rerank_lists}[phase][q["qid"]][:k]
                label = 1 if set(lst) >= gold else 0
                n_suff[k] += label
                docs = " ".join(trunc_words(kb[d], DOC_MAX_WORDS) for d in lst)
                f.write(json.dumps({"qid": q["qid"], "question": q["question"],
                                    "phase": phase, "k": k, "docs": docs,
                                    "label": label, "ranked": lst}) + "\n")
        logger.info(f"{split}: samples done, suff rates={n_suff}")

    with open(SAMPLES / "train.jsonl", "w") as f:
        emit(train_qs, "train", f)
    with open(SAMPLES / "val.jsonl", "w") as f:
        emit(val_qs, "val", f)
    with open(SAMPLES / "test.jsonl", "w") as f:
        emit(test_qs, "test", f)
    logger.info("samples saved -> data/critic_samples/")

    # save gold mapping for eval
    json.dump({q["qid"]: q["gold_docs"] for q in test_qs},
              open(DATA / "critic_samples/test_gold.json", "w"))
    logger.info("ALL DONE")


if __name__ == "__main__":
    main()

"""
Download NQ + HotpotQA for retrieval-budget pilot
==================================================
NQ: 单跳 (1 gold doc per question, answer extracted from the doc)
HotpotQA: 多跳 (2-4 supporting facts docs per question)

Output (data/):
  nq.json          {questions: [...], kb: [doc_id -> text]}
  hotpotqa.json    {questions: [...], kb: [doc_id -> text]}

Usage:
  HF_ENDPOINT=https://hf-mirror.com python3 scripts/download_data.py
"""

import json, logging, random, re, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

NQ_MAX_SCAN = 40000
NQ_KB_SIZE = 10000
NQ_TEST_SIZE = 1000

HOTPOT_MAX_SCAN = 7405  # full validation set
HOTPOT_KB_SIZE = 10000
HOTPOT_TEST_SIZE = 1000

SEED = 42


def download_nq():
    from datasets import load_dataset

    logger.info("Loading NQ (streaming)...")
    ds = load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)

    docs = {}  # doc_id -> full text
    questions = []
    scanned = 0
    for item in ds:
        scanned += 1
        if scanned > NQ_MAX_SCAN:
            break

        tokens = item["document"]["tokens"]["token"]
        if not tokens:
            continue
        doc_text = " ".join(tokens[:1024])  # 截断控制体积

        q = item["question"]["text"]
        if not q:
            continue

        # short / long answer
        answer = ""
        for sa in item["annotations"]["short_answers"]:
            start, end = sa["start_token"], sa["end_token"]
            if isinstance(start, list):
                start = start[0] if start else -1
            if isinstance(end, list):
                end = end[0] if end else -1
            if start >= 0 and end > start:
                answer = " ".join(tokens[start:end]).strip()
                break
        if not answer:
            la = item["annotations"]["long_answer"]
            if la and la[0]["start_token"] >= 0:
                start, end = la[0]["start_token"], la[0]["end_token"]
                if isinstance(start, list):
                    start = start[0] if start else -1
                if isinstance(end, list):
                    end = end[0] if end else -1
                if start >= 0 and end > start:
                    answer = " ".join(tokens[start:end]).strip()
        if not answer or len(answer) < 2:
            continue

        doc_id = f"doc_{scanned:06d}"
        docs[doc_id] = doc_text
        questions.append({"qid": f"nq_{scanned:06d}", "question": q,
                          "answer": answer, "gold_docs": [doc_id]})
        if len(docs) >= NQ_KB_SIZE:
            break

    rng = random.Random(SEED)
    rng.shuffle(questions)
    test = questions[:NQ_TEST_SIZE]
    logger.info(f"NQ: scanned={scanned}, kb_docs={len(docs)}, test={len(test)}")
    return {"name": "NQ", "kb": docs, "questions": test}


def download_hotpotqa():
    from datasets import load_dataset

    logger.info("Loading HotpotQA (distractor validation)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)

    gold_docs = {}
    all_docs = {}
    questions = []
    scanned = 0
    for item in ds:
        scanned += 1
        if scanned > HOTPOT_MAX_SCAN:
            break

        q = item["question"]
        gold_titles = set(item["supporting_facts"]["title"])
        gold_ids = []

        # context: {"title": [...], "sentences": [[...], ...]}
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        for i, title in enumerate(titles):
            sentences = sentences_list[i] if i < len(sentences_list) else []
            doc_id = f"hp_{len(all_docs):06d}"
            all_docs[doc_id] = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            if title in gold_titles:
                gold_docs[doc_id] = all_docs[doc_id]
                gold_ids.append(doc_id)

        if not gold_ids:
            continue

        questions.append({"qid": f"hotpot_{scanned:06d}", "question": q,
                          "answer": item["answer"], "gold_docs": gold_ids,
                          "type": item["type"], "n_supporting": len(gold_ids)})

    # KB: gold docs + random fill to HOTPOT_KB_SIZE
    rng = random.Random(SEED)
    extra = [d for d in all_docs if d not in gold_docs]
    rng.shuffle(extra)
    kb = dict(gold_docs)
    for d in extra:
        if len(kb) >= HOTPOT_KB_SIZE:
            break
        kb[d] = all_docs[d]
    for d in list(gold_docs):
        if d not in kb:
            kb[d] = gold_docs[d]

    # keep only questions whose gold docs are all in kb
    questions = [q for q in questions if all(g in kb for g in q["gold_docs"])]
    rng.shuffle(questions)
    test = questions[:HOTPOT_TEST_SIZE]
    logger.info(f"HotpotQA: scanned={scanned}, kb_docs={len(kb)}, test={len(test)}")
    return {"name": "HotpotQA", "kb": kb, "questions": test}


def main():
    DATA.mkdir(exist_ok=True)
    nq = download_nq()
    with open(DATA / "nq.json", "w", encoding="utf-8") as f:
        json.dump(nq, f, ensure_ascii=False)
    hp = download_hotpotqa()
    with open(DATA / "hotpotqa.json", "w", encoding="utf-8") as f:
        json.dump(hp, f, ensure_ascii=False)

    logger.info(f"Saved → {DATA}/nq.json, {DATA}/hotpotqa.json")


if __name__ == "__main__":
    main()
"""
Unified retrieval for the three pilot phases
=============================================
phase1: dense          (FAISS bge-large-en-v1.5)
phase2: hybrid         (dense + BM25 via RRF)
phase3: rerank         (hybrid top-20 -> bge-reranker-base)

Usage:
  python3 scripts/retrieve.py --dataset nq --phase dense  --ks 3,5,10,20
  python3 scripts/retrieve.py --dataset hotpotqa --phase hybrid --ks 3,5,10,20
  python3 scripts/retrieve.py --dataset nq --phase rerank --ks 3,5,10
"""

import json, logging, pickle, re, argparse
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
INDEX_DIR = DATA / "indices"
RESULTS_DIR = ROOT / "results"

DENSE_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODELS = {"base": "BAAI/bge-reranker-base", "v2m3": "BAAI/bge-reranker-v2-m3"}
RRF_K = 60
COARSE_K = 20
DEFAULT_MAX_LEN = 512
STOPWORDS = set("a an the and or but if because as of at by for with about to in on is are was were be been it its this that these those".split())
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def load_dataset(name):
    data = json.load(open(DATA / f"{name}.json", encoding="utf-8"))
    return data["kb"], data["questions"]


def load_dense(name):
    index = faiss.read_index(str(INDEX_DIR / f"{name}_dense.faiss"))
    doc_ids = np.load(INDEX_DIR / f"{name}_doc_ids.npy", allow_pickle=True).tolist()
    return index, doc_ids


def load_bm25(name):
    with open(INDEX_DIR / f"{name}_bm25.pkl", "rb") as f:
        p = pickle.load(f)
    return p["bm25"], p["doc_ids"]


def rrf_merge(dense_scores, dense_idx, bm25_scores, bm25_idx, k=RRF_K, top=20):
    """dense: scores[0] 按排名排列, idxs[0] 是文档位置; bm25: scores 按文档位置排列。"""
    n = len(dense_scores)
    rrf = np.zeros(n)
    d_ord = dense_idx[:200]          # 文档位置
    b_ord = np.argsort(-bm25_scores)[:200]  # 文档位置
    for rank, i in enumerate(d_ord):
        rrf[i] += 1.0 / (k + rank + 1)
    for rank, i in enumerate(b_ord):
        rrf[i] += 1.0 / (k + rank + 1)
    top_idx = np.argsort(-rrf)[:top]
    return top_idx, rrf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nq", "hotpotqa"])
    parser.add_argument("--phase", required=True, choices=["dense", "hybrid", "rerank"])
    parser.add_argument("--ks", default="3,5,10,20")
    parser.add_argument("--reranker", default="v2m3", choices=["base", "v2m3"])
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    args = parser.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    kb, questions = load_dataset(args.dataset)
    logger.info(f"{args.dataset}: kb={len(kb)}, test={len(questions)}")

    index, doc_ids = load_dense(args.dataset)
    bm25, bm25_ids = load_bm25(args.dataset)
    assert doc_ids == bm25_ids

    q_model = SentenceTransformer(DENSE_MODEL)
    q_embs = q_model.encode([q["question"] for q in questions],
                            normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    # per-query retrieval
    results = []
    if args.phase == "dense":
        for qi, emb in enumerate(q_embs):
            scores, idxs = index.search(np.array([emb], dtype="float32"), k=max(ks))
            results.append({"qid": questions[qi]["qid"], "gold": questions[qi]["gold_docs"],
                            "ranked": [doc_ids[i] for i in idxs[0]]})
    else:
        # hybrid scores per query (needed for both hybrid & rerank)
        for qi, emb in enumerate(q_embs):
            scores, idxs = index.search(np.array([emb], dtype="float32"), k=len(doc_ids))
            dense_scores = scores[0]
            bm25_scores = bm25.get_scores(tokenize(questions[qi]["question"]))
            top_idx, rrf = rrf_merge(dense_scores, idxs[0], bm25_scores, None, top=COARSE_K)
            ranked = [doc_ids[i] for i in top_idx]
            if args.phase == "rerank":
                ranked = ranked  # rerank below
            results.append({"qid": questions[qi]["qid"], "gold": questions[qi]["gold_docs"],
                            "ranked": ranked})
        if args.phase == "rerank":
            reranker = CrossEncoder(RERANK_MODELS[args.reranker], max_length=args.max_len)
            for qi, r in enumerate(results):
                pairs = [(questions[qi]["question"], kb[d]) for d in r["ranked"]]
                scores = reranker.predict(pairs)
                order = np.argsort(-np.array(scores))
                r["ranked"] = [r["ranked"][i] for i in order]
                if (qi + 1) % 200 == 0:
                    logger.info(f"rerank {qi + 1}/{len(results)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.phase == "rerank":
        out = RESULTS_DIR / f"{args.dataset}_{args.phase}_{args.reranker}.json"
    else:
        out = RESULTS_DIR / f"{args.dataset}_{args.phase}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ks": ks, "results": results}, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
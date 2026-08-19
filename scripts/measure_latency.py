"""
Per-stage retrieval latency measurement
=======================================
Measures wall-clock latency of each escalation stage on a query sample:
  dense@20  : FAISS IndexFlatIP search on KB (query embedding excluded)
  hybrid@20 : FAISS + BM25 + RRF
  rerank    : CrossEncoder(bge-reranker-v2-m3) scoring 20 pairs
  embed     : bge-large-en-v1.5 query embedding (one-time cost)

Output: results/latency.md + results/latency.json
"""

import json, logging, pickle, re, time
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
INDEX_DIR = DATA / "indices"
RESULTS = ROOT / "results"

DENSE_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RRF_K = 60
STOPWORDS = set("a an the and or but if because as of at by for with about to in on is are was were be been it its this that these those".split())
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")
N_SAMPLE = 200


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    args = ap.parse_args()
    ds = args.dataset

    data = json.load(open(DATA / f"{ds}.json", encoding="utf-8"))
    kb, questions = data["kb"], data["questions"][:args.sample]
    n = len(questions)

    index = faiss.read_index(str(INDEX_DIR / f"{ds}_dense.faiss"))
    doc_ids = np.load(INDEX_DIR / f"{ds}_doc_ids.npy", allow_pickle=True).tolist()
    with open(INDEX_DIR / f"{ds}_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)["bm25"]

    q_model = SentenceTransformer(DENSE_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    q_embs = q_model.encode([q["question"] for q in questions], normalize_embeddings=True,
                            batch_size=64, show_progress_bar=False)

    t_embed = []
    t_dense = []
    t_hybrid = []
    t_rerank = []
    for qi, (q, emb) in enumerate(zip(questions, q_embs)):
        t0 = time.perf_counter()
        q_model.encode([q["question"]], normalize_embeddings=True)
        t_embed.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        index.search(np.array([emb], dtype="float32"), k=20)
        t_dense.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        scores, idxs = index.search(np.array([emb], dtype="float32"), k=len(doc_ids))
        bm = bm25.get_scores(tokenize(q["question"]))
        n_docs = len(doc_ids)
        rrf = np.zeros(n_docs)
        for rank, i in enumerate(idxs[0][:200]):
            rrf[i] += 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(np.argsort(-bm)[:200]):
            rrf[i] += 1.0 / (RRF_K + rank + 1)
        np.argsort(-rrf)[:20]
        t_hybrid.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        reranker.predict([(q["question"], kb[doc_ids[i]]) for i in idxs[0][:20]])
        t_rerank.append((time.perf_counter() - t0) * 1000)

    def stat(v):
        v = np.array(v)
        return float(v.mean()), float(np.median(v)), float(v.std())

    res = {k: {"mean_ms": stat(v)[0], "median_ms": stat(v)[1], "std_ms": stat(v)[2]}
           for k, v in [("embed", t_embed), ("dense20", t_dense), ("hybrid20", t_hybrid), ("rerank20", t_rerank)]}

    md = [f"# 检索阶段延迟（{ds}, n={n}, GPU1）", "",
          "| 阶段 | mean(ms) | median(ms) | std(ms) |", "|---|---|---|---|"]
    for k in ["embed", "dense20", "hybrid20", "rerank20"]:
        md.append(f"| {k} | {res[k]['mean_ms']:.1f} | {res[k]['median_ms']:.1f} | {res[k]['std_ms']:.1f} |")
    md.append("")
    md.append("> 说明：embed = 单查询 bge-large-en-v1.5 编码；dense20 = FAISS 全库搜索取 top-20；"
              "hybrid20 = dense 全库 + BM25 + RRF 融合；rerank20 = v2-m3 对 20 个候选取分。"
              "注：hybrid20 含 dense 全库搜索（比 dense20 的 top-20 搜索贵）。")
    with open(RESULTS / "latency.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(RESULTS / "latency.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info(f"Saved -> results/latency.md ({json.dumps({k: round(v['mean_ms'], 1) for k, v in res.items()})})")


if __name__ == "__main__":
    main()

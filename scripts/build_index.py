"""
Build FAISS (bge-large-en-v1.5) + BM25 indices
==============================================
Usage:
  HF_HUB_OFFLINE=1 python3 scripts/build_index.py [--dataset nq|hotpotqa|all]
"""

import json, logging, re, argparse
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
INDEX_DIR = ROOT / "data" / "indices"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
STOPWORDS = set("a an the and or but if because as of at by for with about to in on is are was were be been it its this that these those".split())
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def build(dataset: str):
    data = json.load(open(DATA / f"{dataset}.json", encoding="utf-8"))
    kb = data["kb"]
    doc_ids = sorted(kb.keys())
    texts = [kb[d] for d in doc_ids]

    # FAISS
    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(np.array(embs, dtype="float32"))
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / f"{dataset}_dense.faiss"))
    np.save(INDEX_DIR / f"{dataset}_doc_ids.npy", np.array(doc_ids))

    # BM25
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    import pickle
    with open(INDEX_DIR / f"{dataset}_bm25.pkl", "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "bm25": bm25}, f)

    logger.info(f"{dataset}: {len(doc_ids)} docs indexed (dense {embs.shape[1]}d + bm25)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nq", "hotpotqa", "all"], default="all")
    args = parser.parse_args()
    for d in (["nq", "hotpotqa"] if args.dataset == "all" else [args.dataset]):
        build(d)


if __name__ == "__main__":
    main()
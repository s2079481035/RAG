"""
Evaluate Recall@K from retrieval results
=========================================
NQ (单跳):     Recall@K = top-K 含唯一 gold doc 的比例
HotpotQA (多跳): Hit@K  = top-K 含至少一个 gold doc 的比例 (主口径)
                Recall@K = gold docs 中被 top-K 覆盖的比例 (辅口径)

Usage:
  python3 scripts/eval_recall.py
输出: results/recall_matrix.md + results/recall_matrix.json
"""

import json, logging, argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

DEFAULT_KS = [3, 5, 10, 20]


def eval_file(name, ks):
    data = json.load(open(RESULTS / f"{name}.json", encoding="utf-8"))
    n = len(data["results"])
    out = {"ks": {}, "per_query": []}
    for k in ks:
        hit = 0
        rec_total = 0.0
        per_q = []
        for r in data["results"]:
            ranked = r["ranked"][:k]
            gold = set(r["gold"])
            overlap = len(set(ranked) & gold)
            h = 1 if overlap > 0 else 0
            hit += h
            rec_total += overlap / len(gold)
            per_q.append({"qid": r["qid"], "hit": h, "recall": overlap / len(gold),
                          "gold": list(gold), "topk": ranked})
        out["ks"][k] = {"hit@k": hit / n, "recall@k": rec_total / n}
        out["per_query"].append({"k": k, "items": per_q})
    logger.info(f"{name}: n={n} → {json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out['ks'].items()})}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    args = parser.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    matrix = {}
    for name in ["nq_dense", "nq_hybrid", "nq_rerank", "hotpotqa_dense", "hotpotqa_hybrid", "hotpotqa_rerank"]:
        p = RESULTS / f"{name}.json"
        if p.exists():
            matrix[name] = eval_file(name, ks)

    # markdown table
    lines = ["# Recall@K 矩阵 (Retrieval-Budget Pilot)", ""]
    lines.append("| 数据集 | 检索器 | " + " | ".join(f"Recall@{k}" for k in ks) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in ks) + " |")
    for name, m in matrix.items():
        ds, method = name.split("_", 1)
        vals = " | ".join(f"{m['ks'][k]['recall@k']:.3f}" for k in ks)
        lines.append(f"| {ds} | {method} | {vals} |")
    lines.append("")
    lines.append("Hit@K (HotpotQA 主口径):")
    for k in ks:
        h = matrix["hotpotqa_dense"]["ks"][k]["hit@k"]
        lines.append(f"- HotpotQA dense Hit@{k} = {h:.3f}")
    with open(RESULTS / "recall_matrix.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(RESULTS / "recall_matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → results/recall_matrix.md, results/recall_matrix.json")


if __name__ == "__main__":
    main()
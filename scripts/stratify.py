"""
Stratified analysis on top of retrieval results
================================================
Inputs:
  results/{ds}_{method}[_{variant}].json   per-query ranked lists
  data/{ds}.json                            question metadata (type, n_sent, ...)

Analyses:
  1) HotpotQA by question type (bridge / comparison)          -> Recall@K per method
  2) HotpotQA by evidence amount (n_sent: 2 / 3+)             -> Recall@K per method
  3) Query difficulty (retrieval-based): Easy / Medium / Hard -> Recall@K per method
  4) Oracle Adaptive: cheapest strategy with full coverage    -> vs fixed strategies

Difficulty (retrieval-based, oracle-grounded):
  Easy   = every gold doc reachable at rank <= 3 by at least one method
  Medium = not Easy, and every gold doc reachable at rank <= 10
  Hard   = otherwise (some gold doc never reaches top-10)

Oracle strategies (in cost order), pool = hybrid top-20:
  dense@3 < dense@5 < hybrid@5 < hybrid@10 < rerank@10 < hybrid@20 < rerank@20

Usage:
  python3 scripts/stratify.py
Output: results/stratify.md + results/stratify.json
"""

import json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

METHODS = ["dense", "hybrid", "rerank"]
VARIANT = "v2m3"
KS = [3, 5, 10, 20]

# cost-ordered strategies for oracle routing (value = max rank consumed)
STRATEGIES = [
    ("dense@3",     "dense",   3),
    ("dense@5",     "dense",   5),
    ("hybrid@5",    "hybrid",  5),
    ("hybrid@10",   "hybrid",  10),
    ("rerank@10",   "rerank",  10),
    ("hybrid@20",   "hybrid",  20),
    ("rerank@20",   "rerank",  20),
]


def load_results(ds):
    out = {}
    for m in METHODS:
        p = RESULTS / f"{ds}_{m}_{VARIANT}.json" if m == "rerank" else RESULTS / f"{ds}_{m}.json"
        if not p.exists():
            p = RESULTS / f"{ds}_{m}.json"
        out[m] = json.load(open(p, encoding="utf-8"))["results"]
    return out


def load_meta(ds):
    d = json.load(open(DATA / f"{ds}.json", encoding="utf-8"))
    return {q["qid"]: q for q in d["questions"]}


def recall(q, ranked, gold, k):
    ov = len(set(ranked[:k]) & set(gold))
    return ov / len(gold)


def hit(q, ranked, gold, k):
    return 1 if set(ranked[:k]) & set(gold) else 0


def best_rank(method, qid, gold, results):
    """best achievable rank (1-based) of any gold doc; None if not in top-20"""
    ranked = results[method][qid]["ranked"]
    ranks = [i + 1 for i, d in enumerate(ranked) if d in gold]
    return min(ranks) if ranks else None


def difficulty(qid, gold, results):
    """per-query retrieval difficulty using the hardest gold doc"""
    worst = []
    for m in METHODS:
        r = best_rank(m, qid, gold, results)
        if r is not None:
            worst.append(r)
    if not worst:
        return "hard"
    b = min(worst)  # best achievable rank of the easiest-to-find gold doc? no: hardest doc
    # recompute per-doc: we want EVERY gold doc reachable within k by some method
    return difficulty_strict(qid, gold, results)


def difficulty_strict(qid, gold, results):
    every = True
    for g in gold:
        ranks = [best_rank(m, qid, [g], results) for m in METHODS]
        ranks = [r for r in ranks if r is not None]
        if not ranks or min(ranks) > 10:
            every = False
    if every:
        for g in gold:
            ranks = [best_rank(m, qid, [g], results) for m in METHODS]
            ranks = [r for r in ranks if r is not None]
            if not ranks or min(ranks) > 3:
                return "medium"
        return "easy"
    return "hard"


def table_rows(rows, header):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return lines


def main():
    out = {}
    md = ["# 分层分析 (Stratified Recall)", ""]

    for ds in ["nq", "hotpotqa"]:
        results = load_results(ds)
        meta = load_meta(ds)
        by_qid = {m: {r["qid"]: r for r in results[m]} for m in METHODS}
        qids = [r["qid"] for r in results["dense"]]

        # ---- difficulty groups ----
        diff = {}
        for qid in qids:
            gold = set(by_qid["dense"][qid]["gold"])
            diff[qid] = difficulty_strict(qid, gold, by_qid)
        n = len(qids)
        md.append(f"## {ds} — 查询难度分布")
        from collections import Counter
        c = Counter(diff.values())
        md.append(" | ".join(f"{k}={c[k]}({100 * c[k] / n:.1f}%)" for k in ["easy", "medium", "hard"]))
        md.append("")

        # ---- stratified Recall@K per method ----
        def strat_rows(group_fn):
            rows = []
            for gname in sorted({group_fn(q) for q in qids}):
                sub = [q for q in qids if group_fn(q) == gname]
                for m in METHODS:
                    row = [f"{ds} | {gname} | {m}"]
                    for k in KS:
                        rc = sum(recall(q, by_qid[m][q]["ranked"], by_qid[m][q]["gold"], k) for q in sub)
                        row.append(f"{rc / len(sub):.3f}")
                    rows.append(row)
            return rows

        md.append(f"### {ds} — 按难度分组 Recall@K")
        md += table_rows(strat_rows(lambda q: diff[q]), ["数据集", "分组", "检索器"] + [f"R@{k}" for k in KS])
        md.append("")

        if ds == "hotpotqa":
            def typ(q):
                return meta[q]["type"]
            md.append("### HotpotQA — 按 question type 分组 Recall@K")
            md += table_rows(strat_rows(typ), ["数据集", "分组", "检索器"] + [f"R@{k}" for k in KS])
            md.append("")

            def nsent(q):
                return "3+" if meta[q]["n_sent"] >= 3 else "2"
            md.append("### HotpotQA — 按 supporting sentences 数量分组 Recall@K")
            md += table_rows(strat_rows(nsent), ["数据集", "分组", "检索器"] + [f"R@{k}" for k in KS])
            md.append("")

        # ---- oracle adaptive ----
        # for each query, cheapest strategy in STRATEGIES order that covers all gold;
        # fallback: strategy with max coverage (tie -> cheaper cost)
        strat_idx = {s[0]: i for i, s in enumerate(STRATEGIES)}
        covs = {s[0]: [] for s in STRATEGIES}
        oracle_cov, oracle_any, oracle_cost = [], [], []
        for qid in qids:
            gold = set(by_qid["dense"][qid]["gold"])
            full = None
            best_cov, best_i = -1, 0
            for i, (name, m, k) in enumerate(STRATEGIES):
                ranked = by_qid[m][qid]["ranked"]
                c = len(set(ranked[:k]) & gold) / len(gold)
                covs[name].append(c)
                if c >= 1 and full is None:
                    full = name
                if c > best_cov or (c == best_cov and i < best_i):
                    best_cov, best_i = c, i
            pick = full if full is not None else STRATEGIES[best_i][0]
            oracle_cov.append(best_cov if full is None else 1.0)
            oracle_any.append(best_cov)
            oracle_cost.append(strat_idx[pick] + 1)

        fixed = ["dense@3", "dense@5", "hybrid@5", "hybrid@10", "rerank@10", "hybrid@20", "rerank@20"]
        md.append(f"### {ds} — Oracle Adaptive vs Fixed")
        rows = []
        for name in fixed:
            m = STRATEGIES[strat_idx[name]][1]
            k = STRATEGIES[strat_idx[name]][2]
            rc = sum(recall(q, by_qid[m][q]["ranked"], by_qid[m][q]["gold"], k) for q in qids)
            rows.append([name, f"{rc / n:.3f}", "-"])
        rows.append(["ORACLE(全命中)", f"{sum(oracle_cov) / n:.3f}", f"{sum(oracle_cost) / n:.2f}"])
        rows.append(["ORACLE(含未命中)", f"{sum(oracle_any) / n:.3f}", f"{sum(oracle_cost) / n:.2f}"])
        md.append("| 策略 | Recall(全部gold覆盖) | 平均成本级(1-7) |")
        md.append("|---|---|---|")
        md += [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows]
        md.append("")

        out[ds] = {"difficulty": dict(diff),
                   "oracle": {"mean_coverage_full": sum(oracle_cov) / n,
                              "mean_coverage_any": sum(oracle_any) / n,
                              "mean_cost": sum(oracle_cost) / n}}

    with open(RESULTS / "stratify.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(RESULTS / "stratify.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Saved -> results/stratify.md, results/stratify.json")


if __name__ == "__main__":
    main()

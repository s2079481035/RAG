"""
Oracle Early Stopping experiment (Phase 2b)
===========================================
Protocol: EXPERIMENT_PROTOCOL.md section 9 (frozen 2026-08-19)

Oracle sufficiency: current stage's top-K covers ALL gold docs -> stop;
otherwise escalate. Final stage always stops (may be insufficient).

Ladders:
  A: dense@3 -> hybrid@5 -> hybrid@10 -> rerank@10 -> rerank@20
  B: dense@10 -> hybrid@10 -> rerank@10 -> rerank@20

Metrics per dataset: Recall@final, Full Coverage Rate, avg docs cost,
avg ladder level, stage stopping distribution, vs fixed rerank@10/rerank@20.

Output: results/early_stop.md + results/early_stop.json
"""

import json, logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

LADDERS = {
    "A": [("dense@3", "dense", 3), ("hybrid@5", "hybrid", 5), ("hybrid@10", "hybrid", 10),
          ("rerank@10", "rerank", 10), ("rerank@20", "rerank", 20)],
    "B": [("dense@10", "dense", 10), ("hybrid@10", "hybrid", 10),
          ("rerank@10", "rerank", 10), ("rerank@20", "rerank", 20)],
}


def load(ds, m):
    p = RESULTS / f"{ds}_{m}_v2m3.json" if m == "rerank" else RESULTS / f"{ds}_{m}.json"
    if not p.exists():
        p = RESULTS / f"{ds}_{m}.json"
    return json.load(open(p, encoding="utf-8"))["results"]


def run_ladder(ladder, R, qids, gold_of):
    """returns per-query: (stop_level 1-based, stop_docs, final_recall, full_coverage)"""
    out = {}
    never_sufficient = 0
    for qid in qids:
        gold = gold_of[qid]
        stop = len(ladder)  # final stage
        for i, (name, m, k) in enumerate(ladder, start=1):
            ranked = R[m][qid]["ranked"]
            if set(ranked[:k]) >= gold:
                stop = i
                break
        _, m, k = ladder[stop - 1]
        lst = R[m][qid]["ranked"][:k]
        rec = len(set(lst) & gold) / len(gold)
        out[qid] = {"level": stop, "docs": k, "recall": rec, "full": int(rec >= 1)}
        if rec < 1:
            never_sufficient += 1
    return out, never_sufficient


def main():
    md = ["# Oracle Early Stopping（Phase 2b）", "> 协议: EXPERIMENT_PROTOCOL.md 第 9 节（2026-08-19 冻结）", ""]
    summary = {}
    for ds in ["nq", "hotpotqa"]:
        R = {m: {r["qid"]: r for r in load(ds, m)} for m in ["dense", "hybrid", "rerank"]}
        qids = [r["qid"] for r in load(ds, "dense")]
        gold_of = {qid: set(R["dense"][qid]["gold"]) for qid in qids}
        n = len(qids)
        md.append(f"## {ds}（n={n}）")
        md.append("")
        md.append("### Oracle Early Stopping vs Fixed（Recall / 平均成本）")
        md.append("| 系统 | Recall@final | FullCoverage | 平均文档成本 | 平均阶梯级 | 不可充分满足 |")
        md.append("|---|---|---|---|---|---|")
        fixed = [("rerank@10", "rerank", 10), ("rerank@20", "rerank", 20)]
        for name, m, k in fixed:
            rc = sum(len(set(R[m][q]["ranked"][:k]) & gold_of[q]) / len(gold_of[q]) for q in qids) / n
            fc = sum(1 for q in qids if set(R[m][q]["ranked"][:k]) >= gold_of[q]) / n
            md.append(f"| {name}(fixed) | {rc:.3f} | {fc:.3f} | {k:.0f} | - | - |")
        for lname, ladder in LADDERS.items():
            per, never = run_ladder(ladder, R, qids, gold_of)
            rec = sum(v["recall"] for v in per.values()) / n
            fc = sum(v["full"] for v in per.values()) / n
            docs = sum(v["docs"] for v in per.values()) / n
            lvl = sum(v["level"] for v in per.values()) / n
            md.append(f"| oracleES-{lname} | {rec:.3f} | {fc:.3f} | {docs:.2f} | {lvl:.2f} | {never} |")
            summary.setdefault(ds, {})[f"es_{lname}"] = {
                "recall": rec, "full_cov": fc, "avg_docs": docs, "avg_level": lvl,
                "never_sufficient": never,
                "stage_dist": {str(k): v for k, v in Counter(p["level"] for p in per.values()).most_common()}}
        md.append("")
        md.append("### 各阶段停止比例（Oracle Early Stopping）")
        md.append("| 阶梯 | 阶段分布（level 1..n 停止的查询数） |")
        md.append("|---|---|")
        for lname, ladder in LADDERS.items():
            per, _ = run_ladder(ladder, R, qids, gold_of)
            dist = Counter(p["level"] for p in per.values())
            parts = " ".join(f"S{i}:{dist[i]}({100 * dist[i] / n:.1f}%)" for i in range(1, len(ladder) + 1))
            md.append(f"| {lname} | {parts} |")
        md.append("")
        md.append("> False Early Stop（oracle 口径）：按构造为 0（充分性由 gold 判定，不会误停）。"
                  "False Early Stop 只在 Critic 阶段有定义：Critic 判 sufficient 但 gold 未被覆盖的查询比例。"
                  "Oracle 阶段同时报告不可充分满足数作为 recall 天花板信息。")
        md.append("")
    with open(RESULTS / "early_stop.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(RESULTS / "early_stop.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved -> results/early_stop.md, results/early_stop.json")


if __name__ == "__main__":
    main()

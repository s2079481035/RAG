"""
Validation-set policy optimization for the Learned Retrieval Sufficiency Critic
================================================================================
- Threshold sweep (global + stage-specific) on VALIDATION set only
- Full E2E escalation simulation per config: Dense@3 -> Hybrid@5 -> Rerank@20
- Selection: val Recall >= 0.94 constraint, minimize Avg Cost (Pareto-efficient)
- Selected config evaluated ONCE on test (selection never uses test)
- Export test False-Early-Stop queries @ t=0.5 with full diagnostics

Usage: CUDA_VISIBLE_DEVICES=1 python3 scripts/policy_sweep.py
Output: results/critic_policy.md/.json, results/fes_queries_test_t05.json,
        results/recall_cost_curve.png (if matplotlib available)
"""

import json, logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "data" / "critic_samples"
RESULTS = ROOT / "results"
MODEL_DIR = ROOT / "models"

MAX_LEN = 512
STAGE_ORDER = ["dense", "hybrid", "rerank"]
LADDER = [("dense", 3), ("hybrid", 5), ("rerank", 20)]
K_OF = dict(LADDER)
STAGE_LAT = {"dense": 31.5, "hybrid": 93.5, "rerank": 220.9}  # ms cumulative (measured 2026-08-19)

GLOBAL_TS = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
DENSE_TS = [0.55, 0.65, 0.75, 0.85, 0.95]
HYBRID_TS = [0.35, 0.45, 0.55, 0.65, 0.75]
RECALL_FLOOR = 0.94


def load_items(split):
    items = [json.loads(l) for l in open(SAMPLES / f"{split}.jsonl", encoding="utf-8")]
    return items


def predict(model, tok, items):
    probs = []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            texts = [it["question"] + " [SEP] " + it["docs"] for it in items[i:i + 64]]
            enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to("cuda")
            logits = model(**enc).logits
            probs += torch.softmax(logits, -1)[:, 1].cpu().tolist()
    for it, p in zip(items, probs):
        it["p"] = float(p)
    return items


def group_by_qid(items):
    g = {}
    for it in items:
        g.setdefault(it["qid"], {})[it["phase"]] = it
    return g


def simulate(by_qid, gold, t_dense=0.50, t_hybrid=None):
    """Full E2E escalation. t_hybrid defaults to t_dense (global threshold)."""
    if t_hybrid is None:
        t_hybrid = t_dense
    thr = {"dense": t_dense, "hybrid": t_hybrid}
    n = len(by_qid)
    rec = full = docs = lat = fes = 0.0
    dist = {"dense": 0, "hybrid": 0, "rerank": 0}
    detail = {}
    for qid, ph in by_qid.items():
        gset = set(gold[qid])
        stop = None
        for s, k in LADDER:
            cov = len(set(ph[s]["ranked"]) & gset) / len(gset)
            if stop is None and (s == "rerank" or ph[s]["p"] >= thr[s]):
                stop = (s, k, cov)
                break
        s, k, cov = stop
        dist[s] += 1
        rec += cov
        full += 1.0 if cov == 1.0 else 0.0
        docs += k
        lat += STAGE_LAT[s] + CRITIC_LAT * (STAGE_ORDER.index(s) + 1)
        if s != "rerank" and cov < 1.0:
            fes += 1
        detail[qid] = {"stop": s, "cov": cov}
    return {"recall": rec / n, "full": full / n, "docs": docs / n, "lat": lat / n,
            "fes": fes / n, "dist": dist, "_detail": detail}


def fmt_row(name, m):
    d = m["dist"]
    return (f"| {name} | {m['recall']:.3f} | {m['full']:.3f} | {m['docs']:.2f} | "
            f"{m['lat']:.1f} | {m['fes']:.3f} | {d['dense']}/{d['hybrid']}/{d['rerank']} |")


def main():
    global CRITIC_LAT
    try:
        CRITIC_LAT = json.load(open(RESULTS / "critic_eval.json"))["e2e"]["critic_lat_ms"]
    except Exception:
        CRITIC_LAT = 4.7

    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-base", local_files_only=True)
    model.classifier.out_proj = torch.nn.Linear(model.config.hidden_size, 2)
    model.load_state_dict(torch.load(MODEL_DIR / "critic.pt", map_location="cpu",
                                     weights_only=True))
    model.to("cuda").eval()

    # ---- predictions ----
    val_items = load_items("val")
    test_items = load_items("test")
    train_qids = {json.loads(l)["qid"] for l in open(SAMPLES / "train.jsonl", encoding="utf-8")}
    leak = [it["qid"] for it in test_items if it["qid"] in train_qids]
    assert not leak, "leak!"
    logger.info("predicting val...")
    val = group_by_qid(predict(model, tok, val_items))
    logger.info("predicting test...")
    test = group_by_qid(predict(model, tok, test_items))

    allq = json.load(open(ROOT / "data" / "hotpotqa_all.json", encoding="utf-8"))["questions"]
    gold_all = {q["qid"]: q["gold_docs"] for q in allq}
    gold_val = {q: gold_all[q] for q in val}
    gold_test = json.load(open(SAMPLES / "test_gold.json"))

    # ---- 1. global threshold sweep on VAL ----
    rows = []
    for t in GLOBAL_TS:
        m = simulate(val, gold_val, t_dense=t)
        rows.append((("global", t, t), m))
        logger.info(f"val t={t:.2f}: R={m['recall']:.3f} Full={m['full']:.3f} "
                    f"cost={m['docs']:.2f} lat={m['lat']:.1f} FES={m['fes']:.3f}")

    # ---- 2. stage-specific grid on VAL ----
    grid_rows = []
    for td in DENSE_TS:
        for th in HYBRID_TS:
            m = simulate(val, gold_val, t_dense=td, t_hybrid=th)
            grid_rows.append(((f"d{td}/h{th}", td, th), m))

    # ---- 3. Pareto selection on VAL ----
    cand = [(c, m) for c, m in rows + grid_rows]
    ok = [(c, m) for c, m in cand if m["recall"] >= RECALL_FLOOR]
    ok.sort(key=lambda cm: cm[1]["docs"])
    assert ok, f"No config reaches Recall>={RECALL_FLOOR} on val"
    best_cfg, best_m = ok[0]
    # pareto front within feasible set (cost ascending, keep strictly-better recall)
    front, best_r = [], -1
    for c, m in ok:
        if m["recall"] > best_r:
            front.append((c, m))
            best_r = m["recall"]

    # ---- 4. evaluate selected config ONCE on TEST ----
    test_best = simulate(test, gold_test, t_dense=best_cfg[1], t_hybrid=best_cfg[2])
    test_05 = simulate(test, gold_test, 0.50, 0.50)

    # ---- 5. FES query export (test @ t=0.5) ----
    fes_detail = []
    for qid, ph in test.items():
        d = test_05["_detail"][qid]
        if d["stop"] != "rerank" and d["cov"] < 1.0:
            gset = set(gold_test[qid])
            entry = {
                "qid": qid,
                "question": next(it["question"] for it in ph.values()),
                "gold_docs": sorted(gset),
                "stop_stage": d["stop"],
                "coverage_at_stop": round(d["cov"], 4),
                "stages": {
                    s: {"k": K_OF[s], "retrieved": ph[s]["ranked"],
                        "covered_gold": sorted(gset & set(ph[s]["ranked"])),
                        "coverage": round(len(gset & set(ph[s]["ranked"])) / len(gset), 4),
                        "critic_p": round(ph[s]["p"], 4)}
                    for s in STAGE_ORDER},
                "would_continue_at_t08": bool(ph["dense"]["p"] < 0.8 and ph["hybrid"]["p"] < 0.8),
            }
            fes_detail.append(entry)
    json.dump({"n": len(fes_detail),
               "description": "Test False-Early-Stop queries @ global t=0.5 (stopped before rerank without full gold coverage)",
               "queries": fes_detail},
              open(RESULTS / "fes_queries_test_t05.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---- report ----
    L = ["# Critic 阈值策略优化（validation-only 选择）", ""]
    L.append("> 规则：全部配置只在 val(1000q) 上评估与选择；选定配置在 test 上仅报告一次。"
             f"约束 Recall≥{RECALL_FLOOR}，目标最小化 Avg Cost。延迟含 critic 调用（{CRITIC_LAT:.1f}ms/次）。")
    L.append("")
    L.append("## 1. 全局阈值 sweep（val）")
    L.append("| 配置 | FinalRecall | FullCov | AvgCost(docs) | AvgLatency(ms) | FES | 停止比 d/h/r |")
    L.append("|---|---|---|---|---|---|---|")
    for c, m in rows:
        star = " **<-选中**" if c[0] == best_cfg[0] else ""
        L.append(fmt_row(f"t={c[1]:.2f}", m) + star)
    L.append("")
    L.append("## 2. 分阶段阈值 grid（val，仅列出满足 Recall≥%.2f 的前 10 个低成本配置）" % RECALL_FLOOR)
    L.append("| 配置 | FinalRecall | FullCov | AvgCost | AvgLatency(ms) | FES | 停止比 d/h/r |")
    L.append("|---|---|---|---|---|---|---|")
    shown = 0
    for (name, _, _), m in sorted(grid_rows, key=lambda cm: cm[1]["docs"]):
        if m["recall"] >= RECALL_FLOOR and shown < 10:
            star = " **<-选中**" if name == best_cfg[0] else ""
            L.append(fmt_row(name, m) + star)
            shown += 1
    L.append("")
    L.append("## 3. Val Pareto 前沿（Recall≥%.2f 约束下）" % RECALL_FLOOR)
    L.append("| 配置 | FinalRecall | AvgCost | AvgLatency |")
    L.append("|---|---|---|---|")
    for (name, _, _), m in front[:8]:
        L.append(f"| {name} | {m['recall']:.3f} | {m['docs']:.2f} | {m['lat']:.1f} |")
    sel_name = best_cfg[0]
    L.append("")
    L.append(f"**选定配置：{sel_name}**（val 满足约束的最小成本点）")
    L.append("")
    L.append("## 4. Test 最终对比（选择过程未使用 test）")
    L.append("| 系统 | FinalRecall | FullCov | AvgCost | AvgLatency(ms) | FES | 停止比 d/h/r |")
    L.append("|---|---|---|---|---|---|---|")
    L.append("| Fixed rerank@20 | 0.956 | 0.914 | 20.00 | 220.9 | 0.000 | 0/0/1000 |")
    L.append("| Oracle ES-3 | 0.959 | 0.919 | 6.18 | 72.8 | 0.000 | 684/146/170 |")
    L.append(fmt_row("Learned Critic t=0.5", test_05))
    L.append(fmt_row(f"Learned Critic {sel_name} (val-selected)", test_best))
    L.append("")
    # FES summary
    ns = {}
    for e in fes_detail:
        ns[e["stop_stage"]] = ns.get(e["stop_stage"], 0) + 1
    resc = sum(1 for e in fes_detail if e["would_continue_at_t08"])
    covs = [min(x["stages"]["rerank"]["coverage"], 1.0) for x in fes_detail]
    ps = [x["stages"][x["stop_stage"]]["critic_p"] for x in fes_detail]
    L.append("## 5. FES 诊断摘要（test @ t=0.5，共 %d 条，明细见 fes_queries_test_t05.json）" % len(fes_detail))
    L.append(f"- 停止阶段分布: dense={ns.get('dense',0)}, hybrid={ns.get('hybrid',0)}")
    L.append(f"- 停止时平均覆盖率: {np.mean([e['coverage_at_stop'] for e in fes_detail]):.3f}；"
             f"其中覆盖率=0.5 的占 {sum(1 for e in fes_detail if abs(e['coverage_at_stop']-0.5)<1e-9)/len(fes_detail):.1%}")
    L.append(f"- 停止点 critic 概率: mean={np.mean(ps):.3f}, min={np.min(ps):.3f}, max={np.max(ps):.3f}")
    L.append(f"- 若阈值提高到 0.80 可继续升级的: {resc}/{len(fes_detail)}")
    L.append(f"- 这批查询最终 rerank@20 覆盖率均值: {np.mean(covs):.3f}")
    txt = "\n".join(L) + "\n"
    open(RESULTS / "critic_policy.md", "w", encoding="utf-8").write(txt)

    out = {"selected": {"config": sel_name, "t_dense": best_cfg[1], "t_hybrid": best_cfg[2],
                        "val": {k: v for k, v in best_m.items() if k != '_detail'},
                        "test": {k: v for k, v in test_best.items() if k != '_detail'}},
           "global_sweep_val": [{"t": c[1], **{k: v for k, v in m.items() if k != '_detail'}}
                                for c, m in rows],
           "grid_val": [{"cfg": c[0], **{k: v for k, v in m.items() if k != '_detail'}}
                        for c, m in grid_rows],
           "fes_summary": {"n": len(fes_detail), "stops": ns, "rescued_at_t08": resc}}
    json.dump(out, open(RESULTS / "critic_policy.json", "w"), ensure_ascii=False, indent=1)

    # ---- Recall-Cost curve ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        xs = [m["docs"] for _, m in rows]
        ys = [m["recall"] for _, m in rows]
        plt.plot(xs, ys, "o-", label="global threshold (val)")
        gx = [m["docs"] for _, m in grid_rows]
        gy = [m["recall"] for _, m in grid_rows]
        plt.scatter(gx, gy, s=12, c="tab:orange", label="stage-specific grid (val)")
        plt.axhline(RECALL_FLOOR, ls="--", c="gray", lw=1)
        plt.axhline(best_m["recall"], ls=":", c="green", lw=1)
        plt.scatter([best_m["docs"]], [best_m["recall"]], marker="*", s=200, c="red",
                    zorder=5, label=f"selected ({sel_name})")
        plt.scatter([20.0], [0.949], marker="s", c="black", label="Fixed rerank@20 (val~test)")
        plt.xlabel("Avg docs retrieved"); plt.ylabel("Final Recall")
        plt.title("Recall-Cost (validation)")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(RESULTS / "recall_cost_curve.png", dpi=150)
        logger.info("curve saved")
    except Exception as e:
        logger.warning(f"plot skipped: {e}")

    print(txt)


if __name__ == "__main__":
    main()

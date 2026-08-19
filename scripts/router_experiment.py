"""
LLM Difficulty Router experiment (Phase 2)
==========================================
Protocol: EXPERIMENT_PROTOCOL.md (frozen 2026-08-18)

Systems compared:
  fixed:  dense@10, hybrid@10, rerank@10, rerank@20
  rule:   rule-based router  (easy->dense@3, medium->dense@5, hard->hybrid@20)
  llm:    Qwen2.5-7B-Instruct 4-bit router (same mapping)
  oracle: per-query cheapest full-coverage pick (upper bound)

Metrics per system: Recall@3/5/10/20, avg retrieval cost (docs + level),
router accuracy vs oracle labels, difficulty distribution.

Stages:
  python3 scripts/router_experiment.py --stage llm    # GPU1, writes results/router_llm_preds.json
  python3 scripts/router_experiment.py --stage eval   # CPU, reads preds if present
"""

import json, logging, re, sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

STRATEGIES = [("dense@3", "dense", 3), ("dense@5", "dense", 5), ("dense@10", "dense", 10),
              ("hybrid@5", "hybrid", 5), ("hybrid@10", "hybrid", 10), ("rerank@10", "rerank", 10),
              ("hybrid@20", "hybrid", 20), ("rerank@20", "rerank", 20)]
NAME2STRAT = {n: (m, k) for n, m, k in STRATEGIES}
MAPPING = {"easy": "dense@3", "medium": "dense@5", "hard": "hybrid@20"}
FIXED = ["dense@10", "hybrid@10", "rerank@10", "rerank@20"]
KS = [3, 5, 10, 20]
LLM_PREDS = RESULTS / "router_llm_preds.json"

PROMPT = ("Classify the following question by the difficulty of retrieving its answer "
          "documents from a large corpus.\n"
          "- easy: the answer documents are found within the top-3 results of a standard "
          "dense retriever.\n"
          "- medium: they are not in top-3, but within top-10 of some stronger retriever.\n"
          "- hard: the answer documents cannot be found even in top-10.\n"
          "Answer with exactly one word: easy, medium, or hard.\n\nQuestion: {}")


def load(ds, m):
    p = RESULTS / f"{ds}_{m}_v2m3.json" if m == "rerank" else RESULTS / f"{ds}_{m}.json"
    if not p.exists():
        p = RESULTS / f"{ds}_{m}.json"
    return json.load(open(p, encoding="utf-8"))["results"]


def oracle_label_and_pick(qid, gold, R):
    def best(m, g):
        rk = R[m][qid]["ranked"]
        rs = [i + 1 for i, d in enumerate(rk) if d in g]
        return min(rs) if rs else None
    every3 = every10 = True
    for g in gold:
        rs = [best(m, g) for m in ["dense", "hybrid", "rerank"]]
        rs = [r for r in rs if r is not None]
        if not rs or min(rs) > 3:
            every3 = False
        if not rs or min(rs) > 10:
            every10 = False
    label = "easy" if every3 else ("medium" if every10 else "hard")
    full = None
    bestc, besti = -1, 0
    for i, (n, m, k) in enumerate(STRATEGIES):
        rk = R[m][qid]["ranked"]
        c = len(set(rk[:k]) & gold) / len(gold)
        if c >= 1 and full is None:
            full = n
        if c > bestc or (c == bestc and i < besti):
            bestc, besti = c, i
    return label, (full if full is not None else STRATEGIES[besti][0])


def rule_predict(q):
    s = q.lower()
    words = re.findall(r"[a-z0-9]+", s)
    if len(words) >= 18:
        return "hard"
    if any(w in s for w in ["compare", "comparison", "both", "differ", "similar", "versus", "vs.", "unlike"]):
        return "medium"
    ents = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", q))
    if len(words) >= 10 and any(w in s for w in ["and", "or", "which", "whom"]) and ents >= 2:
        return "medium"
    return "easy"


def llm_predict_batch(questions):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = os.environ.get("LLM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    model.eval()
    texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q["question"])}],
                                     tokenize=False, add_generation_prompt=True)
             for q in questions]
    preds = {}
    raw = {}
    unparsed = 0
    for i in range(0, len(texts), 1):
        chunk = texts[i:i + 1]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
        dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for qi, d in enumerate(dec):
            raw[questions[i + qi]["qid"]] = d
            m = re.search(r"\b(easy|medium|hard)\b", d.lower())
            preds[questions[i + qi]["qid"]] = m.group(1) if m else "easy"
            if not m:
                unparsed += 1
        logger.info(f"llm {i + len(chunk)}/{len(texts)}")
    json.dump(raw, open(RESULTS / "router_llm_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    logger.info(f"llm done, unparsed fallback={unparsed}")
    return preds


def strat_system(pick_name, R, qids, gold_of):
    m, k = NAME2STRAT[pick_name]
    return {qid: R[m][qid]["ranked"][:k] for qid in qids}


def recall_at_k(gold, lst, k):
    return len(set(lst[:k]) & gold) / len(gold)


def evaluate(name, lists, qids, gold_of):
    n = len(qids)
    out = {"recall": {}, "cost_docs": None, "cost_level": None}
    for k in KS:
        out["recall"][k] = sum(recall_at_k(gold_of[q], lists[q], k) for q in qids) / n
    lens = [len(lists[q]) for q in qids]
    out["cost_docs"] = sum(lens) / n
    return out


def stage_llm(datasets):
    questions = []
    for ds in datasets:
        meta = json.load(open(ROOT / "data" / f"{ds}.json", encoding="utf-8"))
        qids = [r["qid"] for r in load(ds, "dense")]
        by_qid = {r["qid"]: r["question"] for r in meta["questions"]}
        questions += [{"ds": ds, "qid": q, "question": by_qid[q]} for q in qids]
    preds = llm_predict_batch(questions)
    json.dump(preds, open(LLM_PREDS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    logger.info(f"Saved {len(preds)} preds -> {LLM_PREDS}")


def stage_eval(datasets):
    llm_preds = json.load(open(LLM_PREDS, encoding="utf-8")) if LLM_PREDS.exists() else {}
    md = ["# Router 对比实验（Phase 2）", "> 协议: EXPERIMENT_PROTOCOL.md（2026-08-18 冻结）", ""]
    summary = {}
    for ds in datasets:
        R = {m: {r["qid"]: r for r in load(ds, m)} for m in ["dense", "hybrid", "rerank"]}
        qids = [r["qid"] for r in load(ds, "dense")]
        gold_of = {qid: set(R["dense"][qid]["gold"]) for qid in qids}
        n = len(qids)

        oracle_label = {}
        oracle_pick = {}
        for qid in qids:
            oracle_label[qid], oracle_pick[qid] = oracle_label_and_pick(qid, gold_of[qid], R)
        dist_oracle = Counter(oracle_label.values())

        meta = json.load(open(ROOT / "data" / f"{ds}.json", encoding="utf-8"))
        qtext = {r["qid"]: r["question"] for r in meta["questions"]}
        rule_pred = {qid: rule_predict(qtext[qid]) for qid in qids}
        llm_pred = {qid: llm_preds.get(qid, "easy") for qid in qids}

        systems = {}
        for f in FIXED:
            systems[f] = strat_system(f, R, qids, gold_of)
        systems["rule"] = {qid: strat_system(MAPPING[rule_pred[qid]], R, qids, gold_of)[qid]
                           for qid in qids}
        if llm_preds:
            systems["llm"] = {qid: strat_system(MAPPING[llm_pred[qid]], R, qids, gold_of)[qid]
                              for qid in qids}
        systems["oracle"] = {qid: strat_system(oracle_pick[qid], R, qids, gold_of)[qid]
                             for qid in qids}

        md.append(f"## {ds}（n={n}）")
        md.append("")
        md.append("### Difficulty 分布（oracle 标签 vs 预测）")
        md.append("| 来源 | easy | medium | hard |")
        md.append("|---|---|---|---|")
        for src, d in [("oracle", dist_oracle),
                       ("rule", Counter(rule_pred.values())),
                       ("llm", Counter(llm_pred.values()))]:
            md.append(f"| {src} | {d['easy']}({100 * d['easy'] / n:.1f}%) | "
                      f"{d['medium']}({100 * d['medium'] / n:.1f}%) | {d['hard']}({100 * d['hard'] / n:.1f}%) |")
        md.append("")

        md.append("### 对比总表")
        md.append("| 系统 | R@3 | R@5 | R@10 | R@20 | 平均成本(文档) | 成本级 | Router Acc |")
        md.append("|---|---|---|---|---|---|---|---|")
        lvl = {n: i + 1 for i, (n, _, _) in enumerate(STRATEGIES)}
        rows = {}
        for name, lists in systems.items():
            ev = evaluate(name, lists, qids, gold_of)
            if name in NAME2STRAT:
                cost_level = lvl[name]
            else:
                if name == "oracle":
                    lv = [lvl[oracle_pick[q]] for q in qids]
                elif name == "rule":
                    lv = [lvl[MAPPING[rule_pred[q]]] for q in qids]
                else:
                    lv = [lvl[MAPPING[llm_pred[q]]] for q in qids]
                cost_level = sum(lv) / n
            acc = "-"
            if name in ("rule", "llm"):
                src = rule_pred if name == "rule" else llm_pred
                acc = f"{sum(src[q] == oracle_label[q] for q in qids) / n:.3f}"
            rows[name] = {"recall": ev["recall"], "cost_docs": ev["cost_docs"],
                          "cost_level": cost_level, "acc": acc}
            md.append(f"| {name} | " + " | ".join(
                f"{ev['recall'][k]:.3f}" for k in KS) +
                f" | {ev['cost_docs']:.2f} | {cost_level if isinstance(cost_level, int) else cost_level:.2f} | {acc} |")
        md.append("")
        summary[ds] = {"systems": rows, "dist_oracle": dict(dist_oracle),
                       "dist_rule": dict(Counter(rule_pred.values())),
                       "dist_llm": dict(Counter(llm_pred.values()))}

    with open(RESULTS / "router_eval.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(RESULTS / "router_eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved -> results/router_eval.md, results/router_eval.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["llm", "eval"])
    ap.add_argument("--datasets", default="nq,hotpotqa")
    args = ap.parse_args()
    dss = args.datasets.split(",")
    if args.stage == "llm":
        stage_llm(dss)
    else:
        stage_eval(dss)
"""
Evaluate Learned Retrieval Sufficiency Critic (offline + E2E ladder)
====================================================================
Offline:  Critic P/R/F1 per stage and overall on test samples
E2E:      Dense@3 -> Critic -> Hybrid@5 -> Critic -> Rerank@20 -> Critic
          (final stage always stops)
Metrics:  False Early Stop Rate, Final Recall, FullCov, avg docs, avg latency,
          stage stop ratios; compared vs Fixed rerank@20 and Oracle ES-3.

Latency model (measured 2026-08-19, GPU1, ms): S1=31.5, S2=93.5, S3=220.9,
plus critic inference time per stage call (measured live here).

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_critic.py
Output: results/critic_eval.md + results/critic_eval.json
"""

import json, logging, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "data" / "critic_samples"
RESULTS = ROOT / "results"
MODEL_DIR = ROOT / "models"

MAX_LEN = 512
STAGE_LAT = {"dense": 31.5, "hybrid": 93.5, "rerank": 220.9}
STAGE_ORDER = ["dense", "hybrid", "rerank"]
LADDER = [("dense", 3), ("hybrid", 5), ("rerank", 20)]


class DS(Dataset):
    def __init__(self, path, tok):
        self.tok = tok
        self.items = [json.loads(l) for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return it["question"] + " [SEP] " + it["docs"], it["label"]


def predict_probs(model, tok, items, batch=64):
    model.eval()
    probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(items), batch):
            texts = [it["question"] + " [SEP] " + it["docs"] for it in items[i:i + batch]]
            enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            logits = model(**enc).logits
            probs += torch.softmax(logits, -1)[:, 1].cpu().tolist()
    lat = (time.perf_counter() - t0) * 1000 / len(items)
    return np.array(probs), lat


def main():
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base", local_files_only=True)
    model.classifier.out_proj = torch.nn.Linear(model.config.hidden_size, 2)
    model.num_labels = 2
    model.load_state_dict(torch.load(MODEL_DIR / "critic.pt", map_location="cpu"))
    model.to("cuda").eval()

    test_items = [json.loads(l) for l in open(SAMPLES / "test.jsonl", encoding="utf-8")]
    train_qids = {json.loads(l)["qid"] for l in open(SAMPLES / "train.jsonl", encoding="utf-8")}
    val_qids = {json.loads(l)["qid"] for l in open(SAMPLES / "val.jsonl", encoding="utf-8")}
    leak = [it["qid"] for it in test_items if it["qid"] in train_qids | val_qids]
    assert not leak, f"LEAKAGE: {len(leak)} test qids in train/val!"

    probs, critic_lat = predict_probs(model, tok, test_items)
    for it, p in zip(test_items, probs):
        it["p"] = float(p)

    gold = json.load(open(SAMPLES / "test_gold.json"))
    by_qid = {}
    for it in test_items:
        by_qid.setdefault(it["qid"], {})[it["phase"]] = it

    # ---- offline critic metrics ----
    md = ["# Learned Retrieval Sufficiency Critic 评估", "> 数据: HotpotQA test 1000 queries × 3 stages（test 未参与训练/选择）", ""]
    md.append("## D. Critic 离线指标（test, threshold=0.5）")
    md.append("| 阶段 | N | suff率 | P | R | F1 | Acc |")
    md.append("|---|---|---|---|---|---|---|")
    crit = {}
    for ph in STAGE_ORDER:
        items = [it for it in test_items if it["phase"] == ph]
        tp = sum(1 for it in items if it["p"] >= 0.5 and it["label"] == 1)
        fp = sum(1 for it in items if it["p"] >= 0.5 and it["label"] == 0)
        fn = sum(1 for it in items if it["p"] < 0.5 and it["label"] == 1)
        tn = sum(1 for it in items if it["p"] < 0.5 and it["label"] == 0)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        acc = (tp + tn) / len(items)
        crit[ph] = {"p": p, "r": r, "f1": f1, "acc": acc}
        md.append(f"| {ph}@{dict(LADDER)[ph]} | {len(items)} | {sum(it['label'] for it in items) / len(items):.3f} | {p:.3f} | {r:.3f} | {f1:.3f} | {acc:.3f} |")
    md.append("")

    # ---- E2E ladder ----
    def coverage(qid, ph):
        return set(by_qid[qid][ph]["ranked"]) >= set(gold[qid])

    stop = {}
    for qid in by_qid:
        s = None
        for ph, k in LADDER:
            s = ph
            if by_qid[qid][ph]["p"] >= 0.5:
                break
        stop[qid] = s

    from collections import Counter
    dist = Counter(stop.values())
    n = len(by_qid)
    fes = sum(1 for qid in by_qid if stop[qid] != "rerank" and not coverage(qid, stop[qid]))
    fes_rate = fes / n
    rec = sum(len(set(by_qid[qid][stop[qid]]["ranked"]) & set(gold[qid])) / len(gold[qid]) for qid in by_qid) / n
    full = sum(1 for qid in by_qid if coverage(qid, stop[qid])) / n
    docs = sum(dict(LADDER)[stop[qid]] for qid in by_qid) / n
    stages_run = sum(STAGE_ORDER.index(stop[qid]) + 1 for qid in by_qid) / n
    lat = sum(STAGE_LAT[stop[qid]] for qid in by_qid) / n + critic_lat * stages_run

    # oracle ES-3 (same 3-stage ladder)
    def oracle_stop(qid):
        for ph, k in LADDER:
            if set(by_qid[qid][ph]["ranked"]) >= set(gold[qid]):
                return ph
        return "rerank"
    os_ = {qid: oracle_stop(qid) for qid in by_qid}
    ore = sum(len(set(by_qid[qid][os_[qid]]["ranked"]) & set(gold[qid])) / len(gold[qid]) for qid in by_qid) / n
    ofc = sum(1 for qid in by_qid if coverage(qid, os_[qid])) / n
    odocs = sum(dict(LADDER)[os_[qid]] for qid in by_qid) / n
    olat = sum(STAGE_LAT[os_[qid]] for qid in by_qid) / n

    md.append("## E. End-to-End：Fixed vs Oracle ES-3 vs Learned Critic（test n=1000）")
    md.append("| 系统 | FinalRecall | FullCov | Avg文档成本 | Avg延迟(ms) | FalseEarlyStop | 阶段停止比(d/h/r) |")
    md.append("|---|---|---|---|---|---|---|")
    fixed_rec = sum(len(set(by_qid[q]["rerank"]["ranked"]) & set(gold[q])) / len(gold[q]) for q in by_qid) / n
    fixed_full = sum(1 for q in by_qid if coverage(q, "rerank")) / n
    md.append(f"| Fixed rerank@20 | {fixed_rec:.3f} | {fixed_full:.3f} | 20.00 | {STAGE_LAT['rerank']:.1f} | 0.000 | 0/0/1000 |")
    od = Counter(os_.values())
    md.append(f"| Oracle ES-3 | {ore:.3f} | {ofc:.3f} | {odocs:.2f} | {olat:.1f} | 0.000 | {od['dense']}/{od['hybrid']}/{od['rerank']} |")
    md.append(f"| Learned Critic | {rec:.3f} | {full:.3f} | {docs:.2f} | {lat:.1f} | {fes_rate:.3f} | {dist['dense']}/{dist['hybrid']}/{dist['rerank']} |")
    md.append("")
    md.append(f"> Critic 单次推理: {critic_lat:.1f} ms/query（含 tokenize）。延迟含 critic 调用次数。")
    md.append("")

    # ---- gap analysis ----
    md.append("## F. 与 Oracle 的差距")
    md.append(f"- Recall 差距: oracle {ore:.3f} vs critic {rec:.3f} = **{ore - rec:+.3f}**")
    md.append(f"- FullCov 差距: oracle {ofc:.3f} vs critic {full:.3f} = **{ofc - full:+.3f}**")
    md.append(f"- 成本差距: oracle {odocs:.2f} vs critic {docs:.2f} docs")
    md.append(f"- False Early Stop: oracle 0 按构造; critic {fes} 个查询误停（{fes_rate:.1%}）")
    md.append(f"- 误停造成的 Recall 损失上限: 这些查询在误停点平均覆盖率 = "
              f"{np.mean([len(set(by_qid[q][stop[q]]['ranked']) & set(gold[q])) / len(gold[q]) for q in by_qid if stop[q] != 'rerank' and not coverage(q, stop[q])] or [0]):.3f}")

    with open(RESULTS / "critic_eval.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    out = {"offline": crit, "e2e": {"fixed": {"recall": fixed_rec, "full": fixed_full},
                                    "oracle": {"recall": ore, "full": ofc, "docs": odocs, "lat": olat},
                                    "critic": {"recall": rec, "full": full, "docs": docs, "lat": lat,
                                               "fes": fes_rate, "stop": dict(dist)},
                                    "critic_lat_ms": critic_lat}}
    json.dump(out, open(RESULTS / "critic_eval.json", "w"), ensure_ascii=False, indent=2)
    logger.info("Saved -> results/critic_eval.md, results/critic_eval.json")


if __name__ == "__main__":
    main()

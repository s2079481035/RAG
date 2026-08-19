"""
Train lightweight Retrieval Sufficiency Critic
===============================================
Model: BAAI/bge-reranker-base (cross-encoder, 280M) -> binary classification
       SUFFICIENT=1 / INSUFFICIENT=0
Input:  query + concatenated retrieved docs (per-doc truncated to 48 words,
        total max 512 tokens)
Data:   data/critic_samples/{train,val}.jsonl (HotpotQA non-test questions)
Split:  train 5405 q x 3 stages, val 1000 q x 3 stages. NO test data here.

Usage (GPU1):
  CUDA_VISIBLE_DEVICES=1 python3 scripts/train_critic.py [--epochs 4] [--batch 16]
Saves best (val F1) -> models/critic.pt (+ critic_config.json)
"""

import argparse, json, logging, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "data" / "critic_samples"
MODEL_DIR = ROOT / "models"
MODEL_NAME = "BAAI/bge-reranker-base"
MAX_LEN = 512
DOC_MAX_WORDS = 48


class SampleDS(Dataset):
    def __init__(self, path, tok):
        self.tok = tok
        self.items = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.items = [i for i in self.items if len(i["docs"].split()) > 0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return it["question"] + " [SEP] " + it["docs"], it["label"]


def collate(batch, tok):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch])
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc, labels


def evaluate(model, dl, device):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for enc, labels in dl:
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            preds = logits.argmax(-1).cpu()
            for p, l in zip(preds, labels):
                if p == 1 and l == 1: tp += 1
                elif p == 1 and l == 0: fp += 1
                elif p == 0 and l == 1: fn += 1
                else: tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {"p": prec, "r": rec, "f1": f1, "acc": acc, "n": tp + fp + fn + tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    model.classifier.out_proj = torch.nn.Linear(model.config.hidden_size, 2)
    model.num_labels = 2
    model.to(device)
    train_ds = SampleDS(SAMPLES / "train.jsonl", tok)
    val_ds = SampleDS(SAMPLES / "val.jsonl", tok)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                          collate_fn=lambda b: collate(b, tok))
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=lambda b: collate(b, tok))
    logger.info(f"train={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=steps_per_epoch // 5,
                                            num_training_steps=steps_per_epoch * args.epochs)
    MODEL_DIR.mkdir(exist_ok=True)
    best_f1, best_ep = 0.0, -1
    for ep in range(args.epochs):
        model.train()
        tot, corr = 0, 0
        for bi, (enc, labels) in enumerate(train_dl):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            logits = model(**enc).logits
            loss = cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += labels.numel()
            corr += (logits.argmax(-1) == labels).sum().item()
            if bi % 200 == 0:
                logger.info(f"ep{ep} b{bi}/{steps_per_epoch} loss={loss.item():.4f} acc={corr / tot:.3f}")
        vm = evaluate(model, val_dl, device)
        logger.info(f"ep{ep} val={vm}")
        if vm["f1"] > best_f1:
            best_f1, best_ep = vm["f1"], ep
            torch.save(model.state_dict(), MODEL_DIR / "critic.pt")
    torch.save(model.state_dict(), MODEL_DIR / "critic_final.pt")
    json.dump({"model": MODEL_NAME, "best_epoch": best_ep, "val_f1": best_f1,
               "max_len": MAX_LEN, "doc_max_words": DOC_MAX_WORDS},
              open(MODEL_DIR / "critic_config.json", "w"), indent=2)
    logger.info(f"best val f1={best_f1:.4f} @ep{best_ep} -> models/critic.pt")


if __name__ == "__main__":
    main()

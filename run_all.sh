#!/bin/bash
# Retrieval-Budget Pilot: Phase 1-3 × NQ/HotpotQA
set -e
cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
PY=/usr/bin/python3.12

echo "== Phase 1: Dense =="
$PY scripts/retrieve.py --dataset nq --phase dense --ks 3,5,10,20
$PY scripts/retrieve.py --dataset hotpotqa --phase dense --ks 3,5,10,20

echo "== Phase 2: Hybrid RRF =="
$PY scripts/retrieve.py --dataset nq --phase hybrid --ks 3,5,10,20
$PY scripts/retrieve.py --dataset hotpotqa --phase hybrid --ks 3,5,10,20

echo "== Phase 3: Hybrid + Rerank =="
$PY scripts/retrieve.py --dataset nq --phase rerank --ks 3,5,10
$PY scripts/retrieve.py --dataset hotpotqa --phase rerank --ks 3,5,10

echo "== Evaluate =="
$PY scripts/eval_recall.py
echo "ALL DONE"
"""
Minimal RAG Generation Evaluation
==================================
计算 EM (Exact Match) 和 Token-level F1。

Usage:
  python3 scripts/eval_generation.py --input results/hotpotqa_rerank_k10_generated.json

输出:
  - 控制台: 整体 EM/F1
  - results/{dataset}_{method}_k{k}_eval.json: 详细结果
"""

import json, re, logging, argparse
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def normalize_answer(s):
    """归一化答案：小写、去冠词、去标点、去多余空格"""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def compute_em(predicted, gold):
    """计算 Exact Match"""
    return int(normalize_answer(predicted) == normalize_answer(gold))


def compute_f1(predicted, gold):
    """计算 Token-level F1"""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    # 计算共同 token
    common = Counter(pred_tokens) & Counter(gold_tokens)
    common_count = sum(common.values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(pred_tokens)
    recall = common_count / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Generation")
    parser.add_argument("--input", required=True, help="生成结果 JSON 文件路径")
    args = parser.parse_args()

    # 加载生成结果
    logger.info(f"Loading {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples")

    # 计算指标
    em_scores = []
    f1_scores = []
    details = []

    for item in data:
        em = compute_em(item["generated"], item["gold"])
        f1 = compute_f1(item["generated"], item["gold"])

        em_scores.append(em)
        f1_scores.append(f1)

        details.append({
            "qid": item["qid"],
            "question": item["question"],
            "generated": item["generated"],
            "gold": item["gold"],
            "em": em,
            "f1": f1
        })

    # 汇总
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # 输出结果
    print(f"\n{'='*50}")
    print(f"Generation Evaluation Results")
    print(f"{'='*50}")
    print(f"Input:  {args.input}")
    print(f"N:      {len(data)}")
    print(f"EM:     {avg_em:.4f}")
    print(f"F1:     {avg_f1:.4f}")
    print(f"{'='*50}\n")

    # 保存详细结果
    out_path = args.input.replace("_generated.json", "_eval.json")
    eval_result = {
        "input": args.input,
        "n": len(data),
        "em": avg_em,
        "f1": avg_f1,
        "details": details
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()

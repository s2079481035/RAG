"""
Minimal RAG Generation
======================
读取检索结果，调用 Qwen2.5-7B 生成答案。

Usage (GPU1):
  CUDA_VISIBLE_DEVICES=1 python3 scripts/generate.py \
    --dataset hotpotqa --method rerank --k 10 --max-samples 100

输出: results/{dataset}_{method}_k{k}_generated.json
"""

import json, logging, re, argparse, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 128
MAX_CONTEXT_LENGTH = 2048
DOC_MAX_WORDS = 100

PROMPT_TEMPLATE = """Answer the question based on the context below. 
If the context does not contain enough information to answer the question, output "cannot determine".
Answer in English only.

Context: {context}

Question: {question}

Answer:"""


def load_kb(dataset):
    """加载知识库"""
    data = json.load(open(DATA / f"{dataset}.json", encoding="utf-8"))
    return data["kb"]


def load_questions(dataset):
    """加载问题+答案"""
    data = json.load(open(DATA / f"{dataset}.json", encoding="utf-8"))
    return {q["qid"]: q for q in data["questions"]}


def load_retrieval(dataset, method, k):
    """加载检索结果"""
    suffix = "_v2m3" if method == "rerank" else ""
    path = RESULTS / f"{dataset}_{method}{suffix}.json"
    data = json.load(open(path, encoding="utf-8"))
    return data["results"]


def trunc_words(text, n):
    """截断到 n 个词"""
    return " ".join(text.split()[:n])


def build_context(kb, doc_ids, max_words=DOC_MAX_WORDS):
    """构建上下文，每篇文档截断"""
    docs = []
    for doc_id in doc_ids:
        text = kb.get(doc_id, "")
        if text:
            docs.append(trunc_words(text, max_words))
    return "\n\n".join(docs)


def load_qwen():
    """加载 Qwen 模型"""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.environ.get("LLM_MODEL_PATH", MODEL_NAME)
    logger.info(f"Loading Qwen from {model_path}...")

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, local_files_only=True
    )
    model.eval()

    logger.info("Qwen loaded successfully")
    return model, tok


def generate_answer(model, tokenizer, prompt):
    """生成单个答案"""
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    enc = tokenizer(
        [text], return_tensors="pt", padding=True, truncation=True,
        max_length=MAX_CONTEXT_LENGTH
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.batch_decode(
        output[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]

    return generated.strip()


def main():
    parser = argparse.ArgumentParser(description="Minimal RAG Generation")
    parser.add_argument("--dataset", required=True, choices=["nq", "hotpotqa"])
    parser.add_argument("--method", required=True, choices=["dense", "hybrid", "rerank"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    # 1. 加载数据
    logger.info(f"Loading data for {args.dataset}...")
    kb = load_kb(args.dataset)
    questions = load_questions(args.dataset)
    retrieval_results = load_retrieval(args.dataset, args.method, args.k)

    # 2. 限制样本数
    if args.max_samples:
        retrieval_results = retrieval_results[:args.max_samples]

    logger.info(f"KB: {len(kb)} docs, Questions: {len(questions)}, "
                f"Retrieval results: {len(retrieval_results)}")

    # 3. 加载 Qwen
    model, tokenizer = load_qwen()

    # 4. 生成
    generated_results = []
    start_time = time.time()

    for i, r in enumerate(retrieval_results):
        qid = r["qid"]
        question = questions.get(qid, {}).get("question", "")
        gold = questions.get(qid, {}).get("answer", "")

        # 获取检索到的文档
        retrieved_ids = r["ranked"][:args.k]

        # 构建上下文
        context = build_context(kb, retrieved_ids)

        # 构建 prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # 生成答案
        answer = generate_answer(model, tokenizer, prompt)

        generated_results.append({
            "qid": qid,
            "question": question,
            "generated": answer,
            "gold": gold,
            "retrieved": retrieved_ids
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            logger.info(f"Generated {i + 1}/{len(retrieval_results)} "
                       f"({speed:.2f} q/s, elapsed: {elapsed:.1f}s)")

    # 5. 保存结果
    out_path = RESULTS / f"{args.dataset}_{args.method}_k{args.k}_generated.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(generated_results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    logger.info(f"Done! Generated {len(generated_results)} answers in {total_time:.1f}s")
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

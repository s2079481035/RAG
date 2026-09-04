"""Generate frozen Dev/Test HotpotQA answers from Phase 2 chunk retrieval."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from experiment_utils import collect_environment, portable_path, set_global_seed, utc_now, write_json_atomic
from phase2_controller_inputs import evidence_key
from phase3a_generation_utils import extract_short_answer, file_sha256, load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3a" / "generation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model", help="Local model path; defaults to LLM_MODEL_PATH or config")
    parser.add_argument("--approval", type=Path, help="Required approved Dev-audit manifest for test")
    parser.add_argument("--max-samples", type=int, help="Debug-only limit; forbidden on test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_test_gate(
    args: argparse.Namespace, config_hash: str, config: dict, model_path: str
) -> dict | None:
    if args.split != "test":
        return None
    if args.max_samples is not None:
        raise ValueError("Test generation cannot use --max-samples")
    if args.approval is None or not args.approval.exists():
        raise ValueError("Test generation requires the completed Dev manual-audit approval manifest")
    approval = json.loads(args.approval.read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise ValueError("Dev generation audit is not approved")
    if approval.get("generation_config_sha256") != config_hash:
        raise ValueError("Generation config changed after Dev audit approval")
    if int(approval.get("reviewed_samples", 0)) < int(config["manual_dev_audit_count"]):
        raise ValueError("Dev generation audit reviewed too few samples")
    if approval.get("model") != model_path:
        raise ValueError("Generation model changed after Dev audit approval")
    audit_path = Path(approval["audit"])
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path
    if not audit_path.exists() or file_sha256(audit_path) != approval.get("audit_sha256_after_review"):
        raise ValueError("Reviewed Dev audit changed after approval")
    dev_manifest_path = Path(approval["dev_generation_manifest"])
    if not dev_manifest_path.is_absolute():
        dev_manifest_path = ROOT / dev_manifest_path
    if (
        not dev_manifest_path.exists()
        or file_sha256(dev_manifest_path) != approval.get("dev_generation_manifest_sha256")
    ):
        raise ValueError("Dev generation manifest changed after approval")
    return approval


def render_chunk(chunk: dict) -> str:
    return f"TITLE: {chunk['document_title']}\nTEXT: {chunk['chunk_text']}"


def chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def prompt_token_count(tokenizer, prompt: str) -> int:
    return len(tokenizer(chat_prompt(tokenizer, prompt), add_special_tokens=False)["input_ids"])


def fit_prompt(tokenizer, config: dict, question: str, chunks: list[dict]) -> tuple[str, list[str], bool]:
    template = config["prompt_template"]
    max_tokens = int(config["max_input_tokens"])
    empty_prompt = template.format(context="", question=question)
    if prompt_token_count(tokenizer, empty_prompt) >= max_tokens:
        raise ValueError("Question and fixed prompt exceed generation input budget")
    blocks = []
    used_ids = []
    truncated = False
    for chunk in chunks:
        block = render_chunk(chunk)
        candidate_blocks = [*blocks, block]
        candidate = template.format(context="\n\n".join(candidate_blocks), question=question)
        if prompt_token_count(tokenizer, candidate) <= max_tokens:
            blocks = candidate_blocks
            used_ids.append(chunk["chunk_id"])
            continue
        truncated = True
        low, high = 0, len(block)
        while low < high:
            middle = (low + high + 1) // 2
            partial = template.format(
                context="\n\n".join([*blocks, block[:middle]]), question=question
            )
            if prompt_token_count(tokenizer, partial) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        if low:
            blocks.append(block[:low])
            used_ids.append(chunk["chunk_id"])
        break
    prompt = template.format(context="\n\n".join(blocks), question=question)
    return prompt, used_ids, truncated or len(used_ids) < len(chunks)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_hash = file_sha256(args.config)
    model_path = args.model or os.environ.get("LLM_MODEL_PATH") or config["model"]
    approval = validate_test_gate(args, config_hash, config, model_path)
    output_dir = (args.output_dir or ROOT / "results" / "phase3" / "generation" / args.split).resolve()
    predictions_path = output_dir / "generated.jsonl"
    manifest_path = output_dir / "generation_manifest.json"
    existing = [path for path in [predictions_path, manifest_path] if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite generation outputs: {existing}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_global_seed(int(config["seed"]))
    load_kwargs = {"local_files_only": not args.allow_download}
    tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        **load_kwargs,
    )
    model.eval()
    if approval and approval.get("model_revision") != getattr(model.config, "_commit_hash", None):
        raise ValueError("Generation model revision changed after Dev audit approval")

    variant = config["variant"]
    chunks = read_jsonl(ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    source_path = ROOT / "data" / "phase2" / "controller" / variant / f"{args.split}.jsonl"
    records = [row for row in read_jsonl(source_path) if row["stage"] == config["stage"]]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    start = time.time()
    for index, record in enumerate(records, start=1):
        view = record[evidence_key(config["evidence_mode"])]
        evidence_chunks = [chunk_by_id[item["chunk_id"]] for item in view["items"]]
        prompt, used_chunk_ids, context_truncated = fit_prompt(
            tokenizer, config, record["question"], evidence_chunks
        )
        rendered = chat_prompt(tokenizer, prompt)
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=int(config["max_new_tokens"]),
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_output = tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        generated.append(
            {
                "question_id": record["question_id"],
                "split": args.split,
                "stage": record["stage"],
                "question": record["question"],
                "gold_answer": record["answer"],
                "raw_output": raw_output,
                "extracted_answer": extract_short_answer(raw_output),
                "prompt_tokens": int(encoded["input_ids"].shape[1]),
                "output_tokens": len(tokenizer(raw_output, add_special_tokens=False)["input_ids"]),
                "raw_output_lines": len([line for line in raw_output.splitlines() if line.strip()]),
                "context_truncated": context_truncated,
                "available_evidence_chunk_ids": [item["chunk_id"] for item in view["items"]],
                "used_evidence_chunk_ids": used_chunk_ids,
            }
        )
        if index % 10 == 0:
            logger.info("generated %s/%s (%.3f q/s)", index, len(records), index / (time.time() - start))
    write_jsonl_atomic(predictions_path, generated)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3A",
            "split": args.split,
            "samples": len(generated),
            "source": portable_path(source_path, ROOT),
            "output": portable_path(predictions_path, ROOT),
            "generation_config": portable_path(args.config.resolve(), ROOT),
            "generation_config_sha256": config_hash,
            "model": model_path,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "decoding": config["decoding"],
            "max_input_tokens": config["max_input_tokens"],
            "max_new_tokens": config["max_new_tokens"],
            "answer_extraction": config["answer_extraction"],
            "normalization": config["normalization"],
            "test_gate_approval": portable_path(args.approval.resolve(), ROOT) if approval else None,
            "environment": collect_environment(ROOT),
        },
    )
    logger.info("saved %s generations -> %s", len(generated), predictions_path)


if __name__ == "__main__":
    main()

"""Generate one frozen answer per Phase 3B question and retrieval stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from evidence_utils import supporting_fact_metrics
from experiment_utils import collect_environment, git_commit, portable_path, set_global_seed, utc_now, write_json_atomic
from generate_phase3a import chat_prompt, fit_prompt
from phase2_controller_inputs import evidence_key
from phase3a_generation_utils import exact_match_score, extract_short_answer, load_config, token_f1_score
from train_phase2_controller import read_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", required=True, choices=["dev_policy", "test"])
    parser.add_argument("--model", help="Local generation model path")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def validate_test_gate(config: dict, config_path: Path) -> dict | None:
    gate_path = repo_path(config["test"]["gate"])
    if not gate_path.exists():
        raise ValueError("Phase 3B Test generation requires the frozen Test gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "frozen" or gate.get("test_consulted") is not False:
        raise ValueError("Phase 3B Test gate is not a pre-Test frozen gate")
    if gate.get("config_sha256") != file_sha256(config_path):
        raise ValueError("Phase 3B protocol changed after the Test gate")
    for key, relative in [
        ("selected_thresholds_sha256", "results/phase3b/selected_thresholds.json"),
        ("temperature_scaling_sha256", "results/phase3b/temperature_scaling.json"),
        ("base_controller_sha256", "results/phase3b/base_controller.json"),
    ]:
        if gate.get(key) != file_sha256(ROOT / relative):
            raise ValueError(f"Frozen Test-gate artifact changed: {relative}")
    return gate


def timed_generate(model, encoded: dict, generation_config: dict, torch_module) -> tuple[object, float]:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    started = time.perf_counter()
    with torch_module.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=int(generation_config["max_new_tokens"]),
            do_sample=False,
            num_beams=1,
            pad_token_id=model.config.eos_token_id,
        )
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    generation_settings = config["generation"]
    frozen_config_path = repo_path(generation_settings["frozen_config"])
    if file_sha256(frozen_config_path) != generation_settings["expected_config_sha256"]:
        raise ValueError("Frozen Phase 3A generation configuration changed")
    generation_config = load_config(frozen_config_path)
    approval_path = repo_path(generation_settings["phase3a_dev_approval"])
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval["status"] != "approved" or approval["generation_config_sha256"] != file_sha256(frozen_config_path):
        raise ValueError("Phase 3A generation approval does not match the frozen configuration")
    test_gate = validate_test_gate(config, args.config) if args.split == "test" else None

    model_path = args.model or os.environ.get("LLM_MODEL_PATH") or generation_config["model"]
    if model_path != approval["model"]:
        raise ValueError("Phase 3B must use the exact Phase 3A-approved model path")
    variant = config["base_controller"]["variant"]
    source_split = "dev" if args.split == "dev_policy" else "test"
    source_path = ROOT / "data" / "phase2" / "controller" / variant / f"{source_split}.jsonl"
    records = read_jsonl(source_path)
    if args.split == "dev_policy":
        policy_ids_path = repo_path(config["dev_split"]["policy_ids"])
        policy_ids = read_ids(policy_ids_path)
        records = [row for row in records if row["question_id"] in policy_ids]
    expected_questions = int(config["dev_split"]["policy_questions"]) if args.split == "dev_policy" else 1000
    stage_order = [row["name"] for row in config["stages"]]
    if len(records) != expected_questions * len(stage_order):
        raise ValueError(f"Expected {expected_questions * len(stage_order)} stage records, got {len(records)}")

    output_path = repo_path(generation_settings[f"stage_cache_{args.split}"])
    manifest_path = output_path.with_name("generation_manifest.json")
    existing = [path for path in [output_path, manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3B stage generations: {existing}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_global_seed(int(generation_config["seed"]))
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
    if approval.get("model_revision") != getattr(model.config, "_commit_hash", None):
        raise ValueError("Generation model revision differs from the Phase 3A-approved model")
    chunks = read_jsonl(ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}

    first = records[0]
    first_view = first[evidence_key(config["base_controller"]["evidence_mode"])]
    first_chunks = [chunk_by_id[item["chunk_id"]] for item in first_view["items"]]
    warm_prompt, _, _ = fit_prompt(
        tokenizer, generation_config, first["question"], first_chunks
    )
    warm_encoded = tokenizer(
        chat_prompt(tokenizer, warm_prompt), return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    for _ in range(int(config["cost_measurement"]["warmup_questions"])):
        timed_generate(model, warm_encoded, generation_config, torch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    started_all = time.perf_counter()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            view = record[evidence_key(config["base_controller"]["evidence_mode"])]
            evidence_chunks = [chunk_by_id[item["chunk_id"]] for item in view["items"]]
            prompt, used_chunk_ids, context_truncated = fit_prompt(
                tokenizer, generation_config, record["question"], evidence_chunks
            )
            rendered = chat_prompt(tokenizer, prompt)
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
            output, generation_latency_ms = timed_generate(
                model, encoded, generation_config, torch
            )
            raw_output = tokenizer.decode(
                output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            extracted = extract_short_answer(raw_output)
            visible_metrics = supporting_fact_metrics(
                used_chunk_ids, chunk_by_id, record["gold_supporting_facts"]
            )
            row = {
                "question_id": record["question_id"],
                "source_split": source_split,
                "phase3b_split": args.split,
                "stage": record["stage"],
                "stage_index": int(record["stage_index"]) + 1,
                "question": record["question"],
                "gold_answer": record["answer"],
                "raw_output": raw_output,
                "extracted_answer": extracted,
                "exact_match": exact_match_score(extracted, record["answer"]),
                "token_f1": token_f1_score(extracted, record["answer"]),
                "prompt_tokens": int(encoded["input_ids"].shape[1]),
                "output_tokens": len(tokenizer(raw_output, add_special_tokens=False)["input_ids"]),
                "raw_output_lines": len([line for line in raw_output.splitlines() if line.strip()]),
                "context_truncated": context_truncated,
                "generation_latency_ms": generation_latency_ms,
                "available_evidence_chunk_ids": [item["chunk_id"] for item in view["items"]],
                "used_evidence_chunk_ids": used_chunk_ids,
                "supporting_fact_recall": float(view["supporting_fact_recall"]),
                "complete_evidence_coverage": int(view["complete_evidence_coverage"]),
                "evidence_state": view["evidence_state"],
                "model_visible_supporting_fact_recall": float(
                    visible_metrics["supporting_fact_recall"]
                ),
                "model_visible_complete_evidence_coverage": int(
                    visible_metrics["complete_evidence_coverage"]
                ),
                "model_visible_evidence_state": visible_metrics["evidence_state"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                elapsed = time.perf_counter() - started_all
                logger.info("generated %s/%s stage answers (%.3f/s)", index, len(records), index / elapsed)
    temporary.replace(output_path)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": "3B",
            "git_commit": git_commit(ROOT),
            "split": args.split,
            "source_split": source_split,
            "questions": expected_questions,
            "stage_generations": len(records),
            "stages": stage_order,
            "source": portable_path(source_path, ROOT),
            "output": portable_path(output_path, ROOT),
            "output_sha256": file_sha256(output_path),
            "phase3b_config": portable_path(args.config.resolve(), ROOT),
            "phase3b_config_sha256": file_sha256(args.config),
            "frozen_generation_config": portable_path(frozen_config_path, ROOT),
            "frozen_generation_config_sha256": file_sha256(frozen_config_path),
            "phase3a_approval_sha256": file_sha256(approval_path),
            "phase3b_test_gate_sha256": file_sha256(repo_path(config["test"]["gate"])) if test_gate else None,
            "model": model_path,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "decoding": generation_config["decoding"],
            "answer_extraction": generation_config["answer_extraction"],
            "normalization": generation_config["normalization"],
            "warmup_questions": int(config["cost_measurement"]["warmup_questions"]),
            "generation_latency_cuda_synchronized": bool(torch.cuda.is_available()),
            "environment": collect_environment(ROOT),
        },
    )
    logger.info("saved %s stage generations -> %s", len(records), output_path)


if __name__ == "__main__":
    main()

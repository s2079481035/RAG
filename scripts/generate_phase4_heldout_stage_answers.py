"""Generate one frozen answer for every 2Wiki heldout retrieval stage."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from evidence_utils import supporting_fact_metrics
from experiment_utils import collect_environment, git_commit, portable_path, set_global_seed, utc_now, write_json_atomic
from generate_phase3a import chat_prompt, fit_prompt
from phase2_controller_inputs import evidence_key
from phase3a_generation_utils import exact_match_score, extract_short_answer, file_sha256, load_config, token_f1_score
from phase4_heldout_guard import ROOT, guard_heldout_access, repo_path
from train_phase2_controller import read_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def timed_generate(model, encoded: dict, generation_config: dict, torch_module):
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


def validated_resume_prefix(path: Path, records: list[dict], split: str) -> int:
    """Validate and retain only the complete JSONL prefix from an interrupted run."""
    if not path.exists():
        return 0

    completed = 0
    last_complete_offset = 0
    with path.open("rb") as handle:
        while line := handle.readline():
            next_offset = handle.tell()
            if not line.endswith(b"\n"):
                break
            if completed >= len(records):
                raise ValueError("Partial generation cache has more rows than expected")
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid complete row {completed + 1} in partial generation cache"
                ) from error
            expected = records[completed]
            identity = (
                row.get("question_id"),
                row.get("split"),
                row.get("stage"),
                row.get("stage_index"),
            )
            expected_identity = (
                expected["question_id"],
                split,
                expected["stage"],
                int(expected["stage_index"]) + 1,
            )
            if identity != expected_identity:
                raise ValueError(
                    "Partial generation cache is not an exact prefix of the frozen "
                    f"heldout records at row {completed + 1}"
                )
            completed += 1
            last_complete_offset = next_offset

    file_size = path.stat().st_size
    if last_complete_offset != file_size:
        with path.open("r+b") as handle:
            handle.truncate(last_complete_offset)
        logger.warning(
            "discarded %s trailing bytes from an incomplete JSONL row",
            file_size - last_complete_offset,
        )
    return completed


def main() -> None:
    args = parse_args()
    heldout_config, _ = guard_heldout_access()
    split = heldout_config["split"]
    generation_config_path = repo_path(heldout_config["generation"]["config"])
    generation_config = load_config(generation_config_path)
    freeze_path = ROOT / "results" / "phase4" / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_generation = freeze["files"]["generation_config"]
    if file_sha256(generation_config_path) != frozen_generation["sha256"]:
        raise ValueError("Generation configuration changed after the Phase 4 freeze")
    model_path = Path(
        args.model
        or os.environ.get("LLM_MODEL_PATH")
        or freeze["models"]["generator"]["local_path"]
    )
    if file_sha256(model_path / "config.json") != freeze["models"]["generator"]["config_sha256"]:
        raise ValueError("Generator differs from the frozen Phase 4 model")

    source_path = ROOT / "data" / "2wiki" / "controller" / "sentence_256" / f"{split}.jsonl"
    records = read_jsonl(source_path)
    expected = int(heldout_config["expected_questions"]) * 3
    if len(records) != expected:
        raise ValueError(f"Expected {expected} heldout stage records, got {len(records)}")
    output_path = repo_path(heldout_config["generation"]["stage_cache"])
    manifest_path = output_path.with_name("generation_manifest.json")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite heldout stage generations: {existing}")

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
    chunks = read_jsonl(ROOT / "data" / "2wiki" / "chunks" / "sentence_256.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}

    first = records[0]
    first_view = first[evidence_key("cumulative")]
    warm_prompt, _, _ = fit_prompt(
        tokenizer,
        generation_config,
        first["question"],
        [chunk_by_id[item["chunk_id"]] for item in first_view["items"]],
    )
    warm_encoded = tokenizer(
        chat_prompt(tokenizer, warm_prompt), return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    for _ in range(5):
        timed_generate(model, warm_encoded, generation_config, torch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".jsonl.tmp")
    resumed_stage_generations = validated_resume_prefix(temporary, records, split)
    if resumed_stage_generations:
        logger.info(
            "resuming heldout generation from validated row %s/%s",
            resumed_stage_generations,
            len(records),
        )
    started_all = time.perf_counter()
    mode = "a" if resumed_stage_generations else "w"
    with temporary.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(
            records[resumed_stage_generations:], start=resumed_stage_generations + 1
        ):
            view = record[evidence_key("cumulative")]
            evidence_chunks = [chunk_by_id[item["chunk_id"]] for item in view["items"]]
            prompt, used_chunk_ids, context_truncated = fit_prompt(
                tokenizer, generation_config, record["question"], evidence_chunks
            )
            encoded = tokenizer(
                chat_prompt(tokenizer, prompt), return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            output, generation_latency_ms = timed_generate(
                model, encoded, generation_config, torch
            )
            raw_output = tokenizer.decode(
                output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            extracted = extract_short_answer(raw_output)
            visible = supporting_fact_metrics(
                used_chunk_ids, chunk_by_id, record["gold_supporting_facts"]
            )
            row = {
                "question_id": record["question_id"],
                "split": split,
                "stage": record["stage"],
                "stage_index": int(record["stage_index"]) + 1,
                "question": record["question"],
                "gold_answer": record["answer"],
                "question_type": record.get("question_type"),
                "gold_supporting_fact_count": record["gold_supporting_fact_count"],
                "raw_output": raw_output,
                "extracted_answer": extracted,
                "exact_match": exact_match_score(extracted, record["answer"]),
                "token_f1": token_f1_score(extracted, record["answer"]),
                "prompt_tokens": int(encoded["input_ids"].shape[1]),
                "output_tokens": len(tokenizer(raw_output, add_special_tokens=False)["input_ids"]),
                "raw_output_lines": len([line for line in raw_output.splitlines() if line.strip()]),
                "context_truncated": bool(context_truncated),
                "generation_latency_ms": generation_latency_ms,
                "available_evidence_chunk_ids": [item["chunk_id"] for item in view["items"]],
                "used_evidence_chunk_ids": used_chunk_ids,
                "supporting_fact_recall": float(view["supporting_fact_recall"]),
                "complete_evidence_coverage": int(view["complete_evidence_coverage"]),
                "evidence_state": view["evidence_state"],
                "model_visible_supporting_fact_recall": float(visible["supporting_fact_recall"]),
                "model_visible_complete_evidence_coverage": int(visible["complete_evidence_coverage"]),
                "model_visible_evidence_state": visible["evidence_state"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 250 == 0:
                handle.flush()
                elapsed = time.perf_counter() - started_all
                logger.info(
                    "generated heldout %s/%s stage answers (%.3f/s)",
                    index,
                    len(records),
                    (index - resumed_stage_generations) / elapsed,
                )
    temporary.replace(output_path)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "git_commit": git_commit(ROOT),
            "split": split,
            "heldout_consulted": True,
            "heldout_tuning": False,
            "questions": int(heldout_config["expected_questions"]),
            "stage_generations": len(records),
            "resumed_from_partial_cache": bool(resumed_stage_generations),
            "resumed_stage_generations": resumed_stage_generations,
            "stages": ["dense@5", "hybrid@10", "rerank@20"],
            "source": portable_path(source_path, ROOT),
            "output": portable_path(output_path, ROOT),
            "output_sha256": file_sha256(output_path),
            "frozen_generation_config": portable_path(generation_config_path, ROOT),
            "frozen_generation_config_sha256": file_sha256(generation_config_path),
            "model": str(model_path),
            "model_config_sha256": file_sha256(model_path / "config.json"),
            "decoding": generation_config["decoding"],
            "answer_extraction": generation_config["answer_extraction"],
            "normalization": generation_config["normalization"],
            "generation_latency_cuda_synchronized": bool(torch.cuda.is_available()),
            "environment": collect_environment(ROOT),
        },
    )
    logger.info("saved %s heldout stage generations -> %s", len(records), output_path)


if __name__ == "__main__":
    main()

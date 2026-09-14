"""Run the two frozen LLM-Judge representations once on Phase 4 heldout."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from experiment_utils import collect_environment, git_commit, portable_path, set_global_seed, utc_now, write_json_atomic
from generate_phase3a import chat_prompt
from phase2_controller_inputs import evidence_key
from phase4_heldout_guard import ROOT, guard_heldout_access
from run_phase4_llm_judge import (
    file_sha256,
    object_sha256,
    prepare_judge_input,
    timed_generate,
    validate_model_files,
    validate_prompt_gate,
)
from train_phase2_controller import read_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--representations", default="judge_512,judge_full")
    parser.add_argument("--model")
    parser.add_argument("--packing-tokenizer")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_config, _ = guard_heldout_access()
    split = heldout_config["split"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    judge = config["llm_judge"]
    representations = [value.strip() for value in args.representations.split(",") if value.strip()]
    if not representations or set(representations) - set(judge["representations"]):
        raise ValueError(f"Invalid frozen Judge representations: {representations}")
    if judge["actionable_stages"] != config["controller_ladder"][:-1]:
        raise ValueError("Judge actionable stages differ from the frozen ladder")
    prompt_path = ROOT / judge["prompt_path"]
    template = prompt_path.read_text(encoding="utf-8").strip()
    prompt_gate = validate_prompt_gate(config, prompt_path)
    freeze_path = ROOT / "results" / "phase4" / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    model_path = Path(
        args.model
        or os.environ.get("LLM_MODEL_PATH")
        or freeze["models"]["generator"]["local_path"]
    )
    packing_path = Path(
        args.packing_tokenizer or freeze["models"]["dense"]["local_path"]
    )
    validate_model_files(model_path, packing_path, freeze)

    source_path = ROOT / "data" / "2wiki" / "controller" / "sentence_256" / f"{split}.jsonl"
    all_records = read_jsonl(source_path)
    records = [
        row for row in all_records if row["stage"] in judge["actionable_stages"]
    ]
    question_ids = sorted({row["question_id"] for row in records})
    expected_questions = int(heldout_config["expected_questions"])
    if len(question_ids) != expected_questions:
        raise ValueError(f"Expected {expected_questions} heldout questions")
    if len(records) != expected_questions * len(judge["actionable_stages"]):
        raise ValueError("Heldout Judge source has incomplete actionable trajectories")
    order = {name: index for index, name in enumerate(judge["actionable_stages"])}
    records.sort(key=lambda row: (row["question_id"], order[row["stage"]]))

    output_dir = ROOT / "results" / "phase4" / "2wiki" / "llm_judge" / split
    targets = [
        output_dir / filename
        for representation in representations
        for filename in (f"{representation}.jsonl", f"{representation}_manifest.json")
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite heldout Judge outputs: {existing}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_global_seed(42)
    load_kwargs = {"local_files_only": not args.allow_download}
    generation_tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
    generation_tokenizer.padding_side = "left"
    packing_tokenizer = AutoTokenizer.from_pretrained(packing_path, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        **load_kwargs,
    )
    model.eval()
    chunks = read_jsonl(ROOT / "data" / "2wiki" / "chunks" / "sentence_256.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    output_dir.mkdir(parents=True, exist_ok=True)

    for representation in representations:
        output_path = output_dir / f"{representation}.jsonl"
        manifest_path = output_dir / f"{representation}_manifest.json"
        temporary = output_path.with_suffix(".jsonl.tmp")
        warmup_states = int(judge["latency_warmup_states"])
        if warmup_states:
            warmup_prompt, _ = prepare_judge_input(
                records[0], representation, config, template,
                generation_tokenizer, packing_tokenizer, chunk_by_id,
            )
            warmup_encoded = generation_tokenizer(
                chat_prompt(generation_tokenizer, warmup_prompt),
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            for _ in range(warmup_states):
                timed_generate(model, warmup_encoded, int(judge["max_new_tokens"]), torch)

        started = time.perf_counter()
        valid_outputs = 0
        input_tokens = output_tokens = total_latency = 0.0
        allowed = set(judge["allowed_labels"])
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for index, record in enumerate(records, start=1):
                prompt, packing_audit = prepare_judge_input(
                    record, representation, config, template,
                    generation_tokenizer, packing_tokenizer, chunk_by_id,
                )
                encoded = generation_tokenizer(
                    chat_prompt(generation_tokenizer, prompt),
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(model.device)
                output, latency_ms = timed_generate(
                    model, encoded, int(judge["max_new_tokens"]), torch
                )
                raw_output = generation_tokenizer.decode(
                    output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
                normalized = raw_output.strip().upper()
                parsed = normalized if normalized in allowed else None
                generated_tokens = len(
                    generation_tokenizer(raw_output, add_special_tokens=False)["input_ids"]
                )
                view = record[evidence_key("cumulative")]
                row = {
                    "question_id": record["question_id"],
                    "split": split,
                    "stage": record["stage"],
                    "representation": representation,
                    "actual_stop_label": int(view["stop_label"]),
                    "actual_three_class_label": view["evidence_state"],
                    "true_coverage": float(view["supporting_fact_recall"]),
                    "raw_output": raw_output,
                    "parsed_label": parsed,
                    "parse_valid": parsed is not None,
                    "invalid_output_action": judge["invalid_output_action"],
                    "predicted_stop_label": int(parsed == "SUFFICIENT") if parsed else 0,
                    "input_tokens": int(encoded["input_ids"].shape[1]),
                    "output_tokens": generated_tokens,
                    "judge_latency_ms": latency_ms,
                    "available_evidence_chunk_ids": [item["chunk_id"] for item in view["items"]],
                    "packing_audit": packing_audit,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                valid_outputs += int(parsed is not None)
                input_tokens += row["input_tokens"]
                output_tokens += generated_tokens
                total_latency += latency_ms
                if index % 250 == 0:
                    elapsed = time.perf_counter() - started
                    logger.info(
                        "heldout %s %s/%s (%.3f/s)",
                        representation, index, len(records), index / elapsed,
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
                "questions": len(question_ids),
                "actionable_states": len(records),
                "actionable_stages": judge["actionable_stages"],
                "terminal_stage_excluded": config["controller_ladder"][-1],
                "representation": representation,
                "prompt": portable_path(prompt_path, ROOT),
                "prompt_sha256": file_sha256(prompt_path),
                "llm_judge_config_sha256": object_sha256(judge),
                "prompt_gate_sha256": file_sha256(
                    ROOT / "results" / "phase4" / "2wiki" / "llm_judge" / "prompt_freeze_manifest.json"
                ),
                "source": portable_path(source_path, ROOT),
                "source_sha256": file_sha256(source_path),
                "output": portable_path(output_path, ROOT),
                "output_sha256": file_sha256(output_path),
                "model": str(model_path),
                "model_config_sha256": file_sha256(model_path / "config.json"),
                "torch_dtype": str(next(model.parameters()).dtype),
                "max_context_tokens": int(judge["judge_full"]["max_input_tokens"]),
                "decoding": judge["decoding"],
                "max_new_tokens": int(judge["max_new_tokens"]),
                "label_probability_available": False,
                "strict_parse_rate": valid_outputs / len(records),
                "invalid_outputs": len(records) - valid_outputs,
                "average_input_tokens": input_tokens / len(records),
                "average_output_tokens": output_tokens / len(records),
                "average_judge_latency_ms": total_latency / len(records),
                "latency_warmup_states": warmup_states,
                "environment": collect_environment(ROOT),
            },
        )
        logger.info("saved heldout %s -> %s", representation, output_path)


if __name__ == "__main__":
    main()

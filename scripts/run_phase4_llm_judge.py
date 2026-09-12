"""Run the frozen Qwen LLM-as-a-Judge on Train-derived 2Wiki splits only."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

try:
    from experiment_utils import collect_environment, git_commit, portable_path, set_global_seed, utc_now, write_json_atomic
    from generate_phase3a import chat_prompt, prompt_token_count
    from phase2_controller_inputs import evidence_key, prepare_nonhierarchical_input
    from train_phase2_controller import read_jsonl
except ModuleNotFoundError:  # Imported as scripts.* in tests.
    from scripts.experiment_utils import collect_environment, git_commit, portable_path, set_global_seed, utc_now, write_json_atomic
    from scripts.generate_phase3a import chat_prompt, prompt_token_count
    from scripts.phase2_controller_inputs import evidence_key, prepare_nonhierarchical_input
    from scripts.train_phase2_controller import read_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "protocol.json"
ALLOWED_SPLITS = {"dev_calibration", "dev_policy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=sorted(ALLOWED_SPLITS), required=True)
    parser.add_argument("--representations", default="judge_512,judge_full")
    parser.add_argument("--model", help="Local path to the Phase 3A-frozen generator")
    parser.add_argument("--packing-tokenizer", help="Local BGE tokenizer path")
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_label(raw_output: str, allowed_labels: list[str]) -> str | None:
    normalized = raw_output.strip().upper()
    return normalized if normalized in set(allowed_labels) else None


def render_judge_prompt(template: str, question: str, evidence: str) -> str:
    return template.format(question=question, packed_evidence=evidence)


def render_full_chunk(item: dict, chunk: dict) -> str:
    title = item.get("document_title", chunk["document_title"])
    return f"[DOC] [TITLE] {title} [TEXT] {chunk['chunk_text']}"


def fit_full_evidence(
    tokenizer,
    template: str,
    question: str,
    items: list[dict],
    chunk_by_id: dict[str, dict],
    max_input_tokens: int,
) -> tuple[str, dict]:
    empty_prompt = render_judge_prompt(template, question, "")
    if prompt_token_count(tokenizer, empty_prompt) >= max_input_tokens:
        raise ValueError("Question and fixed Judge prompt exceed the input budget")
    blocks = []
    used_chunk_ids = []
    truncated = False
    for item in items:
        chunk = chunk_by_id[item["chunk_id"]]
        block = render_full_chunk(item, chunk)
        candidate = render_judge_prompt(template, question, "\n\n".join([*blocks, block]))
        if prompt_token_count(tokenizer, candidate) <= max_input_tokens:
            blocks.append(block)
            used_chunk_ids.append(item["chunk_id"])
            continue
        truncated = True
        low, high = 0, len(block)
        while low < high:
            middle = (low + high + 1) // 2
            partial = render_judge_prompt(
                template, question, "\n\n".join([*blocks, block[:middle]])
            )
            if prompt_token_count(tokenizer, partial) <= max_input_tokens:
                low = middle
            else:
                high = middle - 1
        if low:
            blocks.append(block[:low])
            used_chunk_ids.append(item["chunk_id"])
        break
    evidence = "\n\n".join(blocks)
    return evidence, {
        "packing": "retrieval_order_full_chunks_then_final_prefix_truncation",
        "max_input_tokens": max_input_tokens,
        "available_evidence_chunks": len(items),
        "visible_evidence_chunks": len(used_chunk_ids),
        "used_evidence_chunk_ids": used_chunk_ids,
        "truncated": truncated or len(used_chunk_ids) < len(items),
    }


def prepare_judge_input(
    record: dict,
    representation: str,
    config: dict,
    template: str,
    generation_tokenizer,
    packing_tokenizer,
    chunk_by_id: dict[str, dict],
) -> tuple[str, dict]:
    judge = config["llm_judge"]
    view = record[evidence_key("cumulative")]
    if representation == "judge_512":
        settings = judge["judge_512"]
        prepared = prepare_nonhierarchical_input(
            record,
            baseline="query_evidence",
            representation=settings["representation"],
            evidence_mode="cumulative",
            tokenizer=packing_tokenizer,
            chunk_by_id=chunk_by_id,
            max_length=int(settings["critic_pair_max_length"]),
            minimum_chunk_tokens=int(settings["minimum_chunk_tokens"]),
            include_stage_in_query_evidence=bool(
                settings["include_stage_metadata_in_packing_budget"]
            ),
        )
        evidence = prepared["text_b"]
        audit = {
            **prepared["packing_audit"],
            "packing": "critic_score_aware_pair_budget",
            "used_evidence_chunk_ids": [
                allocation["chunk_id"]
                for allocation in prepared["packing_audit"]["allocations"]
                if int(allocation["allocated_tokens"]) > 0
            ],
        }
    elif representation == "judge_full":
        settings = judge["judge_full"]
        evidence, audit = fit_full_evidence(
            generation_tokenizer,
            template,
            record["question"],
            list(view["items"]),
            chunk_by_id,
            int(settings["max_input_tokens"]),
        )
    else:
        raise ValueError(f"Unknown Judge representation: {representation}")
    prompt = render_judge_prompt(template, record["question"], evidence)
    max_input_tokens = int(judge["judge_full"]["max_input_tokens"])
    if prompt_token_count(generation_tokenizer, prompt) > max_input_tokens:
        raise ValueError(f"{representation} prompt exceeds frozen Judge input budget")
    audit["packed_evidence_sha256"] = hashlib.sha256(evidence.encode()).hexdigest()
    audit["packed_evidence_characters"] = len(evidence)
    return prompt, audit


def timed_generate(model, encoded, max_new_tokens: int, torch_module) -> tuple[object, float]:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    started = time.perf_counter()
    with torch_module.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=model.config.eos_token_id,
        )
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def validate_model_files(model_path: Path, packing_path: Path, freeze: dict) -> None:
    model_config = model_path / "config.json"
    packing_config = packing_path / "config.json"
    if not model_config.exists() or not packing_config.exists():
        raise FileNotFoundError("Judge model and packing tokenizer must have config.json")
    if file_sha256(model_config) != freeze["models"]["generator"]["config_sha256"]:
        raise ValueError("Judge model is not the Phase 3A-frozen generator")
    if file_sha256(packing_config) != freeze["models"]["dense"]["config_sha256"]:
        raise ValueError("Judge-512 tokenizer is not the frozen BGE tokenizer")


def validate_prompt_gate(config: dict, prompt_path: Path) -> dict:
    gate_path = ROOT / "results" / "phase4" / "2wiki" / "llm_judge" / "prompt_freeze_manifest.json"
    if not gate_path.exists():
        raise ValueError("Formal dev_policy Judge run requires the frozen prompt gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "frozen" or gate.get("heldout_consulted") is not False:
        raise ValueError("LLM Judge prompt gate is not frozen before heldout")
    if gate.get("prompt_sha256") != file_sha256(prompt_path):
        raise ValueError("LLM Judge prompt changed after format approval")
    if gate.get("llm_judge_config_sha256") != object_sha256(config["llm_judge"]):
        raise ValueError("LLM Judge configuration changed after format approval")
    return gate


def main() -> None:
    args = parse_args()
    if args.split not in ALLOWED_SPLITS:
        raise ValueError("LLM Judge is restricted to Train-derived splits")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    judge = config["llm_judge"]
    if judge["actionable_stages"] != config["controller_ladder"][:-1]:
        raise ValueError("Judge stages must exclude the frozen terminal stage")
    representations = [value.strip() for value in args.representations.split(",") if value.strip()]
    if not representations or set(representations) - set(judge["representations"]):
        raise ValueError(f"Invalid Judge representations: {representations}")
    if args.split == "dev_policy" and args.limit_questions is not None:
        raise ValueError("Formal dev_policy Judge evaluation cannot be truncated")
    if args.split == "dev_calibration":
        expected_limit = int(judge["prompt_check_questions"])
        if args.limit_questions != expected_limit:
            raise ValueError(
                f"Prompt check must use the frozen {expected_limit}-question sample"
            )
    prompt_path = repo_path(judge["prompt_path"])
    template = prompt_path.read_text(encoding="utf-8").strip()
    if set(judge["allowed_labels"]) != {"SUFFICIENT", "INSUFFICIENT"}:
        raise ValueError("Judge labels changed from the binary frozen protocol")
    prompt_gate = validate_prompt_gate(config, prompt_path) if args.split == "dev_policy" else None

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

    source_path = ROOT / "data" / "2wiki" / "controller" / "sentence_256" / f"{args.split}.jsonl"
    all_records = read_jsonl(source_path)
    if {row["split"] for row in all_records} != {args.split}:
        raise ValueError("Controller source contains an unexpected split")
    question_ids = sorted({row["question_id"] for row in all_records})
    if args.limit_questions is not None:
        question_ids = question_ids[: args.limit_questions]
    selected_ids = set(question_ids)
    records = [
        row for row in all_records
        if row["question_id"] in selected_ids
        and row["stage"] in judge["actionable_stages"]
    ]
    expected = len(question_ids) * len(judge["actionable_stages"])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} actionable states, got {len(records)}")
    order = {name: index for index, name in enumerate(judge["actionable_stages"])}
    records.sort(key=lambda row: (row["question_id"], order[row["stage"]]))

    output_root = ROOT / "results" / "phase4" / "2wiki" / "llm_judge"
    run_name = "prompt_check" if args.split == "dev_calibration" else "dev_policy"
    output_dir = output_root / run_name
    targets = []
    for representation in representations:
        targets.extend(
            [
                output_dir / f"{representation}.jsonl",
                output_dir / f"{representation}_manifest.json",
            ]
        )
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite LLM Judge outputs: {existing}")

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
        if records and warmup_states:
            warmup_prompt, _ = prepare_judge_input(
                records[0],
                representation,
                config,
                template,
                generation_tokenizer,
                packing_tokenizer,
                chunk_by_id,
            )
            warmup_rendered = chat_prompt(generation_tokenizer, warmup_prompt)
            warmup_encoded = generation_tokenizer(
                warmup_rendered, return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            for _ in range(warmup_states):
                timed_generate(
                    model, warmup_encoded, int(judge["max_new_tokens"]), torch
                )

        started = time.perf_counter()
        valid_outputs = 0
        input_tokens = output_tokens = total_latency = 0.0
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for index, record in enumerate(records, start=1):
                prompt, packing_audit = prepare_judge_input(
                    record,
                    representation,
                    config,
                    template,
                    generation_tokenizer,
                    packing_tokenizer,
                    chunk_by_id,
                )
                rendered = chat_prompt(generation_tokenizer, prompt)
                encoded = generation_tokenizer(
                    rendered, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                output, latency_ms = timed_generate(
                    model, encoded, int(judge["max_new_tokens"]), torch
                )
                raw_output = generation_tokenizer.decode(
                    output[0, encoded["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()
                parsed = parse_label(raw_output, judge["allowed_labels"])
                predicted_stop = int(parsed == "SUFFICIENT") if parsed else 0
                generated_tokens = len(
                    generation_tokenizer(raw_output, add_special_tokens=False)["input_ids"]
                )
                view = record[evidence_key("cumulative")]
                row = {
                    "question_id": record["question_id"],
                    "split": args.split,
                    "stage": record["stage"],
                    "representation": representation,
                    "actual_stop_label": int(view["stop_label"]),
                    "actual_three_class_label": view["evidence_state"],
                    "true_coverage": float(view["supporting_fact_recall"]),
                    "raw_output": raw_output,
                    "parsed_label": parsed,
                    "parse_valid": parsed is not None,
                    "invalid_output_action": judge["invalid_output_action"],
                    "predicted_stop_label": predicted_stop,
                    "input_tokens": int(encoded["input_ids"].shape[1]),
                    "output_tokens": generated_tokens,
                    "judge_latency_ms": latency_ms,
                    "available_evidence_chunk_ids": [
                        item["chunk_id"] for item in view["items"]
                    ],
                    "packing_audit": packing_audit,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                valid_outputs += int(parsed is not None)
                input_tokens += row["input_tokens"]
                output_tokens += generated_tokens
                total_latency += latency_ms
                if index % 25 == 0:
                    elapsed = time.perf_counter() - started
                    logger.info(
                        "%s %s %s/%s (%.3f/s)",
                        args.split,
                        representation,
                        index,
                        len(records),
                        index / elapsed,
                    )
        temporary.replace(output_path)
        write_json_atomic(
            manifest_path,
            {
                "schema_version": 1,
                "created_at_utc": utc_now(),
                "phase": 4,
                "git_commit": git_commit(ROOT),
                "split": args.split,
                "heldout_consulted": False,
                "purpose": run_name,
                "questions": len(question_ids),
                "actionable_states": len(records),
                "actionable_stages": judge["actionable_stages"],
                "terminal_stage_excluded": config["controller_ladder"][-1],
                "representation": representation,
                "prompt": portable_path(prompt_path, ROOT),
                "prompt_sha256": file_sha256(prompt_path),
                "llm_judge_config_sha256": object_sha256(judge),
                "prompt_gate_sha256": file_sha256(
                    output_root / "prompt_freeze_manifest.json"
                ) if prompt_gate else None,
                "source": portable_path(source_path, ROOT),
                "source_sha256": file_sha256(source_path),
                "output": portable_path(output_path, ROOT),
                "output_sha256": file_sha256(output_path),
                "model": str(model_path),
                "model_config_sha256": file_sha256(model_path / "config.json"),
                "model_revision": getattr(model.config, "_commit_hash", None),
                "torch_dtype": str(next(model.parameters()).dtype),
                "device_map": str(getattr(model, "hf_device_map", None)),
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
        logger.info("saved %s -> %s", representation, output_path)


if __name__ == "__main__":
    main()

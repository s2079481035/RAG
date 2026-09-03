"""Create a deterministic stratified dev audit sheet without changing automatic labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path

from phase2_packing import gold_fact_bucket, length_bucket


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "docs" / "manual_sufficiency_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_tokens(prediction: dict) -> int:
    if prediction.get("raw_evidence_tokens") is not None:
        return int(prediction["raw_evidence_tokens"])
    audit = prediction.get("packing_audit")
    return int(audit["raw_evidence_tokens"]) if audit else 0


def outcome(prediction: dict) -> str:
    actual = int(prediction["actual_stop_label"])
    predicted = int(prediction["predicted_stop_label"])
    if actual == 0 and predicted == 1:
        return "false_stop"
    if actual == 1 and predicted == 1:
        return "correct_stop"
    if actual == 1 and predicted == 0:
        return "unnecessary_escalation"
    return "correct_continue"


def stratified_sample(predictions: list[dict], count: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for prediction in predictions:
        key = (
            prediction["actual_three_class_label"],
            outcome(prediction),
            length_bucket(raw_tokens(prediction)),
            gold_fact_bucket(int(prediction["gold_supporting_fact_count"])),
        )
        groups[key].append(prediction)
    rng = random.Random(seed)
    queues = []
    for key in sorted(groups):
        values = groups[key]
        rng.shuffle(values)
        queues.append((key, deque(values)))
    selected = []
    while len(selected) < min(count, len(predictions)) and queues:
        next_queues = []
        for key, queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                next_queues.append((key, queue))
        queues = next_queues
    return selected


def render_evidence(chunk_ids: list[str], chunk_by_id: dict) -> str:
    blocks = []
    for chunk_id in chunk_ids:
        chunk = chunk_by_id[chunk_id]
        blocks.append(
            f"[{chunk_id}] TITLE: {chunk['document_title']} | SENTENCE_IDS: "
            f"{chunk['sentence_ids']} | TEXT: {chunk['chunk_text']}"
        )
    return "\n\n".join(blocks)


def main() -> None:
    args = parse_args()
    if not 150 <= args.count <= 200:
        raise ValueError("Manual audit count must remain between 150 and 200")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite manual audit: {args.output}")
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    resolved = config["resolved"]
    predictions_path = run_dir / "evaluation" / "dev" / "original_predictions.jsonl"
    predictions = read_jsonl(predictions_path)
    variant = resolved["variant"]
    chunks = read_jsonl(
        ROOT / "data" / "phase2" / "chunks" / f"{variant}.jsonl"
    )
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    controller_records = read_jsonl(
        ROOT / "data" / "phase2" / "controller" / variant / "dev.jsonl"
    )
    record_by_key = {
        (record["question_id"], record["stage"]): record for record in controller_records
    }
    view_key = (
        "raw_stage_evidence"
        if resolved["evidence_mode"] == "raw"
        else "cumulative_evidence_memory"
    )
    selected = stratified_sample(predictions, args.count, args.seed)
    fields = [
        "audit_id",
        "question_id",
        "stage",
        "question",
        "gold_answer",
        "gold_supporting_facts",
        "retrieved_evidence",
        "covered_supporting_facts",
        "automatic_three_class_label",
        "automatic_stop_label",
        "critic_stop_probability",
        "critic_prediction",
        "prediction_outcome",
        "evidence_length_tokens",
        "evidence_length_bucket",
        "gold_supporting_fact_count",
        "gold_supporting_fact_bucket",
        "visible_supporting_fact_ratio_evaluation_only",
        "human_sufficient_to_answer",
        "human_label",
        "human_notes",
        "reviewer",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, prediction in enumerate(selected, start=1):
            record = record_by_key[(prediction["question_id"], prediction["stage"])]
            view = record[view_key]
            tokens = raw_tokens(prediction)
            count = int(prediction["gold_supporting_fact_count"])
            writer.writerow(
                {
                    "audit_id": index,
                    "question_id": prediction["question_id"],
                    "stage": prediction["stage"],
                    "question": prediction["question"],
                    "gold_answer": prediction["gold_answer"],
                    "gold_supporting_facts": json.dumps(
                        prediction["gold_supporting_facts"], ensure_ascii=False
                    ),
                    "retrieved_evidence": render_evidence(
                        prediction["input_evidence_chunk_ids"], chunk_by_id
                    ),
                    "covered_supporting_facts": json.dumps(
                        view["covered_supporting_facts"], ensure_ascii=False
                    ),
                    "automatic_three_class_label": prediction[
                        "actual_three_class_label"
                    ],
                    "automatic_stop_label": prediction["actual_stop_label"],
                    "critic_stop_probability": prediction["stop_probability"],
                    "critic_prediction": prediction["predicted_stop_label"],
                    "prediction_outcome": outcome(prediction),
                    "evidence_length_tokens": tokens,
                    "evidence_length_bucket": length_bucket(tokens),
                    "gold_supporting_fact_count": count,
                    "gold_supporting_fact_bucket": gold_fact_bucket(count),
                    "visible_supporting_fact_ratio_evaluation_only": prediction.get(
                        "visible_supporting_fact_ratio_evaluation_only"
                    ),
                    "human_sufficient_to_answer": "",
                    "human_label": "",
                    "human_notes": "",
                    "reviewer": "",
                }
            )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()

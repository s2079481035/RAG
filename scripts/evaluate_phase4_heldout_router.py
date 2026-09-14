"""Evaluate every frozen Adaptive-RAG-style Router seed once on heldout."""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

from experiment_utils import portable_path, utc_now, write_json_atomic
from phase2_model import configure_cublas_workspace
from phase4_adaptive_router import (
    ROUTE_NAMES,
    file_sha256,
    grouped_trajectories,
    router_record,
    write_jsonl_atomic,
)
from phase4_heldout_guard import ROOT, guard_heldout_access
from train_phase4_adaptive_router import collate_batch, infer, replace_output_head


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "adaptive_rag_router.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_config, _ = guard_heldout_access()
    split = heldout_config["split"]
    run_dir = args.run_dir.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate_path = ROOT / "results" / "phase4" / "2wiki" / "adaptive_rag" / "dev_gate_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    seed = int(manifest.get("seed", -1))
    if seed not in heldout_config["adaptive_router_seeds"]:
        raise ValueError(f"Router seed is not frozen for heldout: {seed}")
    if manifest.get("status") != "complete" or manifest.get("heldout_consulted") is not False:
        raise ValueError("Router training run is not a clean pre-heldout checkpoint")
    if manifest.get("router_config_sha256") != file_sha256(args.config):
        raise ValueError("Router configuration changed after training")
    frozen = gate["run_sources"][f"seed{seed}"]
    if file_sha256(manifest_path) != frozen["run_manifest_sha256"]:
        raise ValueError(f"Router seed {seed} manifest differs from the frozen Dev gate")
    checkpoint = Path(manifest["checkpoint_path"])
    if file_sha256(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError(f"Router seed {seed} checkpoint differs from the frozen Dev gate")
    if resolved["resolved"]["model_input_fields"] != ["question"]:
        raise ValueError("Heldout Router must remain question-only")

    output_dir = run_dir / "evaluation" / split
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    evaluation_manifest_path = output_dir / "evaluation_manifest.json"
    existing = [
        path for path in (predictions_path, metrics_path, evaluation_manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing to overwrite heldout Router evaluation: {existing}")

    source_path = ROOT / config["data"]["source_dir"] / f"{split}.jsonl"
    rows = [
        router_record(trajectory, split)
        for trajectory in grouped_trajectories(source_path, split)
    ]
    if len(rows) != int(heldout_config["expected_questions"]):
        raise ValueError(f"Expected {heldout_config['expected_questions']} heldout questions")

    configure_cublas_workspace(config["training"]["cublas_workspace_config"])
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    backbone = resolved["resolved"]["backbone"]
    load_kwargs = {
        "local_files_only": resolved["resolved"]["local_files_only"]
        and not args.allow_download
    }
    tokenizer = AutoTokenizer.from_pretrained(backbone, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, **load_kwargs)
    replace_output_head(model, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    loader = DataLoader(
        rows,
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=partial(
            collate_batch,
            tokenizer=tokenizer,
            max_length=int(config["model"]["max_length"]),
        ),
    )
    labels, probabilities = infer(model, loader, device)
    predictions = probabilities.argmax(axis=1)
    prediction_rows = []
    for row, probability, prediction in zip(rows, probabilities, predictions):
        prediction_rows.append(
            {
                "question_id": row["question_id"],
                "split": split,
                "actual_route_label": int(row["label"]),
                "actual_route": row["route"],
                "route_probabilities": {
                    name: float(probability[index])
                    for index, name in enumerate(ROUTE_NAMES)
                },
                "predicted_route_label": int(prediction),
                "predicted_route": ROUTE_NAMES[int(prediction)],
            }
        )
    from phase4_adaptive_router import multiclass_metrics

    metrics, _ = multiclass_metrics(labels, probabilities)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(predictions_path, prediction_rows)
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(
        evaluation_manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "split": split,
            "heldout_consulted": True,
            "heldout_tuning": False,
            "prediction_rule": "three_class_argmax_no_threshold_sweep",
            "model_input_fields": ["question"],
            "seed": seed,
            "router_config_sha256": file_sha256(args.config),
            "frozen_dev_gate": portable_path(gate_path, ROOT),
            "frozen_dev_gate_sha256": file_sha256(gate_path),
            "run_dir": portable_path(run_dir, ROOT),
            "checkpoint_sha256": file_sha256(checkpoint),
            "source": portable_path(source_path, ROOT),
            "source_sha256": file_sha256(source_path),
            "output": portable_path(predictions_path, ROOT),
            "output_sha256": file_sha256(predictions_path),
            "metrics": metrics,
        },
    )
    logger.info("evaluated heldout Router seed %s -> %s", seed, output_dir)


if __name__ == "__main__":
    main()

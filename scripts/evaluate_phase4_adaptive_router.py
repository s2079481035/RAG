"""Evaluate a frozen query-only Router on dev_policy without threshold tuning."""

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
    multiclass_metrics,
    read_jsonl,
    write_jsonl_atomic,
)
from train_phase4_adaptive_router import collate_batch, infer, replace_output_head


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "adaptive_rag_router.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="dev_policy", choices=["dev_policy"])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Router training run is incomplete")
    if manifest.get("heldout_consulted") is not False:
        raise ValueError("Router run does not prove heldout isolation")
    if resolved["resolved"]["model_input_fields"] != ["question"]:
        raise ValueError("Router checkpoint was not trained question-only")
    if manifest.get("router_config_sha256") != file_sha256(args.config):
        raise ValueError("Router config changed after training")
    if args.split != config["data"]["evaluation_split"]:
        raise ValueError("Formal pre-heldout Router evaluation is dev_policy only")

    output_dir = run_dir / "evaluation" / args.split
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    evaluation_manifest_path = output_dir / "evaluation_manifest.json"
    existing = [
        path for path in (predictions_path, metrics_path, evaluation_manifest_path)
        if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Router evaluation: {existing}")

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
    model.load_state_dict(
        torch.load(manifest["checkpoint_path"], map_location=device, weights_only=True)
    )
    model.to(device)

    source_path = (
        ROOT / config["data"]["router_dir"] / f"{args.split}.jsonl"
    )
    rows = list(read_jsonl(source_path))
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
    metrics, predictions = multiclass_metrics(labels, probabilities)
    prediction_rows = []
    for row, probability, prediction in zip(rows, probabilities, predictions):
        prediction_rows.append(
            {
                "question_id": row["question_id"],
                "split": args.split,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(predictions_path, prediction_rows)
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(
        evaluation_manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "formal_name": config["formal_name"],
            "split": args.split,
            "heldout_consulted": False,
            "prediction_rule": "three_class_argmax_no_threshold_sweep",
            "model_input_fields": ["question"],
            "router_config_sha256": file_sha256(args.config),
            "run_dir": portable_path(run_dir, ROOT),
            "run_manifest_sha256": file_sha256(run_dir / "run_manifest.json"),
            "checkpoint_sha256": file_sha256(Path(manifest["checkpoint_path"])),
            "source": portable_path(source_path, ROOT),
            "source_sha256": file_sha256(source_path),
            "output": portable_path(predictions_path, ROOT),
            "output_sha256": file_sha256(predictions_path),
            "metrics": metrics,
        },
    )
    logger.info("evaluated %s -> %s", args.split, output_dir)


if __name__ == "__main__":
    main()

"""Build question-level routing targets from frozen 2Wiki retrieval trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic
from phase4_adaptive_router import (
    ALLOWED_DEV_SPLITS,
    ROUTE_NAMES,
    file_sha256,
    grouped_trajectories,
    router_record,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase4" / "adaptive_rag_router.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--splits", default="train_core,dev_calibration,dev_policy"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not splits or set(splits) - ALLOWED_DEV_SPLITS:
        raise ValueError(
            "Router data construction before heldout is restricted to "
            "train_core/dev_calibration/dev_policy"
        )
    if config["inference_features"] != ["question"]:
        raise ValueError("Router inference must remain question-only")
    if config["route_names"] != ROUTE_NAMES:
        raise ValueError("Router class order changed")

    source_dir = repo_path(config["data"]["source_dir"])
    output_dir = repo_path(config["data"]["router_dir"])
    result_dir = repo_path(config["output"]["result_dir"])
    output_paths = {split: output_dir / f"{split}.jsonl" for split in splits}
    manifest_path = result_dir / "dataset_manifest.json"
    distribution_path = result_dir / "label_distribution.json"
    leakage_path = result_dir / "leakage_audit.json"
    targets = [*output_paths.values(), manifest_path, distribution_path, leakage_path]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Router data: {existing}")

    distributions = {}
    sources = {}
    for split in splits:
        source_path = source_dir / f"{split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing frozen Controller data: {source_path}")
        counts = Counter()
        rows = []
        for trajectory in grouped_trajectories(source_path, split):
            row = router_record(trajectory, split)
            counts[row["route"]] += 1
            rows.append(row)
        write_jsonl_atomic(output_paths[split], rows)
        distributions[split] = {
            "questions": len(rows),
            "routes": {name: counts[name] for name in ROUTE_NAMES},
        }
        sources[split] = {
            "path": portable_path(source_path, ROOT),
            "sha256": file_sha256(source_path),
            "output": portable_path(output_paths[split], ROOT),
            "output_sha256": file_sha256(output_paths[split]),
        }

    write_json_atomic(
        distribution_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "heldout_consulted": False,
            "label_construction": config["label_construction"],
            "by_split": distributions,
        },
    )
    write_json_atomic(
        leakage_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "baseline": config["formal_name"],
            "inference_feature_keys": ["question"],
            "forbidden_inference_feature_keys": config[
                "forbidden_inference_features"
            ],
            "model_input_builder": "tokenizer(question_text_only)",
            "gold_used_for_training_target_only": True,
            "retrieved_evidence_used_for_inference": False,
            "future_stage_results_used_for_inference": False,
            "critic_probability_used_for_inference": False,
            "heldout_consulted": False,
            "leakage_detected": False,
        },
    )
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "phase": 4,
            "formal_name": config["formal_name"],
            "splits": splits,
            "heldout_consulted": False,
            "route_names": config["route_names"],
            "stage_names": config["stage_names"],
            "label_construction": config["label_construction"],
            "inference_features": config["inference_features"],
            "sources": sources,
            "config": portable_path(args.config, ROOT),
            "config_sha256": file_sha256(args.config),
            "git_commit": git_commit(ROOT),
        },
    )
    print(manifest_path.relative_to(ROOT))


if __name__ == "__main__":
    main()


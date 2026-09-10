"""Freeze the retrieval execution strategy after non-formal throughput parity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiment_utils import git_commit, portable_path, utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = ROOT / "results" / "phase4" / "2wiki" / "retrieval_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timed_seconds(manifest: dict) -> float:
    return sum(
        float(manifest[key])
        for key in ("query_embedding_seconds", "first_stage_seconds", "reranking_seconds")
    )


def choose_execution(
    serial: dict, parallel: dict, batched: dict, parallel_parity: dict, batched_parity: dict
) -> dict:
    manifests = [serial, parallel, batched]
    if any(item.get("run_scope") != "smoke_throughput_only" for item in manifests):
        raise ValueError("Execution selection requires smoke-only manifests")
    counts = [sum(item["question_counts"].values()) for item in manifests]
    if len(set(counts)) != 1 or counts[0] < 1:
        raise ValueError(f"Smoke runs use different question counts: {counts}")
    if parallel.get("bm25_workers") != 8 or parallel.get("dense_search_batch_size") != 1:
        raise ValueError("Unexpected parallel BM25 candidate settings")
    if batched.get("bm25_workers") != 8 or batched.get("dense_search_batch_size") != 512:
        raise ValueError("Unexpected batched FAISS candidate settings")
    if parallel_parity.get("matches") is not True:
        raise ValueError("Parallel BM25 candidate failed parity")
    if batched_parity.get("matches") is not False:
        raise ValueError("Batched FAISS candidate was expected to be rejected by parity")
    serial_seconds = timed_seconds(serial)
    selected_seconds = timed_seconds(parallel)
    return {
        "selected": {"bm25_workers": 8, "dense_search_batch_size": 1},
        "accepted_optimization": "parallel_bm25_exact_parity",
        "rejected_optimization": "batched_faiss_ranking_parity_failure",
        "questions": counts[0],
        "serial_timed_seconds": serial_seconds,
        "selected_timed_seconds": selected_seconds,
        "timed_pipeline_speedup": serial_seconds / selected_seconds,
        "serial_first_stage_seconds": float(serial["first_stage_seconds"]),
        "selected_first_stage_seconds": float(parallel["first_stage_seconds"]),
        "first_stage_speedup": float(serial["first_stage_seconds"])
        / float(parallel["first_stage_seconds"]),
    }


def main() -> None:
    args = parse_args()
    paths = {
        "serial_manifest": SMOKE_ROOT
        / "dev_policy_100"
        / "sentence_256"
        / "run_manifest_dev_policy.json",
        "parallel_manifest": SMOKE_ROOT
        / "dev_policy_100_parallel8"
        / "sentence_256"
        / "run_manifest_dev_policy.json",
        "parallel_parity": SMOKE_ROOT / "dev_policy_100_parallel8" / "parity.json",
        "batched_manifest": SMOKE_ROOT
        / "dev_policy_100_parallel8_batch512"
        / "sentence_256"
        / "run_manifest_dev_policy.json",
        "batched_parity": SMOKE_ROOT
        / "dev_policy_100_parallel8_batch512"
        / "parity.json",
        "preparation_manifest": ROOT / "data" / "2wiki" / "preparation_manifest.json",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing execution-selection evidence: {missing}")
    loaded = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
    }
    decision = choose_execution(
        loaded["serial_manifest"],
        loaded["parallel_manifest"],
        loaded["batched_manifest"],
        loaded["parallel_parity"],
        loaded["batched_parity"],
    )
    split_counts = loaded["preparation_manifest"]["question_counts"]
    seconds_per_question = decision["selected_timed_seconds"] / decision["questions"]
    estimates = {
        split: seconds_per_question * count / 3600.0
        for split, count in split_counts.items()
    }
    output = ROOT / "results" / "phase4" / "2wiki" / "retrieval_execution_selection.json"
    report = ROOT / "docs" / "phase4" / "retrieval_execution_audit.md"
    if (output.exists() or report.exists()) and not args.force:
        raise FileExistsError("Refusing to overwrite retrieval execution selection")
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": 4,
        "git_commit": git_commit(ROOT),
        "selection_scope": "non_formal_throughput_and_parity_gate",
        "effect_metrics_consulted": False,
        **decision,
        "estimated_hours_excluding_startup": estimates,
        "evidence": {
            name: {"path": portable_path(path, ROOT), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    write_json_atomic(output, manifest)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Phase 4 Retrieval Execution Audit",
                "",
                "This gate uses a deterministic 100-question subset for engineering parity and throughput only. No effect metric is consulted.",
                "",
                f"- Selected BM25 workers: {decision['selected']['bm25_workers']}",
                f"- Selected Dense search batch size: {decision['selected']['dense_search_batch_size']}",
                "- Parallel BM25 parity: passed",
                "- Batched FAISS parity: failed; rejected from formal retrieval",
                f"- Timed pipeline speedup: {decision['timed_pipeline_speedup']:.2f}x",
                f"- First-stage speedup: {decision['first_stage_speedup']:.2f}x",
                "",
                "Estimated hours exclude one-time model/index loading:",
                "",
                *[
                    f"- `{split}`: {hours:.2f} h"
                    for split, hours in sorted(estimates.items())
                ],
                "",
                "The rejected batched run is retained as negative engineering evidence and must not be used for formal results.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()

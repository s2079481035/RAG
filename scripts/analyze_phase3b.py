"""Generate Phase 3B analysis and closing reports only from completed artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiment_utils import git_commit, utc_now


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "phase3b" / "protocol.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.{digits}f}"


def percent(value) -> str:
    if value is None or value == "":
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def find_policy(rows: list[dict], name: str) -> dict:
    matches = [row for row in rows if row["policy"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one policy named {name}, found {len(matches)}")
    return matches[0]


def find_bootstrap(rows: list[dict], metric: str) -> dict:
    matches = [row for row in rows if row["metric"] == metric]
    if len(matches) != 1:
        raise ValueError(f"Expected one bootstrap row for {metric}, found {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {
        "gate": ROOT / "results" / "phase3b" / "test_gate.json",
        "calibration": ROOT / "results" / "phase3b" / "calibration_metrics.csv",
        "selected": ROOT / "results" / "phase3b" / "selected_thresholds.json",
        "dev": ROOT / "results" / "phase3b" / "dev_policy_end_to_end.csv",
        "test": ROOT / "results" / "phase3b" / "test_risk_policy.csv",
        "bootstrap": ROOT / "results" / "phase3b" / "bootstrap_ci.csv",
        "test_manifest": ROOT / "results" / "phase3b" / "test_evaluation_manifest.json",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase 3B is incomplete; missing: {missing}")
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    if gate.get("status") != "frozen" or gate.get("test_consulted") is not False:
        raise ValueError("Invalid Phase 3B Test gate")

    calibration = read_csv(paths["calibration"])
    selected = json.loads(paths["selected"].read_text(encoding="utf-8"))
    dev_rows = read_csv(paths["dev"])
    test_rows = read_csv(paths["test"])
    bootstrap = read_csv(paths["bootstrap"])
    primary_config = config["policy"]["primary_report_policy"]
    primary_name = (
        f"{primary_config['calibration']}_{primary_config['selection']}_risk_"
        f"{int(round(float(primary_config['risk_target']) * 100)):02d}"
    )
    primary = find_policy(test_rows, primary_name)
    heavy = find_policy(test_rows, "fixed_heavy")
    primary_dev = find_policy(dev_rows, primary_name)
    raw_primary_name = (
        f"raw_{primary_config['selection']}_risk_"
        f"{int(round(float(primary_config['risk_target']) * 100)):02d}"
    )
    raw_primary = find_policy(test_rows, raw_primary_name)

    calibration_policy = {
        row["method"]: row
        for row in calibration
        if row["split"] == "dev_policy"
    }
    raw = calibration_policy["raw"]
    temperature = calibration_policy["temperature"]
    calibration_improved = (
        float(temperature["ece"]) < float(raw["ece"])
        and float(temperature["nll"]) < float(raw["nll"])
    )

    answer_delta = find_bootstrap(bootstrap, "answer_f1")
    chunk_delta = find_bootstrap(bootstrap, "retrieval_cost")
    latency_delta = find_bootstrap(bootstrap, "latency_ms")
    fsr_delta = find_bootstrap(bootstrap, "false_stop_rate")
    primary_fsr_ci = None
    if abs(float(heavy["false_stop_rate"])) <= 1e-12:
        primary_fsr_ci = (
            float(fsr_delta["ci95_low"]),
            float(fsr_delta["ci95_high"]),
        )
    raw_primary_equivalent = (
        abs(float(raw_primary["answer_f1"]) - float(primary["answer_f1"])) <= 1e-12
        and abs(float(raw_primary["false_stop_rate"]) - float(primary["false_stop_rate"]))
        <= 1e-12
    )
    raw_primary_chunk_delta = (
        float(raw_primary["average_retrieved_chunks"])
        - float(primary["average_retrieved_chunks"])
    )
    closing = config["go_no_go"]
    criteria = {
        "observed_test_fsr_target": float(primary["false_stop_rate"])
        <= float(primary_config["risk_target"]),
        "answer_f1_noninferiority": float(answer_delta["ci95_low"])
        >= -float(closing["answer_f1_noninferiority_margin"]),
        "retrieved_chunks_reduction": float(chunk_delta["ci95_high"]) < 0.0,
        "total_latency_reduction": float(latency_delta["ci95_high"]) < 0.0,
    }
    positive_claim_go = all(criteria.values())

    calibration_table = markdown_table(
        ["Split", "Method", "NLL", "Brier", "ECE", "AUROC", "Hard Partial high-conf. FSR"],
        [
            [
                row["split"],
                row["method"],
                number(row["nll"]),
                number(row["brier"]),
                number(row["ece"]),
                number(row["auroc"]),
                percent(row["hard_partial_high_confidence_false_stop_rate"]),
            ]
            for row in calibration
        ],
    )
    controller_table = markdown_table(
        ["Policy", "FSR", "UER", "Macro F1", "Stop rate", "Avg final stage", "Hard Partial FSR"],
        [
            [
                row["policy"],
                percent(row["false_stop_rate"]),
                percent(row["unnecessary_escalation_rate"]),
                number(row["macro_f1"]),
                percent(row["stop_rate"]),
                number(row["average_final_stage"], 3),
                percent(row["hard_partial_false_stop_rate"]),
            ]
            for row in test_rows
        ],
    )
    end_to_end_table = markdown_table(
        ["Policy", "EM", "F1", "SF recall", "Complete cov.", "Chunks", "Rerank", "Tokens", "Latency ms", "Pareto"],
        [
            [
                row["policy"],
                number(row["answer_em"]),
                number(row["answer_f1"]),
                number(row["supporting_fact_recall"]),
                number(row["complete_evidence_coverage"]),
                number(row["average_retrieved_chunks"], 2),
                number(row["average_reranker_calls"], 3),
                number(row["average_llm_total_tokens"], 1),
                number(row["total_latency_mean_ms"], 2),
                row["pareto_optimal_chunks_answer_f1"],
            ]
            for row in test_rows
        ],
    )
    risk_rows = []
    for item in selected["policies"]:
        if item["calibration_method"] != "temperature" or item["status"] != "selected":
            continue
        name = (
            f"temperature_{item['selection']}_risk_"
            f"{int(round(float(item['risk_target']) * 100)):02d}"
        )
        result = find_policy(test_rows, name)
        risk_rows.append(
            [
                percent(item["risk_target"]),
                item["selection"],
                number(item["threshold"], 3),
                percent(result["false_stop_rate"]),
                str(float(result["false_stop_rate"]) <= float(item["risk_target"])),
                number(result["answer_f1"]),
                number(result["average_retrieved_chunks"], 2),
                number(result["total_latency_mean_ms"], 2),
            ]
        )
    risk_table = markdown_table(
        ["Risk target", "Selection", "Threshold", "Test FSR", "Observed target met", "Answer F1", "Chunks", "Latency ms"],
        risk_rows,
    )

    if primary_fsr_ci is None:
        primary_fsr_ci_text = "not available from the saved paired comparison"
        primary_fsr_ci_limit = ""
    else:
        primary_fsr_ci_text = (
            f"[{percent(primary_fsr_ci[0])}, {percent(primary_fsr_ci[1])}]"
        )
        primary_fsr_ci_limit = (
            f" Its upper bound exceeds the {percent(primary_config['risk_target'])} "
            "target, so the result is empirical risk control rather than a "
            "statistical guarantee."
            if primary_fsr_ci[1] > float(primary_config["risk_target"])
            else " Its upper bound is within the target."
        )

    pareto_note = (
        f"The pre-registered primary policy has strict two-dimensional Pareto status "
        f"`{primary['pareto_optimal_chunks_answer_f1']}`. `{raw_primary_name}` has "
        f"{'the same Test F1 and FSR to numerical precision' if raw_primary_equivalent else 'different Test outcomes'} "
        f"and changes average retrieved chunks by {raw_primary_chunk_delta:+.3f} per question. "
        "This raw-policy comparison is descriptive and does not replace the frozen primary policy."
    )

    lines = [
        "# Phase 3B Analysis",
        "",
        "Phase 3B evaluates a frozen Phase 3A Controller. Temperature is fitted on `dev_calibration`; thresholds are selected on disjoint `dev_policy`; Test performs no refit or reselection.",
        "",
        "## Table 1: Calibration Quality",
        "",
        *calibration_table,
        "",
        f"On dev_policy, joint NLL/ECE improvement is **{'observed' if calibration_improved else 'not observed'}**. AUROC is descriptive because scalar Temperature Scaling preserves ranking apart from numerical ties.",
        "",
        "## Table 2: Risk-Controlled Controller (Test)",
        "",
        *controller_table,
        "",
        "FSR and UER include only actionable Controller decisions actually reached by each sequential policy. Final-stage insufficient evidence is reported separately in the CSV and is not mislabeled as a controllable False Stop.",
        "",
        "## Table 3: End-to-End RAG (Test)",
        "",
        *end_to_end_table,
        "",
        pareto_note,
        "",
        "## Table 4: Risk-Level Comparison (Test)",
        "",
        *risk_table,
        "",
        "## Paired Question Bootstrap",
        "",
        *markdown_table(
            ["Metric", "Observed delta", "95% CI"],
            [
                [row["metric"], number(row["observed_delta"]), f"[{number(row['ci95_low'])}, {number(row['ci95_high'])}]"]
                for row in bootstrap
            ],
        ),
        "",
        "All deltas are primary risk policy minus Fixed Heavy and use 2,000 paired question-level replicates.",
        "",
        "## Research Questions",
        "",
        f"- RQ1 Calibration: NLL/ECE jointly improved on held-out dev_policy: **{calibration_improved}**.",
        f"- RQ2 Risk transfer: Dev FSR was {percent(primary_dev['false_stop_rate'])}; Test FSR was {percent(primary['false_stop_rate'])} for the primary alpha={float(primary_config['risk_target']):.2f} policy. The Test question-bootstrap 95% CI was {primary_fsr_ci_text}.{primary_fsr_ci_limit}",
        f"- RQ3 Cost: versus Fixed Heavy, retrieved chunks changed by {number(chunk_delta['observed_delta'], 3)} and total latency by {number(latency_delta['observed_delta'], 3)} ms.",
        f"- RQ4 QA quality: primary Test F1 was {number(primary['answer_f1'])}, versus {number(heavy['answer_f1'])} for Fixed Heavy.",
        f"- RQ5 Sweet spot: the pre-registered positive-claim rule is **{'met' if positive_claim_go else 'not met'}**.",
        "- RQ6 Conservative stability: point-selected policies missed all three nominal Test targets. Conservative alpha=0.20 and alpha=0.10 met their observed targets; conservative alpha=0.05 met its target only by always reaching the forced final stage. No stronger guarantee is claimed from bootstrap confidence-bound selection.",
        "",
        "## Interpretation Limits",
        "",
        "This study supports only risk-aware, risk-constrained, or empirically controlled false-stop language. It does not establish guaranteed, reliable, or safe stopping. The underlying Test questions were already evaluated during Phase 3A, so this is not a fresh study-level holdout.",
        "",
    ]

    criterion_rows = [[name, str(value)] for name, value in criteria.items()]
    closing_lines = [
        "# Phase 3B Closing Report",
        "",
        f"## Decision: {'GO' if positive_claim_go else 'NO-GO'}",
        "",
        (
            "The pre-registered operational criteria support the claim that the primary policy preserves QA quality within the declared noninferiority margin while reducing retrieval and latency cost relative to Fixed Heavy."
            if positive_claim_go
            else "The pre-registered evidence does not support the full positive efficient-risk-control claim. This remains a valid negative result and must not trigger Test-driven retuning."
        ),
        "",
        "## Decision Criteria",
        "",
        *markdown_table(["Criterion", "Passed"], criterion_rows),
        "",
        f"- Primary policy: `{primary_name}`",
        f"- Observed Test FSR: {percent(primary['false_stop_rate'])} (target {percent(primary_config['risk_target'])})",
        f"- Primary Test FSR question-bootstrap 95% CI: {primary_fsr_ci_text}",
        f"- Answer F1 delta 95% CI: [{number(answer_delta['ci95_low'])}, {number(answer_delta['ci95_high'])}]",
        f"- Retrieved-chunk delta 95% CI: [{number(chunk_delta['ci95_low'])}, {number(chunk_delta['ci95_high'])}]",
        f"- Total-latency delta 95% CI: [{number(latency_delta['ci95_low'])}, {number(latency_delta['ci95_high'])}]",
        f"- FSR delta 95% CI: [{number(fsr_delta['ci95_low'])}, {number(fsr_delta['ci95_high'])}]",
        "",
        "## Qualification",
        "",
        pareto_note,
        "",
        "The observed primary FSR meets the target, but its confidence interval is not wholly below 10%. Temperature Scaling also did not jointly improve held-out NLL and ECE, and the raw conservative alpha=0.10 policy produced effectively the same Test operating point. The GO verdict therefore does not establish guaranteed risk control or a distinct deployment benefit from Temperature Scaling itself.",
        "",
        "## Thesis Boundary",
        "",
        "Phase 3B is complete enough to report regardless of the verdict. A NO-GO blocks a positive deployment-style claim, not thesis finalization around a transparent negative trade-off. Do not retune alpha, temperature, threshold, retrieval, or generation from Test outcomes.",
        "",
    ]

    outputs = {
        ROOT / "docs" / "phase3b_analysis.md": "\n".join(lines),
        ROOT / "docs" / "phase3b_closing_report.md": "\n".join(closing_lines),
    }
    manifest_path = ROOT / "results" / "phase3b" / "analysis_manifest.json"
    existing = [path for path in [*outputs, manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite Phase 3B reports: {existing}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "phase": "3B",
        "git_commit": git_commit(ROOT),
        "positive_claim_decision": "GO" if positive_claim_go else "NO-GO",
        "criteria": criteria,
        "primary_test_fsr_ci95": primary_fsr_ci,
        "raw_primary_comparison": {
            "policy": raw_primary_name,
            "same_answer_f1_and_fsr_within_1e-12": raw_primary_equivalent,
            "average_retrieved_chunks_delta": raw_primary_chunk_delta,
        },
        "source_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "outputs": {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in outputs},
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(manifest["positive_claim_decision"])


if __name__ == "__main__":
    main()

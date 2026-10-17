from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .validation import run_p0_fixtures


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_means(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return frame.groupby("method", sort=True)[columns].mean().reset_index().to_dict("records")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def build_q1_small_report(repo: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo)
    raw = repo / "reports" / "raw"
    analysis = repo / "reports" / "analysis"
    pilot = pd.read_csv(raw / "q1_pilot_small.csv")
    private = pd.read_csv(raw / "q1_e2_private_small.csv")
    longitudinal = pd.read_csv(raw / "q1_e2_longitudinal_small.csv")
    combined = pd.concat([pilot, private, longitudinal], ignore_index=True)
    e6 = pd.read_json(raw / "q1_e6_audit_small.json")
    e7 = pd.read_json(raw / "q1_e7_availability_small.json")
    power = {
        metric: json.loads((analysis / filename).read_text(encoding="utf-8"))
        for metric, filename in {
            "worst_group_rmse": "q1_power_worst_group_rmse.json",
            "rmse": "q1_power_rmse.json",
            "restricted_mean_aoi": "q1_power_aoi.json",
        }.items()
    }

    pilot_means = _group_means(
        pilot,
        [
            "rmse",
            "worst_group_rmse",
            "restricted_mean_aoi",
            "mean_max_service_deficit",
            "runtime_seconds",
            "peak_memory_mb",
        ],
    )
    means_by_method = {row["method"]: row for row in pilot_means}
    airproof = means_by_method["airproof"]
    prediction_candidates = [row for row in pilot_means if row["method"] != "airproof"]
    best_worst = min(prediction_candidates, key=lambda row: row["worst_group_rmse"])
    best_overall = min(prediction_candidates, key=lambda row: row["rmse"])
    best_freshness = min(prediction_candidates, key=lambda row: row["restricted_mean_aoi"])
    validation_decision = {
        "primary_claims_unlocked": False,
        "worst_group_comparator": best_worst["method"],
        "worst_group_airproof_minus_baseline": (
            airproof["worst_group_rmse"] - best_worst["worst_group_rmse"]
        ),
        "overall_comparator": best_overall["method"],
        "overall_airproof_minus_baseline": airproof["rmse"] - best_overall["rmse"],
        "freshness_comparator": best_freshness["method"],
        "freshness_airproof_minus_baseline": (
            airproof["restricted_mean_aoi"] - best_freshness["restricted_mean_aoi"]
        ),
        "interpretation": (
            "Validation smoke does not support the current end-to-end superiority or "
            "noninferiority claims; retain the negative result and do not open primary seeds."
        ),
    }

    private_summary = (
        private.groupby(["epsilon_mean", "epsilon_count", "k_min"], sort=True)[
            [
                "release_count",
                "suppressed_release_count",
                "release_rmse",
                "max_composed_user_epsilon",
            ]
        ]
        .mean()
        .reset_index()
        .to_dict("records")
    )
    longitudinal_summary = (
        longitudinal.groupby(
            ["epsilon_user_max", "participation_skew", "method"], sort=True
        )[
            [
                "release_count",
                "suppressed_release_count",
                "declined_privacy_users",
                "max_composed_user_epsilon",
            ]
        ]
        .mean()
        .reset_index()
        .to_dict("records")
    )
    accountant_valid = bool(
        (
            longitudinal["max_composed_user_epsilon"]
            <= longitudinal["epsilon_user_max"] + 1e-12
        ).all()
    )
    e6_summary = (
        e6.groupby(["method", "batch_size"], sort=True)[
            [
                "append_ns_per_record",
                "proof_bytes",
                "verify_ns",
                "consistency_proof_bytes",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict("records")
    )
    e7_half_loss = e7[e7["loss_probability"] == 0.5].to_dict("records")
    p0 = run_p0_fixtures()

    contract_log = repo / "reports" / "q1_contract_test.log"
    gas_match = re.search(
        r'"gasUsed":"(\d+)"', contract_log.read_text(encoding="utf-8", errors="replace")
    )
    artifact_paths = sorted(
        [
            *raw.glob("q1_*"),
            *analysis.glob("q1_*"),
            contract_log,
        ]
    )
    artifacts = {
        str(path.relative_to(repo)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in artifact_paths
        if path.is_file()
    }
    pilot_manifest = json.loads(
        (raw / "q1_pilot_small.manifest.json").read_text(encoding="utf-8")
    )
    return _json_safe({
        "status": "completed-validation-small-studies-not-primary-evidence",
        "code_version": __version__,
        "source_tree_sha256": pilot_manifest["source_tree_sha256"],
        "registry_sha256": pilot_manifest["registry_sha256"],
        "p0": p0,
        "simulation_rows": len(combined),
        "unique_run_ids": int(combined["run_id"].nunique()),
        "summed_method_runtime_minutes": float(combined["runtime_seconds"].sum() / 60),
        "max_peak_rss_mb": float(combined["peak_memory_mb"].max()),
        "solver_failure_rows": int((combined["solver_failure_rate"] > 0).sum()),
        "pilot_means": pilot_means,
        "power": power,
        "validation_decision": validation_decision,
        "private_eligibility_summary": private_summary,
        "longitudinal_summary": longitudinal_summary,
        "accountant_budget_invariant": accountant_valid,
        "e6_all_inclusion_proofs_valid": bool(e6["proof_valid"].all()),
        "e6_all_consistency_proofs_valid": bool(
            e6["consistency_valid"].dropna().astype(bool).all()
        ),
        "e6_summary": e6_summary,
        "e7_half_loss": e7_half_loss,
        "contract_anchor_batch_gas": int(gas_match.group(1)) if gas_match else None,
        "artifacts": artifacts,
        "deferred": [
            "E1 full primary",
            "E2 full interactions",
            "E3 factorial primary",
            "E4 attacks primary",
            "E5 5k/10k scale",
            "EPA/ERA5/ACS primary archived regime",
            "Sepolia sample",
            "production ZK/Semaphore adapter",
        ],
    })


def write_q1_small_report(report: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report = {**report, "report_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    decision = report["validation_decision"]
    lines = [
        "# AirProof Q1 small-study report",
        "",
        f"Status: **{report['status']}**",
        "",
        (
            f"Executable evidence: {len(report['p0']) - 1} P0 checks; "
            f"{report['simulation_rows']} simulation rows / {report['unique_run_ids']} unique runs; "
            f"solver-failure rows = {report['solver_failure_rows']}."
        ),
        "",
        "## Validation pilot",
        "",
        "| Method | RMSE | Worst-group RMSE | Restricted mean AoI | Mean max deficit |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["pilot_means"]:
        lines.append(
            f"| {row['method']} | {row['rmse']:.4f} | {row['worst_group_rmse']:.4f} | "
            f"{row['restricted_mean_aoi']:.4f} | {row['mean_max_service_deficit']:.4f} |"
        )
    lines += [
        "",
        "This is validation-only evidence. " + decision["interpretation"],
        "",
        (
            f"AirProof minus endpoint-best validation baseline: worst-group RMSE "
            f"{decision['worst_group_airproof_minus_baseline']:.4f}; overall RMSE "
            f"{decision['overall_airproof_minus_baseline']:.4f}; restricted mean AoI "
            f"{decision['freshness_airproof_minus_baseline']:.4f}. Lower is better."
        ),
        "",
        (
            "Power sizing recommends 30 independent worlds for each registered primary "
            "endpoint; primary execution remains locked."
        ),
        "",
        "## Privacy, audit and availability",
        "",
        (
            f"Longitudinal accountant respected every configured budget: "
            f"**{report['accountant_budget_invariant']}**. Strict private eligibility produced "
            "substantial suppression in small cohorts; this is retained as the privacy--utility result."
        ),
        (
            f"E6 inclusion proofs all valid: **{report['e6_all_inclusion_proofs_valid']}**; "
            f"consistency proofs all valid: **{report['e6_all_consistency_proofs_valid']}**."
        ),
        f"Local Hardhat `anchorBatch` gas: {report['contract_anchor_batch_gas']}.",
        "",
        "## Resource envelope",
        "",
        (
            f"Summed method runtime: {report['summed_method_runtime_minutes']:.2f} minutes; "
            f"maximum observed process RSS: {report['max_peak_rss_mb']:.1f} MB."
        ),
        "",
        "## Deferred large work",
        "",
    ]
    lines.extend(f"- {item}" for item in report["deferred"])
    lines += [
        "",
        f"Source tree SHA-256: `{report['source_tree_sha256']}`",
        "",
        f"Registry SHA-256: `{report['registry_sha256']}`",
        "",
        f"Report SHA-256: `{report['report_sha256']}`",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

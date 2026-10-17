from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _means(frame: pd.DataFrame, metrics: list[str]) -> list[dict]:
    return frame.groupby("method", sort=True)[metrics].mean().reset_index().to_dict(orient="records")


def _factor_effect(frame: pd.DataFrame, bit: int, metric: str) -> float:
    values: list[float] = []
    for seed, group in frame.groupby("seed"):
        del seed
        on = group[group["method"].str[10 + bit] == "1"][metric].mean()
        off = group[group["method"].str[10 + bit] == "0"][metric].mean()
        values.append(float(on - off))
    return float(np.mean(values))


def _attack_did(frame: pd.DataFrame, method: str, kind: str, fraction: float, metric: str) -> float:
    subset = frame[(frame["method"] == method) & (frame["attack_kind"] == kind)]
    attacked = subset[np.isclose(subset["attack_fraction"], fraction)].groupby("seed")[metric].mean()
    clean = subset[np.isclose(subset["attack_fraction"], 0.0)].groupby("seed")[metric].mean()
    common = attacked.index.intersection(clean.index)
    return float((attacked.loc[common] - clean.loc[common]).mean())


def build_measured_report(repo: str | Path = ".") -> dict:
    repo = Path(repo)
    raw = repo / "reports" / "raw"
    analysis = repo / "reports" / "analysis"
    frozen = pd.read_csv(raw / "measured_frozen.csv")
    e2 = pd.read_csv(raw / "measured_e2_privacy.csv")
    e3_nominal = pd.read_csv(raw / "measured_frozen_e3_nominal.csv")
    e3_severe = pd.read_csv(raw / "measured_frozen_e3_severe.csv")
    e4 = pd.read_csv(raw / "measured_frozen_e4_attack.csv")
    e5_agents = pd.read_csv(raw / "measured_e5_agents.csv")
    e5_grid = pd.read_csv(raw / "measured_e5_grid.csv")
    pilot = pd.read_csv(raw / "measured_pilot_huber.csv")
    paired = json.loads((analysis / "measured_frozen.json").read_text(encoding="utf-8"))
    archived = json.loads((analysis / "beijing_transfer.json").read_text(encoding="utf-8"))
    primary_frames = [frozen, e2, e3_nominal, e3_severe, e4, e5_agents, e5_grid, pilot]
    combined = pd.concat(primary_frames, ignore_index=True)

    privacy = (
        e2.groupby(["epsilon_mean", "method"])[
            ["release_rmse", "suppressed_release_count", "declined_privacy_users"]
        ]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    scale_agents = (
        e5_agents.groupby("agents")[["runtime_seconds", "peak_memory_mb", "rmse", "delivery_ratio"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    scale_grid = (
        e5_grid.groupby("grid_side")[["runtime_seconds", "peak_memory_mb", "rmse", "solver_failure_rate"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    factor_metrics = ["worst_group_rmse", "rmse", "aoi_p95", "coverage_gap"]
    factorial = {}
    for regime, frame in (("nominal", e3_nominal), ("severe", e3_severe)):
        factorial[regime] = {
            name: {metric: _factor_effect(frame, bit, metric) for metric in factor_metrics}
            for bit, name in enumerate(("fairness_on_minus_off", "relay_on_minus_off", "huber_on_minus_off"))
        }
    attack = {}
    for kind in ("gradual_drift", "hotspot_suppression", "inlier"):
        attack[kind] = {
            method: _attack_did(e4, method, kind, 0.2, "worst_group_rmse")
            for method in ("airproof", "squared_loss", "opportunistic_ttl")
        }

    contract_log = repo / "reports" / "contract_test.log"
    gas = None
    if contract_log.exists():
        match = re.search(r'"gasUsed":"(\d+)"', contract_log.read_text(encoding="utf-8", errors="replace"))
        gas = int(match.group(1)) if match else None
    report = {
        "status": "measured-workstation-smoke-not-primary-confirmatory",
        "run_rows": len(combined),
        "unique_run_ids": int(combined["run_id"].nunique()),
        "frozen_main_rows": len(frozen),
        "frozen_main_world_cells": int(frozen[["config_hash", "seed"]].drop_duplicates().shape[0]),
        "solver_failure_rows": int((combined["solver_failure_rate"] > 0).sum()),
        "frozen_main_means": _means(frozen, ["rmse", "worst_group_rmse", "aoi_p95", "coverage_gap", "delivery_ratio"]),
        "frozen_paired_comparisons": paired["comparisons"],
        "rq1_supported": False,
        "rq1_reason": "Worst-group and overall RMSE are worse than the strongest applicable smoke baselines; AoI ties opportunistic TTL.",
        "privacy_curve": privacy,
        "factorial_main_effects": factorial,
        "attack_increment_at_20_percent": attack,
        "scale_agents": scale_agents,
        "scale_grid": scale_grid,
        "archived_transfer": archived["aggregate"],
        "archived_dataset_sha256": archived["dataset_sha256"],
        "contract_anchor_batch_gas": gas,
        "limitations": [
            "Six test seeds per main scenario are smoke evidence, below the pre-registered 30-100 world target.",
            "Synthetic mobility/contact behavior is not field validation.",
            "Beijing is regulatory archived transfer evidence, not citizen-sensing or equity validation.",
            "No public-testnet transaction was sent because no user-authorized credential/funding context was provided.",
        ],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report


def write_measured_report(report: dict, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    means = {row["method"]: row for row in report["frozen_main_means"]}
    lines = [
        "# AirProof measured implementation report",
        "",
        f"Status: **{report['status']}**",
        "",
        (
            f"Evidence ledger: {report['run_rows']} rows, {report['unique_run_ids']} unique run IDs; "
            f"frozen main = {report['frozen_main_rows']} rows over "
            f"{report['frozen_main_world_cells']} paired world-cells."
        ),
        "",
        "## Frozen main means",
        "",
        "| Method | RMSE | Worst-group RMSE | AoI p95 | Coverage gap | Delivery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in sorted(means.items()):
        lines.append(
            f"| {method} | {row['rmse']:.4f} | {row['worst_group_rmse']:.4f} | "
            f"{row['aoi_p95']:.4f} | {row['coverage_gap']:.4f} | {row['delivery_ratio']:.4f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"RQ1 supported: **{report['rq1_supported']}**. {report['rq1_reason']}",
        "",
        (
            "The smoke run therefore falsifies the current end-to-end improvement hypothesis at this scale. "
            "The result is retained; it is not replaced by the paper's target values."
        ),
        "",
        "## Other measured branches",
        "",
        f"- Archived Beijing station transfer: {len(report['archived_transfer'])} methods; IDW is the observed strongest comparator.",
        f"- Contract local anchorBatch gas: {report['contract_anchor_batch_gas']} gas (Hardhat, optimizer 200 runs).",
        f"- Rows with any solver nonconvergence: {report['solver_failure_rows']} (retained, not imputed).",
        "- Privacy, factorial, attack and scale details are in the adjacent JSON and machine-readable CSV files.",
        "",
        "## Boundaries",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += ["", f"Report SHA-256: `{report['report_sha256']}`", ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

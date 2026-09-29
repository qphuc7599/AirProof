"""Analyze all registered DP cells; no silent removal of suppressed outcomes."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp


FIELDS = ("epsilon_user_max", "scheduled_epochs", "k_min", "private_eligibility", "count_budget_fraction", "release_mode")


def setting_key(row, *, mode=True):
    return tuple(row[field] for field in FIELDS if mode or field != "release_mode")


def paired_effect(raw, residual, *, draws=10000):
    raw, residual = np.asarray(raw, float), np.asarray(residual, float)
    if raw.shape != residual.shape or raw.ndim != 1 or len(raw) < 2 or not np.isfinite(raw).all() or not np.isfinite(residual).all() or np.any(raw <= 0):
        raise ValueError("finite matched positive-RMSE world pairs required")
    gain = 1 - residual / raw
    rng = np.random.default_rng(20260903)
    samples = gain[rng.integers(len(raw), size=(draws, len(raw)))].mean(axis=1)
    return {"mean_paired_rmse_reduction": float(gain.mean()), "lower95": float(np.quantile(samples, .025)),
            "upper95": float(np.quantile(samples, .975)), "paired_worlds": len(raw),
            "raw_mean_rmse": float(raw.mean()), "residual_mean_rmse": float(residual.mean())}


def analyze(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    worlds = [json.loads(line) for line in (directory / "worlds.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_worlds = {(row["seed"], row["cell"], row["config_hash"]) for row in manifest["job_plan"]}
    if len(worlds) != len(expected_worlds) or {(row["seed"], row["cell"], row["config_hash"]) for row in worlds} != expected_worlds:
        raise ValueError("incomplete/duplicate physical world matrix")
    expected = {(seed, cell, cfg, setting_key(setting)) for seed, cell, cfg in expected_worlds for setting in manifest["expected_settings"]}
    actual = {(row["seed"], row["cell"], row["config_hash"], setting_key(row)) for row in rows}
    if len(rows) != len(actual) or actual != expected or len(rows) != completion["rows"]:
        raise ValueError("incomplete/duplicate privacy interaction matrix")
    if {row["source_hash"] for row in rows} != {manifest["source_hash"]}:
        raise ValueError("mixed source inside interaction campaign")
    if any(not row["public_baseline_citizen_independent"] or row["counts_released"]
           or row["privacy_budget_violation_count"] for row in rows):
        raise ValueError("privacy contract violation")
    indexed = {(row["seed"], row["cell"], setting_key(row)): row for row in rows}
    settings = [setting for setting in manifest["expected_settings"] if setting["release_mode"] == "raw"]
    protocol = manifest["protocol"]
    summary, primary = [], None
    for cell in protocol["cells"]:
        for setting in settings:
            raw, residual = [], []
            for seed in protocol["seeds"]:
                raw.append(indexed[seed, cell, setting_key(setting)])
                residual.append(indexed[seed, cell, setting_key({**setting, "release_mode": "residual"})])
            for a, b in zip(raw, residual, strict=True):
                if a["release_count"] != b["release_count"] or a["composed_epsilon"] != b["composed_epsilon"]:
                    raise ValueError("raw/residual comparators had different support or privacy exposure")
                if (a["release_rmse"] is None) != (b["release_rmse"] is None):
                    raise ValueError("raw/residual missing-support mismatch")
            supported = [(a, b) for a, b in zip(raw, residual, strict=True) if a["release_rmse"] is not None]
            result = {"cell": cell, **{key: value for key, value in setting.items() if key != "release_mode"},
                "total_worlds": len(raw), "worlds_with_releases": len(supported),
                "worlds_without_releases": len(raw) - len(supported),
                "mean_scored_releases": float(np.mean([a["release_count"] for a in raw])),
                "mean_all_epoch_release_fraction": float(np.mean([a["all_epoch_release_fraction"] for a in raw])),
                "mean_release_age_hours_at_collection_close": float(np.mean([a["mean_age_of_last_release_at_query_closure"] for a in raw])),
                "paired_utility": paired_effect([a["release_rmse"] for a, _ in supported],
                    [b["release_rmse"] for _, b in supported], draws=protocol["analysis"]["bootstrap_replicates"])
                    if len(supported) >= 2 else None,
                "inference_scope": "conditional-on-emitted-cohorts; report-suppression-and-cadence-alongside-RMSE"}
            summary.append(result)
            claim = protocol["analysis"]
            if cell == claim["primary_secondary_claim_cell"] and all(setting[field] == claim["primary_secondary_claim_" + field]
                    for field in ("epsilon_user_max", "scheduled_epochs", "k_min", "private_eligibility")):
                if len(supported) != len(protocol["seeds"]):
                    raise ValueError("prespecified all-world privacy contrast lacks support")
                contrast = np.array([b["release_rmse"] - (1 - claim["reduction_target"]) * a["release_rmse"] for a, b in supported])
                test = ttest_1samp(contrast, 0., alternative="less")
                primary = {"setting": result, "contrast": "residual_RMSE_minus_0.50_raw_RMSE",
                    "mean_contrast": float(contrast.mean()), "t_statistic": float(test.statistic),
                    "one_sided_p": float(test.pvalue), "degrees_freedom": len(contrast) - 1,
                    "statistical_family": "single-prespecified-secondary-mechanism-contrast; not-seven-primary-twin-family",
                    "rest_of_grid": "descriptive-only-no-grid-wide-significance-claim"}
    if primary is None:
        raise ValueError("registered secondary contrast not found")
    report = {"role": "complete-prespecified-secondary-privacy-interactions", "rows": len(rows),
        "physical_worlds": len(worlds), "independent_seed_worlds_per_cell": len(protocol["seeds"]),
        "source_hash": manifest["source_hash"], "zero_accounting_violations": True,
        "prespecified_secondary_contrast": primary, "interaction_cells": summary,
        "resource_contract": protocol["resource_contract"], "privacy_scope": protocol["privacy_scope"],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    report = analyze(args.campaign.resolve())
    print(json.dumps({"rows": report["rows"], "primary_secondary_contrast": report["prespecified_secondary_contrast"]}, indent=2))


if __name__ == "__main__":
    main()

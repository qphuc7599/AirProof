"""Registered world-paired core contrasts and complete-matrix validation."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


AP, SQ, NF, NR = "airproof_v4", "squared_loss_v4", "airproof_v4_no_fairness", "airproof_v4_no_relay"


def load_core(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {(row["seed"], row["cell"], row["method"], row["config_hash"]) for row in manifest["job_plan"]}
    actual = {(row["seed"], row["physical_cell"], row["method"], row["config_hash"]) for row in rows}
    if actual != expected or len(actual) != len(rows) or len(rows) != completion["completed"] or len(rows) != manifest["total_jobs"]:
        raise ValueError("incomplete/duplicate/misconfigured core campaign")
    if {row["source_tree_sha256"] for row in rows} != {manifest["source_hash"]}:
        raise ValueError("mixed numerical sources")
    indexed = {(row["seed"], row["physical_cell"], row["method"]): row for row in rows}
    for row in rows:
        metrics = row["metrics"]
        if metrics["scheduler_allocation_clock"] != "arrival" or metrics["predictive_adaptive_delta"]:
            raise ValueError("wrong final estimator/scheduler architecture")
        if not metrics["release_baseline_citizen_independent"] or metrics["max_composed_user_epsilon"] > 8 + 1e-10:
            raise ValueError("public-baseline/history-budget contract violated")
        if metrics["privacy_scheduled_epochs"] != 28 or metrics["python_allocation_tracing_enabled"]:
            raise ValueError("prospective execution/release cadence changed")
        if metrics["solver_failure_rate"] or metrics["privacy_budget_violation_count"] or metrics["constraint_violation_when_feasible_epochs"]:
            raise ValueError("numerical/feasible-floor/privacy violation in core campaign")
    for seed in manifest["seeds"]:
        for cell in manifest["protocol"]["cells"]:
            a, b = indexed[seed, cell, AP], indexed[seed, cell, SQ]
            if a["world"] != b["world"] or a["config_hash"] != b["config_hash"]:
                raise ValueError("core comparators are not paired on the same physical world")
        public = [indexed[seed, cell, AP]["metrics"]["public_backbone_calibration"] for cell in
                  ("severe_clean", "severe_adversarial_drift", "severe_hotspot_suppression")]
        if public[0] != public[1] or public[0] != public[2]:
            raise ValueError("public calibration depends on private corruption")
        for method in (NF, NR):
            if indexed[seed, "severe_clean", method]["world"] != indexed[seed, "severe_clean", AP]["world"]:
                raise ValueError("unpaired ablation world")
    return manifest, indexed, rows


def vector(indexed, seeds, cell, method, metric="rmse"):
    values = np.array([indexed[seed, cell, method]["metrics"][metric] for seed in seeds], float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite world endpoint")
    return values


def core_contrasts(indexed, seeds):
    result = {}
    for number, cell in enumerate(("anchor_clean", "outage_clean", "severe_clean"), 1):
        result[f"H{number}"] = vector(indexed, seeds, cell, AP) - 1.05 * vector(indexed, seeds, cell, SQ)
    for number, cell in enumerate(("severe_adversarial_drift", "severe_hotspot_suppression"), 4):
        ap_growth = vector(indexed, seeds, cell, AP) - vector(indexed, seeds, "severe_clean", AP)
        sq_growth = vector(indexed, seeds, cell, SQ) - vector(indexed, seeds, "severe_clean", SQ)
        result[f"H{number}"] = ap_growth - .8 * sq_growth
    result["H6"] = vector(indexed, seeds, "severe_clean", AP, "coverage_gap") - .8 * vector(indexed, seeds, "severe_clean", NF, "coverage_gap")
    result["H7"] = vector(indexed, seeds, "severe_clean", AP, "worst_group_rmse") - .9 * vector(indexed, seeds, "severe_clean", NF, "worst_group_rmse")
    return result


def holm_adjust(p_values):
    names = sorted(p_values, key=lambda name: p_values[name])
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("invalid p values")
    result, maximum = {}, 0.
    for index, name in enumerate(names):
        maximum = max(maximum, min(1., (len(names) - index) * p_values[name]))
        result[name] = maximum
    return result


def paired_ratio(numerator, denominator, *, reduction=False, draws=10000):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    if numerator.shape != denominator.shape or len(numerator) < 2:
        raise ValueError("paired world vectors required")
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all() or denominator.mean() <= 0:
        raise ValueError("positive mean denominator required")
    rng = np.random.default_rng(20260903)
    indices = rng.integers(len(numerator), size=(draws, len(numerator)))
    denominators = denominator[indices].mean(axis=1)
    valid = denominators > 0
    samples = numerator[indices].mean(axis=1)[valid] / denominators[valid]
    point = numerator.mean() / denominator.mean()
    if reduction:
        samples, point = 1 - samples, 1 - point
    return {"estimate": float(point), "lower95": float(np.quantile(samples, .025)) if valid.all() else None,
            "upper95": float(np.quantile(samples, .975)) if valid.all() else None,
            "nonpositive_bootstrap_denominators": int((~valid).sum()),
            "nonpositive_world_denominators": int((denominator <= 0).sum()),
            "statistic": "one-minus-ratio-of-paired-world-means" if reduction else "ratio-of-paired-world-means"}

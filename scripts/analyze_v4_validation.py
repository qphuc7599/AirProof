"""Summarize a complete paired v4 validation snapshot; never imply final confirmation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airproof.statistics import paired_bootstrap_ci


def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or len(values) < 2:
        raise ValueError("at least two finite paired world estimates are required")
    low, high = paired_bootstrap_ci(values, draws=10000, seed=20260903)
    return {"mean": float(values.mean()), "lower95": low, "upper95": high,
            "worlds": len(values), "sd": float(values.std(ddof=1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.campaign / "manifest.json").read_text(encoding="utf-8"))
    if not (args.campaign / "completion.json").exists():
        raise ValueError("campaign has no completion record; do not summarize a partial matrix")
    rows = [json.loads(line) for line in (args.campaign / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    seeds = manifest["seeds"]
    kinds = ("clean", "adversarial_drift", "hotspot_suppression")
    methods = ("airproof_v4", "squared_loss_v4")
    expected = {(seed, kind, method) for seed in seeds for kind in kinds for method in methods}
    indexed = {(row["seed"], row["scenario"]["attack_kind"], row["method"]): row for row in rows}
    if len(rows) != len(indexed) or set(indexed) != expected or len(rows) != manifest["total_jobs"]:
        raise ValueError("matrix is incomplete or duplicated")
    if {row["source_tree_sha256"] for row in rows} != {manifest["source_hash"]}:
        raise ValueError("source mismatch inside campaign")
    for seed in seeds:
        for kind in kinds:
            left, right = (indexed[seed, kind, method] for method in methods)
            if left["config_hash"] != right["config_hash"] or left["world"] != right["world"]:
                raise ValueError("paired methods have different configuration or physical world")
        selections = [indexed[seed, kind, "airproof_v4"]["metrics"]["public_backbone_calibration"]
                      for kind in kinds]
        if any(value != selections[0] for value in selections):
            raise ValueError("public model unexpectedly depends on the citizen attack")

    def metric(seed, kind, method, name="rmse"):
        return float(indexed[seed, kind, method]["metrics"][name])

    contrasts = {}
    for kind in kinds:
        ratios = [metric(seed, kind, methods[0]) / metric(seed, kind, methods[1]) for seed in seeds]
        contrast = {"rmse_ratio": interval(ratios)}
        if kind != "clean":
            controls = [metric(seed, kind, methods[1]) - metric(seed, "clean", methods[1]) for seed in seeds]
            candidates = [metric(seed, kind, methods[0]) - metric(seed, "clean", methods[0]) for seed in seeds]
            attenuations = [1 - new / old for new, old in zip(candidates, controls, strict=True) if old > 0]
            contrast["undefined_attenuation_worlds"] = sum(old <= 0 for old in controls)
            contrast["rmse_growth_attenuation"] = interval(attenuations) if len(attenuations) >= 2 else None
            contrast["paired_growth_contrast"] = interval(np.asarray(candidates) - controls)
        contrasts[kind] = contrast
    reductions = [metric(seed, kind, methods[0], "residual_dp_rmse_reduction")
                  for seed in seeds for kind in kinds]
    # Cluster by world rather than treating three attack cells as independent worlds.
    dp_world_reductions = np.asarray(reductions).reshape(len(seeds), len(kinds)).mean(axis=1)
    summary = {
        "role": "validation-selection-only-not-final-confirmation",
        "rows": len(rows), "seeds": seeds, "source_hash": manifest["source_hash"],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "confidence_intervals": "10000 paired world bootstrap draws; descriptive, not multiplicity-adjusted final tests",
        "contrasts": contrasts, "same_budget_dp_rmse_reduction": interval(dp_world_reductions),
        "max_solver_failure_rate": max(row["metrics"]["solver_failure_rate"] for row in rows),
        "privacy_budget_violations": sum(row["metrics"]["privacy_budget_violation_count"] for row in rows),
        "floor_violations_when_feasible": sum(row["metrics"]["constraint_violation_when_feasible_epochs"] for row in rows),
        "all_public_baselines": all(row["metrics"]["release_baseline_citizen_independent"] for row in rows),
        "allocation_clocks": sorted({row["metrics"].get("scheduler_allocation_clock", "legacy-acquisition-snapshot") for row in rows}),
    }
    (args.campaign / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

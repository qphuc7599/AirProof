"""Select from the complete registered final-estimator fairness frontier."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("finite paired world estimates required")
    rng = np.random.default_rng(20260903)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lower95": float(np.quantile(means, .025)),
            "upper95": float(np.quantile(means, .975)), "worlds": len(values),
            "sd": float(values.std(ddof=1))}


def analyze(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {(item["seed"], item["label"], item["method"], item["config_hash"]) for item in manifest["job_plan"]}
    observed = {(row["seed"], row["frontier_candidate"], row["method"], row["config_hash"]) for row in rows}
    if len(rows) != len(observed) or observed != expected or len(rows) != completion["completed"]:
        raise ValueError("incomplete/duplicated/misconfigured frontier")
    if {row["source_tree_sha256"] for row in rows} != {manifest["source_hash"]}:
        raise ValueError("mixed source inside frontier")
    indexed = {(row["seed"], row["frontier_candidate"]): row for row in rows}
    seeds = manifest["seeds"]
    for seed in seeds:
        per_world = [row for row in rows if row["seed"] == seed]
        control = indexed[seed, "no_fairness"]
        if any(row["world"] != control["world"] for row in per_world):
            raise ValueError("fairness settings changed the physical world")
        if len({row["metrics"]["intersectional_audit"]["definition"]["mask_sha256"] for row in per_world}) != 1:
            raise ValueError("audit groups changed with treatment")
        for row in per_world:
            metrics = row["metrics"]
            if metrics["scheduler_allocation_clock"] != "arrival" or metrics["predictive_adaptive_delta"]:
                raise ValueError("frontier is not the causal single-radius architecture")
            if not metrics["release_baseline_citizen_independent"]:
                raise ValueError("private baseline entered final frontier")
        for label, settings in manifest["candidates"].items():
            if settings["strength"] == 0:
                for metric in ("rmse", "live_rmse", "worst_group_rmse", "coverage_gap"):
                    if not np.isclose(indexed[seed, label]["metrics"][metric], control["metrics"][metric], rtol=1e-10, atol=1e-10):
                        raise ValueError("lambda-zero does not reproduce the explicit no-fairness control")

    def vector(label, metric):
        return np.array([indexed[seed, label]["metrics"][metric] for seed in seeds], dtype=float)

    results = {}
    protocol = manifest["protocol"]
    for label, settings in manifest["candidates"].items():
        metrics = [indexed[seed, label]["metrics"] for seed in seeds]
        effects = {}
        for field in ("rmse", "live_rmse", "worst_group_rmse", "live_worst_group_rmse", "coverage_gap"):
            base, current = vector("no_fairness", field), vector(label, field)
            effects[field] = interval((current - base) / np.maximum(np.abs(base), 1e-12))
        violations = {
            "feasible_floor": int(sum(item["constraint_violation_when_feasible_epochs"] for item in metrics)),
            "privacy_cap": int(sum(item["privacy_budget_violation_count"] for item in metrics)),
            "max_solver_failure_rate": float(max(item["solver_failure_rate"] for item in metrics)),
        }
        gap_reduction, worst_reduction = -effects["coverage_gap"]["mean"], -effects["worst_group_rmse"]["mean"]
        eligible = (settings["strength"] > 0 and not any(violations.values())
                    and effects["rmse"]["mean"] <= protocol["price_of_fairness_max"]
                    and effects["live_rmse"]["mean"] <= protocol["price_of_fairness_max"])
        score = min(gap_reduction / protocol["coverage_gap_reduction_target"],
                    worst_reduction / protocol["worst_group_rmse_reduction_target"])
        results[label] = {**settings, "paired_relative_changes": effects, "violations": violations,
                          "eligible": bool(eligible), "balanced_target_score": score,
                          "both_fairness_targets_pass": bool(score >= 1),
                          "unconditional_global_floor_feasibility": float(np.mean([
                              item["globally_feasible_epochs"] / item["scoring_steps"] for item in metrics])),
                          "mean_opportunity_shortfall_group_epochs": float(np.mean([
                              item["opportunity_shortfall_group_epochs"] for item in metrics])),
                          "means": {field: float(vector(label, field).mean()) for field in (
                              "rmse", "live_rmse", "worst_group_rmse", "live_worst_group_rmse", "coverage_gap")}}
    eligible = [label for label in results if results[label]["eligible"]]
    selected = max(eligible, key=lambda label: (results[label]["balanced_target_score"],
        -results[label]["paired_relative_changes"]["rmse"]["mean"], results[label]["strength"],
        -results[label]["target"])) if eligible else None
    audit = {}
    if selected is not None:
        labels = indexed[seeds[0], selected]["metrics"]["intersectional_audit"]["strata"]
        for stratum in labels:
            records = [(indexed[seed, selected]["metrics"]["intersectional_audit"]["strata"][stratum],
                        indexed[seed, "no_fairness"]["metrics"]["intersectional_audit"]["strata"][stratum]) for seed in seeds]
            supported = [(a, b) for a, b in records if a["supported_for_worst_stratum"] and b["supported_for_worst_stratum"]]
            audit[stratum] = {"cells_by_world": [a["cells"] for a, _ in records], "supported_worlds": len(supported),
                              "paired_reconstruction_rmse_change": interval([
                                  (a["reconstruction_rmse"] - b["reconstruction_rmse"]) / max(b["reconstruction_rmse"], 1e-12)
                                  for a, b in supported]) if len(supported) >= 2 else None,
                              "paired_live_rmse_change": interval([
                                  (a["live_rmse"] - b["live_rmse"]) / max(b["live_rmse"], 1e-12)
                                  for a, b in supported]) if len(supported) >= 2 else None}
        sys.path.insert(0, str(directory / "source_snapshot"))
        from airproof.config import load_config, with_overrides
        base = load_config(directory / "base_configuration.yaml")
        selected_cfg = with_overrides(base, {"scheduler.target_contributors": results[selected]["target"],
                                            "scheduler.fairness_strength": results[selected]["strength"]})
        (directory / "selected_configuration.yaml").write_text(yaml.safe_dump(selected_cfg, sort_keys=False), encoding="utf-8")
    summary = {"role": "validation-selection-only-not-primary-inference", "source_hash": manifest["source_hash"],
               "rows": len(rows), "seeds": seeds, "candidates": results, "selected": selected,
               "selection_target_pass": bool(selected is not None and results[selected]["both_fairness_targets_pass"]),
               "selected_intersectional_outcome_audit": audit,
               "selection_rule": protocol["selection_rule"], "feasibility_interpretation": protocol["feasibility_interpretation"],
               "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / "selection.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    report = analyze(args.campaign.resolve())
    print(json.dumps({"rows": report["rows"], "selected": report["selected"],
                      "selection_target_pass": report["selection_target_pass"]}, indent=2))


if __name__ == "__main__":
    main()

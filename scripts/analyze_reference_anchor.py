"""Summarize small paired diagnostics without turning pilot ranges into confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-inputs", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    indexed = {(r["scenario"]["attack_kind"], r["seed"], r["method"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate diagnostic cells")
    replacement_keys = set()
    for path in args.candidate_inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            replacement = json.loads(line)
            key = (replacement["scenario"]["attack_kind"], replacement["seed"],
                   replacement["method"])
            if key not in indexed or key in replacement_keys:
                raise ValueError("unknown or duplicated candidate replacement cell")
            if replacement["method"] != "airproof_reference_anchored":
                raise ValueError("candidate replacement must not change comparator rows")
            if replacement["world"] != indexed[key]["world"]:
                raise ValueError("candidate/comparator physical-world metadata differ")
            indexed[key] = replacement
            replacement_keys.add(key)
    if args.candidate_inputs:
        required_replacements = {key for key in indexed if key[2] == "airproof_reference_anchored"}
        if replacement_keys != required_replacements:
            raise ValueError("replace all candidate cells before summarizing an alternative")
        rows = list(indexed.values())
    seeds = sorted({r["seed"] for r in rows})
    methods = ["squared_loss", "airproof_predictive_residual", "airproof_reference_anchored"]
    kinds = ["clean", "adversarial_drift", "hotspot_suppression"]
    expected = {(k, s, m) for k in kinds for s in seeds for m in methods}
    if set(indexed) != expected:
        raise ValueError(f"incomplete paired diagnostic: {len(indexed)}/{len(expected)} cells")

    def value(kind, seed, method, metric="rmse"):
        return float(indexed[kind, seed, method]["metrics"][metric])

    cells = []
    contrasts = []
    for kind in kinds:
        for method in methods:
            cells.append({"kind": kind, "method": method, "worlds": len(seeds), **{
                metric: mean(value(kind, s, method, metric) for s in seeds)
                for metric in ("rmse", "bias", "worst_group_rmse", "release_rmse")
            }})
        for method in methods[1:]:
            paired = []
            for seed in seeds:
                control = value(kind, seed, methods[0])
                row = {"seed": seed, "rmse_ratio": value(kind, seed, method) / control}
                if kind != "clean":
                    control_growth = control - value("clean", seed, methods[0])
                    candidate_growth = value(kind, seed, method) - value("clean", seed, method)
                    row["control_rmse_growth"] = control_growth
                    row["candidate_rmse_growth"] = candidate_growth
                    row["growth_attenuation_pct"] = (
                        100 * (1 - candidate_growth / control_growth) if control_growth > 0 else None
                    )
                paired.append(row)
            contrasts.append({"kind": kind, "method": method, "paired_worlds": paired, **{
                key: {"mean": mean(values), "min": min(values), "max": max(values)}
                for key in ("rmse_ratio", "growth_attenuation_pct")
                if (values := [r[key] for r in paired if r.get(key) is not None])
            }})
    summary = {
        "role": "diagnostic; three-world min/max are not confidence intervals",
        "candidate_inputs": [str(path) for path in args.candidate_inputs],
        "rows": len(rows), "seeds": seeds,
        "agents": rows[0]["scenario"]["agents"],
        "steps": rows[0]["scenario"]["steps"],
        "grid_side": rows[0]["scenario"]["grid_side"],
        "source_tree_sha256": sorted({r["source_tree_sha256"] for r in rows}),
        "cell_means": cells, "paired_contrasts": contrasts,
        "max_solver_failure_rate": max(r["metrics"]["solver_failure_rate"] for r in rows),
        "new_method_public_baseline_independent": all(
            r["metrics"]["release_baseline_citizen_independent"]
            for r in rows if r["method"] == "airproof_reference_anchored"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

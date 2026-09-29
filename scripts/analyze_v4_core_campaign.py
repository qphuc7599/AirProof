"""Analyze all five physical cells and ablations without optional stopping."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

from v4_core_analysis_common import AP, SQ, NF, NR, core_contrasts, holm_adjust, load_core, paired_ratio, vector


def analyze(directory):
    manifest, indexed, rows = load_core(directory)
    seeds, protocol = manifest["seeds"], manifest["protocol"]
    tests = {}
    for name, values in core_contrasts(indexed, seeds).items():
        if values.std(ddof=1) == 0:
            raise ValueError("degenerate variance in registered t contrast; requires explicit reporting")
        result = ttest_1samp(values, 0, alternative="less")
        tests[name] = {"mean_world_contrast": float(values.mean()), "sd": float(values.std(ddof=1)),
                       "t_statistic": float(result.statistic), "one_sided_p": float(result.pvalue), "df": len(seeds) - 1}
    adjusted = holm_adjust({name: item["one_sided_p"] for name, item in tests.items()})
    for name, item in tests.items():
        item["holm_p"] = adjusted[name]
        item["reject_at_familywise_0.05"] = adjusted[name] < protocol["familywise_alpha"]
    comparisons = {}
    for cell in protocol["cells"]:
        main, control = vector(indexed, seeds, cell, AP), vector(indexed, seeds, cell, SQ)
        result = {"reconstruction_ratio_vs_squared": paired_ratio(main, control),
            "live_ratio_vs_squared": paired_ratio(vector(indexed, seeds, cell, AP, "live_rmse"),
                                                   vector(indexed, seeds, cell, SQ, "live_rmse")),
            "mean_rmse": {AP: float(main.mean()), SQ: float(control.mean())},
            "integrated_daily_dp_rmse_reduction": paired_ratio(vector(indexed, seeds, cell, AP, "release_rmse"),
                vector(indexed, seeds, cell, AP, "same_budget_raw_dp_rmse"), reduction=True)}
        if cell.startswith("severe_") and cell != "severe_clean":
            result["growth_attenuation"] = paired_ratio(main - vector(indexed, seeds, "severe_clean", AP),
                control - vector(indexed, seeds, "severe_clean", SQ), reduction=True)
        comparisons[cell] = result
    fairness = {name: paired_ratio(vector(indexed, seeds, "severe_clean", AP, name),
        vector(indexed, seeds, "severe_clean", NF, name), reduction=True) for name in
        ("rmse", "live_rmse", "worst_group_rmse", "live_worst_group_rmse", "coverage_gap")}
    relay = {name: paired_ratio(vector(indexed, seeds, "severe_clean", AP, name),
        vector(indexed, seeds, "severe_clean", NR, name), reduction=True) for name in ("rmse", "live_rmse", "coverage_gap")}
    audit = {}
    reference_audit = indexed[seeds[0], "severe_clean", AP]["metrics"]["intersectional_audit"]
    for seed in seeds:
        a = indexed[seed, "severe_clean", AP]["metrics"]["intersectional_audit"]
        b = indexed[seed, "severe_clean", NF]["metrics"]["intersectional_audit"]
        if a["definition"]["mask_sha256"] != b["definition"]["mask_sha256"]:
            raise ValueError("primary intersectional groups depend on allocator output")
    for label in reference_audit["strata"]:
        pairs = [(indexed[seed, "severe_clean", AP]["metrics"]["intersectional_audit"]["strata"][label],
                  indexed[seed, "severe_clean", NF]["metrics"]["intersectional_audit"]["strata"][label]) for seed in seeds]
        supported = [(a, b) for a, b in pairs if a["supported_for_worst_stratum"] and b["supported_for_worst_stratum"]]
        result = {"cells_by_world": [a["cells"] for a, _ in pairs], "supported_worlds": len(supported),
                  "total_worlds": len(seeds), "minimum_cells_for_support": reference_audit["minimum_cells_for_worst_stratum"]}
        if len(supported) >= 2:
            for metric in ("reconstruction_rmse", "live_rmse"):
                result[metric + "_reduction"] = paired_ratio([a[metric] for a, _ in supported],
                    [b[metric] for _, b in supported], reduction=True)
            values = np.array([100 * (a["citizen_covered_cell_epoch_fraction"] - b["citizen_covered_cell_epoch_fraction"])
                               for a, b in supported])
            rng = np.random.default_rng(20260903)
            samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
            result["citizen_coverage_difference_percentage_points"] = {"estimate": float(values.mean()),
                "lower95": float(np.quantile(samples, .025)), "upper95": float(np.quantile(samples, .975))}
        audit[label] = result
    report = {"stage": manifest["stage"], "source_hash": manifest["source_hash"], "worlds": len(seeds),
        "rows": len(rows), "primary_hypotheses": tests, "all_primary_hypotheses_rejected": all(
            item["reject_at_familywise_0.05"] for item in tests.values()),
        "inferential_scope": "confirmatory-Holm-family" if manifest["stage"] == "primary" else "validation-only-not-primary-confirmation",
        "paired_bootstrap_scope": "descriptive-marginal-95-percent-intervals-not-simultaneous-bands",
        "comparisons": comparisons, "fairness_reductions_vs_no_fairness": fairness,
        "relay_reductions_vs_no_relay": relay,
        "intersectional_outcome_audit": audit,
        "intersectional_scope": "fixed-prefix-outcome-audit-not-overlapping-floor-guarantee-or-income-classification",
        "zero_solver_privacy_feasible_floor_violations": True,
        "maximum_history_epsilon": max(row["metrics"]["max_composed_user_epsilon"] for row in rows),
        "privacy_acquisition_cadence_hours": 24, "privacy_collection_deadline_hours": 24,
        "epoch_update_p95_seconds_max_over_worlds": max(row["metrics"]["epoch_update_p95_seconds"] for row in rows),
        "maximum_sampled_worker_rss_mb": max(row["metrics"]["peak_memory_mb"] for row in rows),
        "all_alloc_profiler_disabled": all(not row["metrics"]["python_allocation_tracing_enabled"] for row in rows),
        "results_sha256": hashlib.sha256((directory / "results.jsonl").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
        "completion_sha256": hashlib.sha256((directory / "completion.json").read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    report = analyze(args.campaign.resolve())
    print(json.dumps({"stage": report["stage"], "worlds": report["worlds"], "hypotheses": report["primary_hypotheses"]}, indent=2))


if __name__ == "__main__":
    main()

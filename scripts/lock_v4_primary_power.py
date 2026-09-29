"""Prospective power and sample-size lock from completed final-config validation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import nct, t

from v4_core_analysis_common import core_contrasts, load_core


def one_sided_power(mean, sd, n, alpha):
    if sd <= 0 or n < 3:
        raise ValueError("positive pilot standard deviation and n>=3 required")
    return float(nct.cdf(t.ppf(alpha, n - 1), n - 1, mean * np.sqrt(n) / sd))


def required_worlds(mean, sd, alpha, target):
    if mean >= 0:
        return None
    if sd <= 0:
        raise ValueError("degenerate pilot variance cannot establish prospective t-test power")
    lower, upper = 3, 3
    while one_sided_power(mean, sd, upper, alpha) < target:
        upper *= 2
        if upper > 100000:
            return None
    while lower < upper:
        middle = (lower + upper) // 2
        if one_sided_power(mean, sd, middle, alpha) >= target:
            upper = middle
        else:
            lower = middle + 1
    return lower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    directory = args.validation.resolve()
    output = directory / "power_lock.json"
    if output.exists():
        raise ValueError("prospective lock already exists; never silently replace it")
    manifest, indexed, _ = load_core(directory)
    if manifest["stage"] != "validation":
        raise ValueError("do not retrospectively size from primary outcomes")
    protocol, seeds = manifest["protocol"], manifest["seeds"]
    rng = np.random.default_rng(20260903)
    bootstrap_indices = rng.integers(len(seeds), size=(10000, len(seeds)))
    hypotheses = {}
    for name, values in core_contrasts(indexed, seeds).items():
        sd_upper = float(np.quantile(values[bootstrap_indices].std(axis=1, ddof=1), .95))
        mean = float(values.mean())
        count = required_worlds(mean, sd_upper, protocol["power_sizing_alpha"], protocol["power_target"])
        hypotheses[name] = {"pilot_mean_contrast": mean, "pilot_sd": float(values.std(ddof=1)),
            "upper95_bootstrap_pilot_sd": sd_upper, "required_worlds": count,
            "effect_in_target_direction": mean < 0,
            "power_at_30": one_sided_power(mean, sd_upper, 30, protocol["power_sizing_alpha"])}
    finite = [item["required_worlds"] for item in hypotheses.values() if item["required_worlds"] is not None]
    proposed = max(protocol["primary_min_worlds"], max(finite, default=protocol["primary_min_worlds"]))
    selected = min(proposed, protocol["primary_max_worlds"])
    for item in hypotheses.values():
        item["power_at_selected_n"] = one_sided_power(item["pilot_mean_contrast"],
            item["upper95_bootstrap_pilot_sd"], selected, protocol["power_sizing_alpha"])
    report = {"role": "prospective-before-any-primary-world", "protocol": protocol,
        "source_hash": manifest["source_hash"], "base_configuration_hash": manifest["base_configuration_hash"],
        "validation_results_sha256": hashlib.sha256((directory / "results.jsonl").read_bytes()).hexdigest(),
        "validation_seeds": seeds, "hypotheses": hypotheses, "unconstrained_proposed_n": proposed,
        "selected_n": selected, "resource_cap_binds": proposed > selected,
        "all_estimated_powers_at_least_target": all(item["power_at_selected_n"] >= protocol["power_target"] for item in hypotheses.values()),
        "primary_seeds": list(range(protocol["primary_seed_start"], protocol["primary_seed_start"] + selected)),
        "assumptions": "plug-in paired-world Gaussian t-test power using validation mean and upper bootstrap SD; conditional planning, not guaranteed achieved power",
        "nonpositive_effect_handling": "no sample size promises a target-direction effect absent in validation; retain this finding, do not claim 90% power",
        "locked_unix": time.time(), "primary_worlds_observed": 0,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected_n": selected, "all_estimated_powers_at_least_target": report["all_estimated_powers_at_least_target"],
                      "hypotheses": hypotheses}, indent=2))


if __name__ == "__main__":
    main()

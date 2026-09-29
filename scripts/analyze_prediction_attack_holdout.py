from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def _interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    rng: np.random.Generator,
    replicates: int = 10_000,
) -> dict[str, Any]:
    observed = float(statistic(values))
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 1_000):
        stop = min(start + 1_000, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = [statistic(values[index]) for index in indices]
    finite = samples[np.isfinite(samples)]
    return {
        "estimate": observed,
        "bootstrap_95_ci": [
            float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)),
        ],
        "bootstrap_replicates": replicates,
        "n_paired_worlds": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    raw_bytes = args.input.read_bytes()
    raw = json.loads(raw_bytes)
    rows = raw["rows"]
    lookup = {
        (int(row["seed"]), str(row["scenario"]), str(row["predictor"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    methods = sorted({str(row["predictor"]) for row in rows})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    expected_scenarios = {"clean", "adversarial_drift20", "hotspot_suppression20"}
    if set(scenarios) != expected_scenarios:
        raise ValueError("holdout is missing a frozen attack scenario")
    if len(rows) != len(seeds) * len(methods) * len(scenarios):
        raise ValueError("holdout cells are incomplete")

    def metric(seed: int, scenario: str, method: str, name: str) -> float:
        value = float(lookup[(seed, scenario, method)]["metrics"][name])
        return abs(value) if name == "bias" else value

    clean_means = {
        method: {
            name: float(np.mean([metric(seed, "clean", method, name) for seed in seeds]))
            for name in ("rmse", "mae", "worst_group_rmse", "bias")
        }
        for method in methods
    }
    scenario_means = {
        scenario: {
            method: {
                name: float(
                    np.mean([metric(seed, scenario, method, name) for seed in seeds])
                )
                for name in ("rmse", "mae", "worst_group_rmse", "bias")
            }
            for method in methods
        }
        for scenario in scenarios
    }
    baseline_methods = [method for method in methods if method != "airproof_predictive_residual"]
    strongest = min(baseline_methods, key=lambda method: clean_means[method]["rmse"])
    rng = np.random.default_rng(args.seed)
    clean_ratios = np.asarray(
        [
            metric(seed, "clean", "airproof_predictive_residual", "rmse")
            / metric(seed, "clean", strongest, "rmse")
            for seed in seeds
        ]
    )
    clean_ratio = _interval(clean_ratios, np.mean, rng=rng)

    attacks: dict[str, Any] = {}
    attacked_ratios: dict[str, Any] = {}
    for scenario in ("adversarial_drift20", "hotspot_suppression20"):
        attack_report: dict[str, Any] = {}
        for name in ("rmse", "worst_group_rmse", "bias"):
            baseline_growth = np.asarray(
                [
                    metric(seed, scenario, "graph_squared", name)
                    - metric(seed, "clean", "graph_squared", name)
                    for seed in seeds
                ]
            )
            airproof_growth = np.asarray(
                [
                    metric(seed, scenario, "airproof_predictive_residual", name)
                    - metric(seed, "clean", "airproof_predictive_residual", name)
                    for seed in seeds
                ]
            )
            paired = np.column_stack([airproof_growth - baseline_growth, baseline_growth])

            def attenuation(sample: np.ndarray) -> float:
                denominator = float(np.mean(sample[:, 1]))
                return float(-np.mean(sample[:, 0]) / denominator) if denominator > 0 else np.nan

            attack_report[name] = {
                "difference_in_differences": _interval(
                    paired[:, 0], np.mean, rng=rng
                ),
                "attenuation_fraction": _interval(paired, attenuation, rng=rng),
                "graph_squared_growth_mean": float(np.mean(baseline_growth)),
                "airproof_growth_mean": float(np.mean(airproof_growth)),
            }
        attacks[scenario] = attack_report
        rmse_ratios = np.asarray(
            [
                metric(seed, scenario, "airproof_predictive_residual", "rmse")
                / metric(seed, scenario, "graph_squared", "rmse")
                for seed in seeds
            ]
        )
        bias_pairs = np.asarray(
            [
                [
                    metric(seed, scenario, "airproof_predictive_residual", "bias"),
                    metric(seed, scenario, "graph_squared", "bias"),
                ]
                for seed in seeds
            ]
        )

        def bias_reduction(sample: np.ndarray) -> float:
            return float(1.0 - np.mean(sample[:, 0]) / np.mean(sample[:, 1]))

        attacked_ratios[scenario] = {
            "airproof_to_graph_squared_rmse_ratio": _interval(
                rmse_ratios, np.mean, rng=rng
            ),
            "airproof_absolute_bias_reduction": _interval(
                bias_pairs, bias_reduction, rng=rng
            ),
        }

    drift = attacks["adversarial_drift20"]["rmse"]["attenuation_fraction"]
    hotspot = attacks["hotspot_suppression20"]["rmse"]["attenuation_fraction"]
    report: dict[str, Any] = {
        "status": "completed-locked-independent-prediction-attack-holdout",
        "interpretation_boundary": (
            f"The estimator was selected on seeds {raw['selection_seeds']} before this "
            f"disjoint holdout {seeds}. Clean noninferiority is relative to the strongest executable "
            "endpoint predictor. Attack attenuation is relative to graph squared under paired "
            "common-random-number worlds."
        ),
        "input": str(args.input).replace("\\", "/"),
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_tree_sha256": raw["source_tree_sha256"],
        "correction_delta": raw["correction_delta"],
        "seeds": seeds,
        "rows": len(rows),
        "clean_means": clean_means,
        "scenario_means": scenario_means,
        "strongest_clean_baseline": strongest,
        "clean_airproof_to_strongest_rmse_ratio": clean_ratio,
        "attacks": attacks,
        "attacked_ratios": attacked_ratios,
        "holdout_gate": {
            "clean_mean_rmse_ratio_lte_1_05": clean_ratio["estimate"] <= 1.05,
            "clean_ci_upper_lte_1_05": clean_ratio["bootstrap_95_ci"][1] <= 1.05,
            "drift_rmse_attenuation_gte_0_08": drift["estimate"] >= 0.08,
            "drift_rmse_attenuation_ci_lower_gt_0": drift["bootstrap_95_ci"][0] > 0.0,
            "hotspot_rmse_attenuation_gte_0_20": hotspot["estimate"] >= 0.20,
            "hotspot_rmse_attenuation_ci_lower_gt_0": hotspot["bootstrap_95_ci"][0] > 0.0,
        },
    }
    report["holdout_gate"]["all_precommitted_gates"] = all(
        report["holdout_gate"].values()
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["holdout_gate"], sort_keys=True))


if __name__ == "__main__":
    main()

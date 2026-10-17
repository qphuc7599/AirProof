from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, wilcoxon


def paired_bootstrap_ci(differences: np.ndarray, *, seed: int = 20260901, draws: int = 5000) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two paired differences are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_permutation_pvalue(
    differences: np.ndarray, *, seed: int = 20260901, draws: int = 20_000
) -> float:
    """Two-sided paired sign-flip randomization test at the world level."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two paired differences are required")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(draws, len(values)))
    sampled = np.abs(np.mean(signs * values, axis=1))
    return float((1 + np.sum(sampled >= observed)) / (draws + 1))


def pilot_power_analysis(
    results: Iterable[dict[str, Any]],
    *,
    metric: str,
    reference: str = "airproof",
    comparator: str | None = None,
    practical_delta: float,
    alpha: float = 0.05,
    target_power: float = 0.8,
    minimum_worlds: int = 30,
    maximum_worlds: int = 100,
    seed: int = 20260901,
    bootstrap_draws: int = 5000,
) -> dict[str, Any]:
    """Validation-only paired pilot sizing with an upper bootstrap SD bound."""
    rows = list(results)
    if practical_delta <= 0 or not 0 < alpha < 1 or not 0 < target_power < 1:
        raise ValueError("invalid power-analysis inputs")
    candidates = sorted({row["method"] for row in rows if row["method"] != reference})
    if comparator is None:
        if not candidates:
            raise ValueError("pilot needs at least one comparator")
        comparator = min(
            candidates,
            key=lambda method: np.mean(
                [float(row["metrics"][metric]) for row in rows if row["method"] == method]
            ),
        )
    paired: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if row["method"] in {reference, comparator} and row["metrics"].get(metric) is not None:
            key = (row["config_hash"], int(row["seed"]))
            paired.setdefault(key, {})[row["method"]] = float(row["metrics"][metric])
    differences = np.array(
        [values[reference] - values[comparator] for values in paired.values() if len(values) == 2]
    )
    if len(differences) < 3:
        raise ValueError("pilot needs at least three paired worlds")
    pilot_sd = float(np.std(differences, ddof=1))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(bootstrap_draws, len(differences)))
    sd_draws = np.std(differences[indices], axis=1, ddof=1)
    sd_upper = float(np.percentile(sd_draws, 95))
    critical = norm.ppf(1 - alpha / 2) + norm.ppf(target_power)
    raw_worlds = int(np.ceil((critical * sd_upper / practical_delta) ** 2))
    recommendation = min(maximum_worlds, max(minimum_worlds, raw_worlds))
    return {
        "status": "validation-pilot-only-not-primary-evidence",
        "metric": metric,
        "reference": reference,
        "selected_comparator": comparator,
        "pilot_paired_worlds": len(differences),
        "mean_paired_difference": float(np.mean(differences)),
        "pilot_sd": pilot_sd,
        "bootstrap_sd_upper95": sd_upper,
        "practical_delta": practical_delta,
        "alpha": alpha,
        "target_power": target_power,
        "raw_required_worlds": raw_worlds,
        "recommended_worlds": recommendation,
        "underpowered_at_cap": raw_worlds > maximum_worlds,
        "minimum_worlds": minimum_worlds,
        "maximum_worlds": maximum_worlds,
    }


def hodges_lehmann(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    walsh = [(values[i] + values[j]) / 2 for i in range(len(values)) for j in range(i, len(values))]
    return float(np.median(walsh))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    comparisons = np.sign(a[:, None] - b[None, :])
    return float(np.mean(comparisons))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def analyze_results(
    results: Iterable[dict[str, Any]],
    *,
    reference: str = "airproof",
    metrics: tuple[str, ...] = ("worst_group_rmse", "rmse", "aoi_p95", "coverage_gap"),
) -> dict[str, Any]:
    rows = list(results)
    by_key = {(row["config_hash"], int(row["seed"]), row["method"]): row for row in rows}
    configs = sorted({row["config_hash"] for row in rows})
    methods = sorted({row["method"] for row in rows if row["method"] != reference})
    report: dict[str, Any] = {"reference": reference, "comparisons": {}}
    raw_p: dict[str, float] = {}
    for method in methods:
        for metric in metrics:
            pairs: list[tuple[float, float]] = []
            for config_hash in configs:
                seeds = sorted({int(row["seed"]) for row in rows if row["config_hash"] == config_hash})
                for seed in seeds:
                    ref_row = by_key.get((config_hash, seed, reference))
                    other_row = by_key.get((config_hash, seed, method))
                    if ref_row is None or other_row is None:
                        continue
                    ref_value = ref_row["metrics"].get(metric)
                    other_value = other_row["metrics"].get(metric)
                    if ref_value is not None and other_value is not None:
                        pairs.append((float(ref_value), float(other_value)))
            if len(pairs) < 2:
                continue
            # A seed identifies one independent world.  Multiple registered scenario
            # cells evaluated on that world are repeated measures, not extra worlds.
            clustered: dict[int, list[tuple[float, float]]] = {}
            for config_hash in configs:
                seeds = sorted({int(row["seed"]) for row in rows if row["config_hash"] == config_hash})
                for seed in seeds:
                    ref_row = by_key.get((config_hash, seed, reference))
                    other_row = by_key.get((config_hash, seed, method))
                    if ref_row is None or other_row is None:
                        continue
                    ref_value = ref_row["metrics"].get(metric)
                    other_value = other_row["metrics"].get(metric)
                    if ref_value is not None and other_value is not None:
                        clustered.setdefault(seed, []).append((float(ref_value), float(other_value)))
            reference_values = np.array(
                [np.mean([pair[0] for pair in values]) for values in clustered.values()]
            )
            comparison_values = np.array(
                [np.mean([pair[1] for pair in values]) for values in clustered.values()]
            )
            differences = reference_values - comparison_values
            if len(differences) < 2:
                continue
            try:
                p_value = (
                    1.0
                    if np.allclose(differences, 0.0)
                    else float(wilcoxon(differences, alternative="two-sided").pvalue)
                )
            except ValueError:
                p_value = 1.0
            if not np.isfinite(p_value):
                p_value = 1.0
            key = f"{method}:{metric}"
            raw_p[key] = p_value
            report["comparisons"][key] = {
                "n_worlds": len(differences),
                "n_scenario_pairs": len(pairs),
                "mean_difference_airproof_minus_baseline": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "hodges_lehmann": hodges_lehmann(differences),
                "paired_bootstrap_ci95": paired_bootstrap_ci(differences),
                "wilcoxon_p": p_value,
                "paired_permutation_p": paired_permutation_pvalue(differences),
                "cliffs_delta_unpaired_supplement": cliffs_delta(reference_values, comparison_values),
            }
    adjusted = holm_adjust(raw_p)
    for key, value in adjusted.items():
        report["comparisons"][key]["holm_adjusted_p"] = value
    return report


def analyze_factorial_results(
    results: Iterable[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = ("worst_group_rmse", "rmse", "restricted_mean_aoi"),
) -> dict[str, Any]:
    """Estimate balanced 2^3 contrasts inside each independent world."""
    rows = [row for row in results if str(row["method"]).startswith("factorial_")]
    regimes = sorted({str(row.get("scenario_id", "unspecified")) for row in rows})
    effects = {
        "fairness": (0,),
        "relay": (1,),
        "huber": (2,),
        "fairness_x_relay": (0, 1),
        "fairness_x_huber": (0, 2),
        "relay_x_huber": (1, 2),
        "fairness_x_relay_x_huber": (0, 1, 2),
    }
    report: dict[str, Any] = {"design": "paired-balanced-2x2x2", "contrasts": {}}
    raw_p: dict[str, float] = {}
    for regime in regimes:
        regime_rows = [
            row for row in rows if str(row.get("scenario_id", "unspecified")) == regime
        ]
        seeds = sorted({int(row["seed"]) for row in regime_rows})
        lookup = {(int(row["seed"]), row["method"]): row for row in regime_rows}
        for metric in metrics:
            for effect_name, indices in effects.items():
                world_contrasts: list[float] = []
                for seed in seeds:
                    weighted: list[float] = []
                    complete = True
                    for bits in itertools.product((0, 1), repeat=3):
                        method = f"factorial_{bits[0]}{bits[1]}{bits[2]}"
                        row = lookup.get((seed, method))
                        if row is None or row["metrics"].get(metric) is None:
                            complete = False
                            break
                        sign = float(
                            np.prod([1 if bits[index] else -1 for index in indices])
                        )
                        weighted.append(sign * float(row["metrics"][metric]))
                    if complete:
                        world_contrasts.append(float(sum(weighted) / 4.0))
                if len(world_contrasts) < 2:
                    continue
                values = np.asarray(world_contrasts)
                key = f"{regime}:{metric}:{effect_name}"
                p_value = paired_permutation_pvalue(values)
                raw_p[key] = p_value
                report["contrasts"][key] = {
                    "n_worlds": len(values),
                    "mean_contrast": float(np.mean(values)),
                    "median_contrast": float(np.median(values)),
                    "paired_bootstrap_ci95": paired_bootstrap_ci(values),
                    "paired_permutation_p": p_value,
                }
    adjusted = holm_adjust(raw_p)
    for key, value in adjusted.items():
        report["contrasts"][key]["holm_adjusted_p"] = value
    return report


def analyze_attack_induced_effects(
    results: Iterable[dict[str, Any]],
    *,
    reference: str = "airproof",
    comparator: str = "squared_loss",
) -> dict[str, Any]:
    """Paired attack-minus-clean differences, compared between estimators."""
    rows = list(results)
    lookup = {
        (str(row.get("scenario_id", "")), int(row["seed"]), str(row["method"])): row
        for row in rows
    }
    scenarios = sorted({key[0] for key in lookup if key[0] != "clean"})
    report: dict[str, Any] = {
        "design": "paired-attack-minus-clean-difference-in-differences",
        "comparisons": {},
    }
    raw_p: dict[str, float] = {}
    for scenario in scenarios:
        seeds = sorted({key[1] for key in lookup if key[0] == scenario})
        for metric, use_absolute in (
            ("bias", True),
            ("worst_group_rmse", False),
            ("rmse", False),
        ):
            differences: list[float] = []
            for seed in seeds:
                required = [
                    lookup.get((scenario, seed, reference)),
                    lookup.get(("clean", seed, reference)),
                    lookup.get((scenario, seed, comparator)),
                    lookup.get(("clean", seed, comparator)),
                ]
                if any(row is None for row in required):
                    continue
                ref_attack, ref_clean, cmp_attack, cmp_clean = required
                ref_attack_value = float(ref_attack["metrics"][metric])
                ref_clean_value = float(ref_clean["metrics"][metric])
                cmp_attack_value = float(cmp_attack["metrics"][metric])
                cmp_clean_value = float(cmp_clean["metrics"][metric])
                if use_absolute:
                    ref_attack_value, ref_clean_value = abs(ref_attack_value), abs(ref_clean_value)
                    cmp_attack_value, cmp_clean_value = abs(cmp_attack_value), abs(cmp_clean_value)
                differences.append(
                    (ref_attack_value - ref_clean_value)
                    - (cmp_attack_value - cmp_clean_value)
                )
            if len(differences) < 2:
                continue
            values = np.asarray(differences)
            key = f"{scenario}:{metric}"
            p_value = paired_permutation_pvalue(values)
            raw_p[key] = p_value
            report["comparisons"][key] = {
                "n_worlds": len(values),
                "mean_difference_in_induced_effect": float(np.mean(values)),
                "paired_bootstrap_ci95": paired_bootstrap_ci(values),
                "paired_permutation_p": p_value,
            }
    adjusted = holm_adjust(raw_p)
    for key, value in adjusted.items():
        report["comparisons"][key]["holm_adjusted_p"] = value
    return report


def write_analysis(report: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# AirProof measured paired analysis", "", f"Reference: `{report['reference']}`", ""]
    lines.append("| Comparison | n | Mean paired delta | 95% bootstrap CI | Holm p |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, item in sorted(report["comparisons"].items()):
        low, high = item["paired_bootstrap_ci95"]
        lines.append(
            f"| {key} | {item['n_worlds']} | {item['mean_difference_airproof_minus_baseline']:.6g} "
            f"| [{low:.6g}, {high:.6g}] | {item['holm_adjusted_p']:.6g} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

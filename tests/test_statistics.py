import numpy as np

from airproof.statistics import (
    analyze_attack_induced_effects,
    analyze_factorial_results,
    analyze_results,
    holm_adjust,
    paired_bootstrap_ci,
    paired_permutation_pvalue,
    pilot_power_analysis,
)


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_paired_bootstrap_is_deterministic():
    values = np.array([-2.0, -1.0, -3.0, -2.5])
    assert paired_bootstrap_ci(values) == paired_bootstrap_ci(values)


def test_paired_permutation_and_validation_power_are_reproducible():
    differences = np.array([-1.0, -1.5, -0.8, -1.2])
    assert paired_permutation_pvalue(differences, seed=4, draws=1000) < 0.2
    rows = []
    for seed, difference in enumerate(differences):
        for method, value in (("airproof", 5.0 + difference), ("baseline", 5.0)):
            rows.append(
                {
                    "config_hash": "c",
                    "seed": seed,
                    "method": method,
                    "metrics": {"worst_group_rmse": value},
                }
            )
    report = pilot_power_analysis(
        rows,
        metric="worst_group_rmse",
        comparator="baseline",
        practical_delta=0.5,
        bootstrap_draws=1000,
        seed=4,
    )
    assert report["pilot_paired_worlds"] == 4
    assert 30 <= report["recommended_worlds"] <= 100


def test_analysis_clusters_repeated_scenarios_by_independent_world():
    rows = []
    for config_hash in ("nominal", "severe"):
        for seed in range(3):
            for method, value in (("airproof", 1.0 + seed), ("baseline", 2.0 + seed)):
                rows.append(
                    {
                        "config_hash": config_hash,
                        "seed": seed,
                        "method": method,
                        "metrics": {"rmse": value},
                    }
                )
    report = analyze_results(rows, metrics=("rmse",))
    comparison = report["comparisons"]["baseline:rmse"]
    assert comparison["n_worlds"] == 3
    assert comparison["n_scenario_pairs"] == 6


def test_factorial_and_attack_analyses_keep_world_as_unit():
    factorial_rows = []
    for seed in range(4):
        for fair in (0, 1):
            for relay in (0, 1):
                for huber in (0, 1):
                    factorial_rows.append(
                        {
                            "scenario_id": "nominal",
                            "seed": seed,
                            "method": f"factorial_{fair}{relay}{huber}",
                            "metrics": {
                                "rmse": 10.0 - fair - 2 * relay - 3 * huber,
                                "worst_group_rmse": 10.0 - fair,
                                "restricted_mean_aoi": 5.0 - relay,
                            },
                        }
                    )
    factorial = analyze_factorial_results(factorial_rows)
    assert factorial["contrasts"]["nominal:rmse:fairness"]["n_worlds"] == 4
    assert factorial["contrasts"]["nominal:rmse:fairness"]["mean_contrast"] == -1.0

    attack_rows = []
    for seed in range(4):
        for scenario, attack_addition in (("clean", 0.0), ("drift20", 2.0)):
            for method, protection in (("airproof", 0.5), ("squared_loss", 0.0)):
                attack_rows.append(
                    {
                        "scenario_id": scenario,
                        "seed": seed,
                        "method": method,
                        "metrics": {
                            "bias": attack_addition - protection * attack_addition,
                            "rmse": 3.0 + attack_addition - protection * attack_addition,
                            "worst_group_rmse": 4.0
                            + attack_addition
                            - protection * attack_addition,
                        },
                    }
                )
    attack = analyze_attack_induced_effects(attack_rows)
    comparison = attack["comparisons"]["drift20:rmse"]
    assert comparison["n_worlds"] == 4
    assert comparison["mean_difference_in_induced_effect"] == -1.0

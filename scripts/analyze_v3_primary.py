from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SCENARIOS = ("anchor", "outage", "severe_clean", "severe_drift", "severe_hotspot")
PRIMARY = "airproof_predictive_residual"
CONTROL = "squared_loss"
FAIRNESS_CONTROL = "airproof_predictive_no_fairness"
EXPECTED_SOURCE = "a11734addcabd75b6c1b178f731bc355ed6debafa2a60b8540250413dbd3931e"
EXPECTED_REGISTRY = "ccd4b408c5e3770cfde3f2361819fe27e708f424433faf2cd80ec2506227bd71"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, hashlib.sha256(raw).hexdigest()


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


def _ratio(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    seeds = sorted(set(left) & set(right))
    values = np.asarray(
        [left[seed]["metrics"][metric] / right[seed]["metrics"][metric] for seed in seeds]
    )
    return _interval(values, np.mean, rng=rng)


def _fractional_reduction(
    improved: dict[int, dict[str, Any]],
    control: dict[int, dict[str, Any]],
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    seeds = sorted(set(improved) & set(control))

    def value(row: dict[str, Any]) -> float:
        result = float(row["metrics"][metric])
        return abs(result) if metric == "bias" else result

    values = np.asarray(
        [[value(improved[seed]), value(control[seed])] for seed in seeds]
    )

    def reduction(sample: np.ndarray) -> float:
        denominator = float(np.mean(sample[:, 1]))
        return float(1.0 - np.mean(sample[:, 0]) / denominator)

    return _interval(values, reduction, rng=rng)


def _attack_attenuation(
    clean: dict[str, dict[int, dict[str, Any]]],
    attack: dict[str, dict[int, dict[str, Any]]],
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    seeds = sorted(
        set(clean[PRIMARY])
        & set(clean[CONTROL])
        & set(attack[PRIMARY])
        & set(attack[CONTROL])
    )

    def value(row: dict[str, Any]) -> float:
        result = float(row["metrics"][metric])
        return abs(result) if metric == "bias" else result

    control_growth = np.asarray(
        [value(attack[CONTROL][seed]) - value(clean[CONTROL][seed]) for seed in seeds]
    )
    primary_growth = np.asarray(
        [value(attack[PRIMARY][seed]) - value(clean[PRIMARY][seed]) for seed in seeds]
    )
    paired = np.column_stack([primary_growth - control_growth, control_growth])

    def attenuation(sample: np.ndarray) -> float:
        denominator = float(np.mean(sample[:, 1]))
        return float(-np.mean(sample[:, 0]) / denominator) if denominator > 0 else np.nan

    return {
        "difference_in_differences": _interval(paired[:, 0], np.mean, rng=rng),
        "attenuation_fraction": _interval(paired, attenuation, rng=rng),
        "control_growth_mean": float(np.mean(control_growth)),
        "airproof_growth_mean": float(np.mean(primary_growth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for scenario in SCENARIOS:
        parser.add_argument(f"--{scenario.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    paths = {
        scenario: getattr(args, scenario)
        for scenario in SCENARIOS
    }
    rows_by_scenario: dict[str, list[dict[str, Any]]] = {}
    input_hashes = {}
    for scenario, path in paths.items():
        rows_by_scenario[scenario], input_hashes[str(path).replace("\\", "/")] = (
            _read_jsonl(path)
        )

    index: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    expected_seeds = set(range(6000, 6030))
    for scenario, rows in rows_by_scenario.items():
        methods: dict[str, dict[int, dict[str, Any]]] = {}
        for row in rows:
            methods.setdefault(str(row["method"]), {})[int(row["seed"])] = row
        expected_methods = {PRIMARY, CONTROL}
        if scenario == "severe_clean":
            expected_methods.add(FAIRNESS_CONTROL)
        if set(methods) != expected_methods:
            raise ValueError(f"{scenario} methods do not match the frozen design")
        if any(set(seed_map) != expected_seeds for seed_map in methods.values()):
            raise ValueError(f"{scenario} is missing a frozen primary seed")
        index[scenario] = methods

    all_rows = [row for rows in rows_by_scenario.values() for row in rows]
    source_hashes = sorted({row["source_tree_sha256"] for row in all_rows})
    registry_hashes = sorted({row["registry_sha256"] for row in all_rows})
    if len(source_hashes) != 1 or len(registry_hashes) != 1:
        raise ValueError("primary artifacts span multiple executable or registry versions")
    if source_hashes != [EXPECTED_SOURCE] or registry_hashes != [EXPECTED_REGISTRY]:
        raise ValueError("primary artifacts do not match the frozen manifest digests")
    run_ids = [str(row["run_id"]) for row in all_rows]
    if len(all_rows) != 330 or len(set(run_ids)) != 330:
        raise ValueError("primary row count or run-id uniqueness invariant failed")

    means = {
        scenario: {
            method: {
                metric: float(np.mean([row["metrics"][metric] for row in seed_map.values()]))
                for metric in (
                    "rmse",
                    "mae",
                    "worst_group_rmse",
                    "bias",
                    "coverage_gap",
                    "restricted_mean_aoi",
                    "delivery_ratio",
                    "release_rmse",
                    "solver_failure_rate",
                    "epoch_update_p95_seconds",
                    "peak_memory_mb",
                )
            }
            for method, seed_map in methods.items()
        }
        for scenario, methods in index.items()
    }
    clean = {
        scenario: {
            metric: _ratio(index[scenario][PRIMARY], index[scenario][CONTROL], metric, rng)
            for metric in ("rmse", "worst_group_rmse")
        }
        for scenario in ("anchor", "outage", "severe_clean")
    }
    fairness = {
        "coverage_gap_reduction": _fractional_reduction(
            index["severe_clean"][PRIMARY],
            index["severe_clean"][FAIRNESS_CONTROL],
            "coverage_gap",
            rng,
        ),
        "rmse_ratio": _ratio(
            index["severe_clean"][PRIMARY],
            index["severe_clean"][FAIRNESS_CONTROL],
            "rmse",
            rng,
        ),
        "worst_group_rmse_reduction": _fractional_reduction(
            index["severe_clean"][PRIMARY],
            index["severe_clean"][FAIRNESS_CONTROL],
            "worst_group_rmse",
            rng,
        ),
    }
    attacks = {
        scenario: {
            metric: _attack_attenuation(
                index["severe_clean"], index[scenario], metric, rng
            )
            for metric in ("rmse", "worst_group_rmse", "bias")
        }
        for scenario in ("severe_drift", "severe_hotspot")
    }
    attacked_ratios = {
        scenario: {
            metric: _ratio(index[scenario][PRIMARY], index[scenario][CONTROL], metric, rng)
            for metric in ("rmse", "worst_group_rmse")
        }
        for scenario in ("severe_drift", "severe_hotspot")
    }
    drift_bias_reduction = _fractional_reduction(
        index["severe_drift"][PRIMARY],
        index["severe_drift"][CONTROL],
        "bias",
        rng,
    )
    primary_rows = [row for row in all_rows if row["method"] == PRIMARY]
    feasible_violations = sum(
        int(row["metrics"]["constraint_violation_when_feasible_epochs"])
        for row in primary_rows
    )
    solver_failure_max = max(
        float(row["metrics"]["solver_failure_rate"]) for row in primary_rows
    )
    p95_update_max = max(
        float(row["metrics"]["epoch_update_p95_seconds"]) for row in primary_rows
    )
    privacy_cap_violations = sum(
        float(row["metrics"]["max_composed_user_epsilon"])
        > float(row["scenario"]["epsilon_user_max"])
        for row in primary_rows
    )
    total_runtime_seconds = float(
        sum(float(row["metrics"]["runtime_seconds"]) for row in all_rows)
    )
    peak_process_memory_mb = max(
        float(row["metrics"]["peak_memory_mb"]) for row in all_rows
    )
    gates = {
        "all_clean_rmse_ratios_lte_1_05": all(
            clean[scenario]["rmse"]["estimate"] <= 1.05 for scenario in clean
        ),
        "hotspot_rmse_growth_attenuation_gte_0_20": attacks["severe_hotspot"][
            "rmse"
        ]["attenuation_fraction"]["estimate"]
        >= 0.20,
        "drift_attacked_rmse_ratio_lte_1_05": attacked_ratios["severe_drift"][
            "rmse"
        ]["estimate"]
        <= 1.05,
        "drift_attacked_bias_reduction_gt_0": drift_bias_reduction["estimate"] > 0.0,
        "severe_fairness_coverage_gap_reduction_gte_0_20": fairness[
            "coverage_gap_reduction"
        ]["estimate"]
        >= 0.20,
        "severe_fairness_rmse_price_lte_0_05": fairness["rmse_ratio"]["estimate"]
        - 1.0
        <= 0.05,
        "feasible_epoch_floor_violations_eq_0": feasible_violations == 0,
        "solver_failure_rate_lt_0_01": solver_failure_max < 0.01,
        "p95_epoch_update_seconds_lt_3600": p95_update_max < 3600,
        "privacy_cap_violations_eq_0": privacy_cap_violations == 0,
    }
    gates["all_precommitted_gates"] = all(gates.values())
    report: dict[str, Any] = {
        "status": "completed-frozen-v3-full-horizon-primary",
        "design": {
            "seeds": sorted(expected_seeds),
            "worlds_per_cell": 30,
            "agents": 1000,
            "epochs": 672,
            "rows": len(all_rows),
            "pairing": "common-random-number-by-seed",
            "bootstrap_replicates": 10_000,
        },
        "input_sha256": input_hashes,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_tree_sha256": source_hashes[0],
        "registry_sha256": registry_hashes[0],
        "means": means,
        "clean_noninferiority": clean,
        "fairness": fairness,
        "attacks": attacks,
        "attacked_ratios": attacked_ratios,
        "drift_attacked_bias_reduction": drift_bias_reduction,
        "operational_extrema": {
            "feasible_epoch_floor_violations": feasible_violations,
            "solver_failure_rate_max": solver_failure_max,
            "p95_epoch_update_seconds_max": p95_update_max,
            "privacy_cap_violations": privacy_cap_violations,
            "total_method_runtime_seconds": total_runtime_seconds,
            "peak_process_memory_mb": peak_process_memory_mb,
            "unique_run_ids": len(set(run_ids)),
        },
        "decision_gates": gates,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(gates, sort_keys=True))


if __name__ == "__main__":
    main()

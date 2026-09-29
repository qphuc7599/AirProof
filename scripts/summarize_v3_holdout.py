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
    replicates: int,
) -> dict[str, Any]:
    observed = float(statistic(values))
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = np.asarray([statistic(values[index]) for index in indices], dtype=float)
    finite = samples[np.isfinite(samples)]
    return {
        "estimate": observed,
        "bootstrap_95_ci": [
            float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)),
        ],
        "bootstrap_replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="airproof_predictive_residual")
    parser.add_argument("--control", default="airproof")
    parser.add_argument("--drift-name", default="adversarial_drift")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    lookup = {
        (int(row["seed"]), str(row["scenario"]["attack_kind"]), str(row["method"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    rng = np.random.default_rng(args.seed)

    def metric(seed: int, regime: str, method: str, name: str) -> float:
        value = float(lookup[(seed, regime, method)]["metrics"][name])
        return abs(value) if name == "bias" else value

    report: dict[str, Any] = {
        "artifact_status": "validation-only-locked-holdout",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
        "seeds": seeds,
        "candidate": args.candidate,
        "control": args.control,
        "source_tree_sha256": sorted({row["source_tree_sha256"] for row in rows}),
        "registry_sha256": sorted({row["registry_sha256"] for row in rows}),
        "config_hashes": sorted({row["config_hash"] for row in rows}),
        "clean": {},
        "attacks": {},
    }
    for method in (args.candidate, args.control):
        relative = np.asarray(
            [
                metric(seed, "clean", method, "rmse")
                / metric(seed, "clean", "squared_loss", "rmse")
                - 1.0
                for seed in seeds
            ]
        )
        report["clean"][method] = {
            "relative_rmse_penalty": _interval(
                relative, np.mean, rng=rng, replicates=args.bootstrap
            ),
            "maximum_world_penalty": float(np.max(relative)),
        }

    for regime in (args.drift_name, "hotspot_suppression"):
        report["attacks"][regime] = {}
        for method in (args.candidate, args.control):
            method_report: dict[str, Any] = {}
            for name in ("rmse", "worst_group_rmse", "bias"):
                squared_growth = np.asarray(
                    [
                        metric(seed, regime, "squared_loss", name)
                        - metric(seed, "clean", "squared_loss", name)
                        for seed in seeds
                    ]
                )
                method_growth = np.asarray(
                    [
                        metric(seed, regime, method, name)
                        - metric(seed, "clean", method, name)
                        for seed in seeds
                    ]
                )
                paired = np.column_stack([method_growth - squared_growth, squared_growth])

                def attenuation(sample: np.ndarray) -> float:
                    denominator = float(np.mean(sample[:, 1]))
                    return float(-np.mean(sample[:, 0]) / denominator) if denominator > 0 else float("nan")

                method_report[name] = {
                    "difference_in_differences": _interval(
                        paired[:, 0], np.mean, rng=rng, replicates=args.bootstrap
                    ),
                    "attenuation_fraction": _interval(
                        paired, attenuation, rng=rng, replicates=args.bootstrap
                    ),
                    "squared_growth_mean": float(np.mean(squared_growth)),
                    "method_growth_mean": float(np.mean(method_growth)),
                }
            report["attacks"][regime][method] = method_report

    clean_estimate = report["clean"][args.candidate]["relative_rmse_penalty"]["estimate"]
    drift_estimate = report["attacks"][args.drift_name][args.candidate]["rmse"][
        "attenuation_fraction"
    ]["estimate"]
    hotspot_did = report["attacks"]["hotspot_suppression"][args.candidate]["rmse"][
        "difference_in_differences"
    ]["estimate"]
    report["mechanism_gate"] = {
        "clean_mean_relative_penalty_lte_0_05": clean_estimate <= 0.05,
        "drift_rmse_attenuation_gte_0_20": drift_estimate >= 0.20,
        "hotspot_rmse_noninferior_to_squared": hotspot_did <= 0.0,
    }
    report["mechanism_gate"]["all"] = all(report["mechanism_gate"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["mechanism_gate"], sort_keys=True))


if __name__ == "__main__":
    main()

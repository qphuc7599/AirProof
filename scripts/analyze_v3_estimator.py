from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _index(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["seed"]), str(row["method"])): row for row in rows}


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _metric(row: dict[str, Any], name: str) -> float:
    value = float(row["metrics"][name])
    return abs(value) if name == "bias" else value


def summarize(
    clean_rows: list[dict[str, Any]],
    attacked: dict[str, list[dict[str, Any]]],
    candidate: str,
) -> dict[str, Any]:
    clean = _index(clean_rows)
    methods = sorted({method for _, method in clean})
    seeds = sorted({seed for seed, _ in clean})
    if "squared_loss" not in methods or candidate not in methods:
        raise ValueError("clean file must contain squared_loss and the candidate")

    report: dict[str, Any] = {
        "candidate": candidate,
        "seeds": seeds,
        "clean_means": {
            method: {
                metric: _mean([_metric(clean[(seed, method)], metric) for seed in seeds])
                for metric in ("rmse", "worst_group_rmse", "bias")
            }
            for method in methods
        },
    }
    clean_penalties = [
        100.0
        * (
            _metric(clean[(seed, candidate)], "rmse")
            / _metric(clean[(seed, "squared_loss")], "rmse")
            - 1.0
        )
        for seed in seeds
    ]
    report["clean_penalty_percent"] = {
        "mean": _mean(clean_penalties),
        "max": float(np.max(clean_penalties)),
    }

    report["attacks"] = {}
    for attack_name, rows in attacked.items():
        attack = _index(rows)
        attack_seeds = sorted(set(seeds) & {seed for seed, _ in attack})
        details: dict[str, Any] = {
            "means": {
                method: {
                    metric: _mean([_metric(attack[(seed, method)], metric) for seed in attack_seeds])
                    for metric in ("rmse", "worst_group_rmse", "bias")
                }
                for method in ("squared_loss", candidate)
            }
        }
        for metric in ("rmse", "worst_group_rmse", "bias"):
            squared_growth = [
                _metric(attack[(seed, "squared_loss")], metric)
                - _metric(clean[(seed, "squared_loss")], metric)
                for seed in attack_seeds
            ]
            did = [
                (
                    _metric(attack[(seed, candidate)], metric)
                    - _metric(clean[(seed, candidate)], metric)
                )
                - squared_growth[index]
                for index, seed in enumerate(attack_seeds)
            ]
            mean_growth = _mean(squared_growth)
            mean_did = _mean(did)
            details[f"{metric}_attack_did"] = mean_did
            details[f"{metric}_squared_growth"] = mean_growth
            details[f"{metric}_attenuation_fraction"] = (
                -mean_did / mean_growth if mean_growth > 0 else None
            )
        candidate_rows = [attack[(seed, candidate)] for seed in attack_seeds]
        details["correction"] = {
            "tail_fraction_mean": _mean(
                [float(row["metrics"].get("correction_tail_fraction_mean", 0.0)) for row in candidate_rows]
            ),
            "absolute_mean": _mean(
                [float(row["metrics"].get("correction_abs_mean", 0.0)) for row in candidate_rows]
            ),
            "absolute_max": float(
                np.max([float(row["metrics"].get("correction_abs_max", 0.0)) for row in candidate_rows])
            ),
        }
        report["attacks"][attack_name] = details
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--drift", type=Path)
    parser.add_argument("--hotspot", type=Path)
    parser.add_argument("--candidate", default="airproof_residual")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    attacks = {}
    if args.drift:
        attacks["gradual_drift"] = _read(args.drift)
    if args.hotspot:
        attacks["hotspot_suppression"] = _read(args.hotspot)
    report = summarize(_read(args.clean), attacks, args.candidate)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

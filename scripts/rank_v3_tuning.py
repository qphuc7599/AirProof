from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("rmse", "worst_group_rmse", "bias")


def _metric(row: dict[str, Any], name: str) -> float:
    value = float(row["metrics"][name])
    return abs(value) if name == "bias" else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="airproof_residual")
    parser.add_argument("--drift-name", default="gradual_drift")
    parser.add_argument("--seed-min", type=int)
    parser.add_argument("--seed-max", type=int)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if args.seed_min is not None:
        rows = [row for row in rows if int(row["seed"]) >= args.seed_min]
    if args.seed_max is not None:
        rows = [row for row in rows if int(row["seed"]) <= args.seed_max]

    def setting(row: dict[str, Any]) -> tuple[float, ...]:
        scenario = row["scenario"]
        return (
            float(scenario.get("reference_stations", 0)),
            float(scenario["correction_delta"]),
            float(scenario["lambda_correction"]),
            float(scenario.get("correction_clean_gain", 1.0)),
            float(scenario.get("correction_attack_gain", 1.0)),
            float(scenario.get("correction_gate_start", 0.0)),
            float(scenario.get("correction_gate_full", 1.0)),
            float(scenario.get("correction_gate_ewma", 1.0)),
        )

    index = {
        (
            *setting(row),
            str(row["scenario"]["attack_kind"]),
            int(row["seed"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    setting_width = 8
    settings = sorted({key[:setting_width] for key in index})
    seeds = sorted({key[setting_width + 1] for key in index})
    candidates: list[dict[str, Any]] = []
    for current_setting in settings:
        (
            reference_stations,
            delta,
            regularization,
            clean_gain,
            attack_gain,
            gate_start,
            gate_full,
            gate_ewma,
        ) = current_setting
        clean_penalty = []
        for seed in seeds:
            squared = index[(*current_setting, "clean", seed, "squared_loss")]
            candidate = index[(*current_setting, "clean", seed, args.candidate)]
            clean_penalty.append(
                100.0 * (_metric(candidate, "rmse") / _metric(squared, "rmse") - 1.0)
            )
        item: dict[str, Any] = {
            "correction_delta": delta,
            "reference_stations": int(reference_stations),
            "lambda_correction": regularization,
            "correction_clean_gain": clean_gain,
            "correction_attack_gain": attack_gain,
            "correction_gate_start": gate_start,
            "correction_gate_full": gate_full,
            "correction_gate_ewma": gate_ewma,
            "clean_penalty_mean_percent": float(np.mean(clean_penalty)),
            "clean_penalty_max_percent": float(np.max(clean_penalty)),
        }
        for regime in ("clean", args.drift_name, "hotspot_suppression"):
            candidate_rows = [
                index[(*current_setting, regime, seed, args.candidate)] for seed in seeds
            ]
            item[f"{regime}_activation_mean"] = float(
                np.mean(
                    [
                        float(row["metrics"].get("correction_activation_mean", 0.0))
                        for row in candidate_rows
                    ]
                )
            )
            item[f"{regime}_closed_tail_ewma_mean"] = float(
                np.mean(
                    [
                        float(row["metrics"].get("closed_tail_ewma_mean", 0.0))
                        for row in candidate_rows
                    ]
                )
            )
        for attack in (args.drift_name, "hotspot_suppression"):
            for metric in METRICS:
                squared_growth = []
                candidate_growth = []
                for seed in seeds:
                    squared_clean = index[(*current_setting, "clean", seed, "squared_loss")]
                    squared_attack = index[(*current_setting, attack, seed, "squared_loss")]
                    candidate_clean = index[(*current_setting, "clean", seed, args.candidate)]
                    candidate_attack = index[(*current_setting, attack, seed, args.candidate)]
                    squared_growth.append(
                        _metric(squared_attack, metric) - _metric(squared_clean, metric)
                    )
                    candidate_growth.append(
                        _metric(candidate_attack, metric) - _metric(candidate_clean, metric)
                    )
                squared_mean = float(np.mean(squared_growth))
                did = float(np.mean(np.asarray(candidate_growth) - np.asarray(squared_growth)))
                item[f"{attack}_{metric}_did"] = did
                item[f"{attack}_{metric}_attenuation_fraction"] = (
                    -did / squared_mean if squared_mean > 0 else None
                )
        item["clean_gate"] = item["clean_penalty_mean_percent"] <= 5.0
        drift_attenuation = item[f"{args.drift_name}_rmse_attenuation_fraction"]
        item["drift_gate"] = bool(
            drift_attenuation is not None and drift_attenuation >= 0.20
        )
        item["hotspot_noninferior"] = item["hotspot_suppression_rmse_did"] <= 0.0
        item["joint_gate"] = bool(
            item["clean_gate"] and item["drift_gate"] and item["hotspot_noninferior"]
        )
        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            not item["joint_gate"],
            not item["clean_gate"],
            -(
                item[f"{args.drift_name}_rmse_attenuation_fraction"]
                if item[f"{args.drift_name}_rmse_attenuation_fraction"] is not None
                else float("-inf")
            ),
            item["clean_penalty_mean_percent"],
        )
    )
    report = {
        "artifact_status": "validation-only-hyperparameter-ranking",
        "candidate": args.candidate,
        "rows": len(rows),
        "seeds": seeds,
        "joint_gate_count": sum(item["joint_gate"] for item in candidates),
        "ranking": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

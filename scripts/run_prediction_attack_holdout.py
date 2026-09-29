from __future__ import annotations

import argparse
import json
from pathlib import Path

from airproof.config import load_config, with_overrides
from airproof.prediction import PREDICTORS, run_prediction_benchmark

SCENARIOS = {
    "clean": {"attack.kind": "clean", "attack.fraction": 0.0},
    "adversarial_drift20": {
        "attack.kind": "adversarial_drift",
        "attack.fraction": 0.2,
    },
    "hotspot_suppression20": {
        "attack.kind": "hotspot_suppression",
        "attack.fraction": 0.2,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="5500:5512")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--predictors", default=",".join(PREDICTORS))
    parser.add_argument("--status", default="locked-independent-prediction-attack-holdout")
    parser.add_argument("--selection-seeds", default="5400:5406")
    parser.add_argument("--correction-delta", type=float)
    parser.add_argument("--adaptive-delta", choices=("keep", "true", "false"), default="keep")
    parser.add_argument("--predictive-clean-delta", type=float)
    parser.add_argument("--predictive-robust-delta", type=float)
    parser.add_argument("--gate-start", type=float)
    parser.add_argument("--gate-full", type=float)
    parser.add_argument("--tail-z", type=float)
    parser.add_argument("--activation-gain", type=float)
    parser.add_argument("--excess-gate-start", type=float)
    parser.add_argument("--excess-gate-full", type=float)
    args = parser.parse_args()
    start, stop = [int(item) for item in args.seeds.split(":", 1)]
    seeds = list(range(start, stop))
    predictors = tuple(item for item in args.predictors.split(",") if item)
    selection_start, selection_stop = [int(item) for item in args.selection_seeds.split(":", 1)]
    base = load_config(args.config)
    overrides = {}
    if args.correction_delta is not None:
        overrides["twin.correction_delta"] = args.correction_delta
    if args.adaptive_delta != "keep":
        overrides["twin.predictive_adaptive_delta"] = args.adaptive_delta == "true"
    if args.predictive_clean_delta is not None:
        overrides["twin.predictive_clean_delta"] = args.predictive_clean_delta
    if args.predictive_robust_delta is not None:
        overrides["twin.predictive_robust_delta"] = args.predictive_robust_delta
    if args.gate_start is not None:
        overrides["twin.innovation_gate_start"] = args.gate_start
    if args.gate_full is not None:
        overrides["twin.innovation_gate_full"] = args.gate_full
    if args.tail_z is not None:
        overrides["twin.innovation_gate_tail_z"] = args.tail_z
    if args.activation_gain is not None:
        overrides["twin.predictive_activation_gain"] = args.activation_gain
    if args.excess_gate_start is not None:
        overrides["twin.predictive_excess_gate_start"] = args.excess_gate_start
    if args.excess_gate_full is not None:
        overrides["twin.predictive_excess_gate_full"] = args.excess_gate_full
    if overrides:
        base = with_overrides(base, overrides)
    rows = []
    source_hashes = set()
    config_hashes = {}
    for scenario, overrides in SCENARIOS.items():
        report = run_prediction_benchmark(
            with_overrides(base, overrides),
            seeds,
            predictors,
            workers=args.workers,
        )
        source_hashes.add(report["source_tree_sha256"])
        config_hashes[scenario] = report["config_hash"]
        for row in report["rows"]:
            rows.append({"scenario": scenario, **row})
    output = {
        "status": args.status,
        "selection_seeds": list(range(selection_start, selection_stop)),
        "holdout_seeds": seeds,
        "correction_delta": float(base["twin"]["correction_delta"]),
        "source_tree_sha256": sorted(source_hashes),
        "config_hashes": config_hashes,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

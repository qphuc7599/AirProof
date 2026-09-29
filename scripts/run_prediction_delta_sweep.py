from __future__ import annotations

import argparse
import json
from pathlib import Path

from airproof.config import load_config, with_overrides
from airproof.prediction import run_prediction_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="5400,5401,5402,5403,5404,5405")
    parser.add_argument("--deltas", default="2.9,4,6,8,12")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    deltas = [float(item) for item in args.deltas.split(",") if item]
    base = load_config(args.config)
    rows = []
    source_hashes = set()
    for delta in deltas:
        config = with_overrides(base, {"twin.correction_delta": delta})
        report = run_prediction_benchmark(
            config,
            seeds,
            ("graph_squared", "airproof_predictive_residual"),
            workers=args.workers,
        )
        source_hashes.add(report["source_tree_sha256"])
        for row in report["rows"]:
            rows.append({"correction_delta": delta, **row})
    output = {
        "status": "validation-only-correction-delta-sweep",
        "selection_rule": (
            "Choose the smallest delta whose paired mean clean RMSE ratio to graph squared is "
            "at most 1.05, then subject it to a separate locked attack holdout."
        ),
        "seeds": seeds,
        "deltas": deltas,
        "source_tree_sha256": sorted(source_hashes),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

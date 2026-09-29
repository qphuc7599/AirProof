"""Run a bounded, paired reference-anchor diagnostic, one method per worker job."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airproof.config import load_config, with_overrides
from airproof.experiment import run_method
from airproof.simulator import generate_world


def run_job(job):
    config, seed, method = job
    return run_method(generate_world(config, seed), config, method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7004,7005,7006")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new diagnostic output; existing results are not overwritten")
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/v3/primary_full_horizon.yaml")
    jobs = []
    for kind in ("clean", "adversarial_drift", "hotspot_suppression"):
        config = with_overrides(base, {
            "protocol_version": "airproof-v4-independent-reference-long-diagnostic-1",
            "status": "diagnostic-not-primary",
            "world.participation_skew": 10.0,
            "network.availability": 0.4,
            "network.outage_median_hours": 24,
            "network.outage_p95_hours": 72,
            "attack.fraction": 0.2,
            "attack.kind": kind,
        })
        for seed in map(int, args.seeds.split(",")):
            for method in ("squared_loss", "airproof_predictive_residual",
                           "airproof_reference_anchored"):
                jobs.append((config, seed, method))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.output.open("x", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                print(json.dumps({
                    "complete": completed, "total": len(jobs), "seed": row["seed"],
                    "method": row["method"], "kind": row["scenario"]["attack_kind"],
                    "rmse": row["metrics"]["rmse"],
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }), flush=True)


if __name__ == "__main__":
    main()

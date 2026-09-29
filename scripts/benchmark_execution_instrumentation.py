"""Numerically matched A/B measurement of Python allocation-profiling overhead."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve the previous instrumentation diagnostic")
    from airproof.config import load_config, with_overrides
    from airproof.experiment import run_method, source_tree_digest
    from airproof.simulator import generate_world
    cfg = with_overrides(load_config("configs/v4/public_reference_validation.yaml"), {
        "world.agents": 300, "world.grid_side": 16, "world.steps": 96, "world.burn_in_steps": 24,
        "twin.reference_calibration_epochs": 24, "attack.start_epoch": 48,
        "attack.kind": "adversarial_drift",
    })
    world = generate_world(cfg, 7452)
    ignored = {"runtime_seconds", "peak_memory_mb", "python_peak_alloc_mb", "python_allocation_tracing_enabled",
        "epoch_update_p50_seconds", "epoch_update_p95_seconds", "epoch_update_max_seconds", "intersectional_audit_seconds"}
    results = []
    for enabled in (True, False, False, True):
        result = run_method(world, with_overrides(cfg, {"execution.trace_python_allocations": enabled}), "airproof_v4")
        if results:
            for key, value in result["metrics"].items():
                if key not in ignored and value != results[0]["metrics"][key]:
                    raise ValueError(f"instrumentation changed scientific metric: {key}")
        results.append(result)
        print(json.dumps({"tracing": enabled, "seconds": result["metrics"]["runtime_seconds"]}), flush=True)
    on = sum(row["metrics"]["runtime_seconds"] for row in results if row["metrics"]["python_allocation_tracing_enabled"]) / 2
    off = sum(row["metrics"]["runtime_seconds"] for row in results if not row["metrics"]["python_allocation_tracing_enabled"]) / 2
    report = {"role": "instrumentation-diagnostic-not-final-primary-results", "configuration": cfg,
              "source_hash": source_tree_digest(), "seed": 7452, "ordering": [True, False, False, True],
              "every_scientific_metric_exactly_equal": True, "profiled_mean_seconds": on,
              "unprofiled_mean_seconds": off, "speedup": on / off, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

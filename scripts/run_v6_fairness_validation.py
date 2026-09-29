"""Run the registered severe-clean shared-trace fairness validation."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airproof.config import config_hash, load_config
from airproof.records import canonical_json
from airproof.simulator import generate_world
from airproof.v5_experiment import cell_configuration
from airproof.v6_fairness_validation import allocate_shared_arrivals, collector_view, outcome_strata_39
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_numerical_experiment import evaluate, prepare_public
from airproof.v6_transport import simulate_transport


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "airproof/v6_fairness.py", "airproof/v6_fairness_validation.py",
    "airproof/v6_numerical_experiment.py", "airproof/v6_transport.py",
    "airproof/v6_mobility.py", "airproof/v6_estimator.py", "airproof/scheduler.py",
    "airproof/simulator.py", "configs/v6/fairness_validation.json",
    "configs/v6/protocol.yaml", "scripts/run_v6_fairness_validation.py",
)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+"\n", encoding="utf-8")


def source_inventory():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def transport_summary(result):
    metrics = {key: value for key, value in result.metrics.items() if key != "local_decisions"}
    cfg = metrics
    checks = {
        "directional_capacity": metrics["max_contact_direction_bytes"] <= 1024,
        "control_capacity": metrics["max_control_direction_bytes"] <= 128,
        "raw_buffer": metrics["max_raw_buffer_bytes"] <= 18432,
        "release_buffer": metrics["max_release_buffer_bytes"] <= 6144,
        "copy_tokens": metrics["token_violations"] == 0 and metrics["max_live_copies"] <= 4,
        "raw_denominator": metrics["raw_generated"] == metrics["raw_delivered"] + metrics["raw_undelivered"],
    }
    return {"metrics": cfg, "checks": checks, "all_checks_pass": all(checks.values())}


def run_world(seed, output, source_hash):
    started = time.perf_counter(); output = Path(output); target = output/"worlds"/str(seed)
    existing = target/"result.json"
    registration = json.loads((ROOT/"configs/v6/fairness_validation.json").read_text())
    base = load_config(ROOT/"reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml")
    protocol = load_config(ROOT/"configs/v6/protocol.yaml")
    base["transport"] = protocol["transport"] | {"drain_epochs": 24}
    cfg = cell_configuration(base, registration["cell"])
    identity = config_hash({"seed": seed, "source_hash": source_hash, "registration": registration, "config": cfg})
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous["identity"] != identity:
            raise ValueError("existing world has different frozen identity")
        return previous
    try:
        world = generate_world(cfg, seed)
        trace, paths = coupled_public_trace(cfg, seed, world.observations)
        prepared = prepare_public(world, cfg)
        routes = {}
        for relay, policy in ((False, "direct"), (True, "deadline6")):
            result = simulate_transport(trace, world.observations, cfg, policy)
            routes[relay] = result
        allocations = {(allocator, relay): allocate_shared_arrivals(
            world, routes[relay].raw_arrivals, cfg, allocator, window=24)
            for allocator in ("v6_minimax", "utility_only", "v4_fallback")
            for relay in (False, True)}
        rows = []
        for allocator in ("v6_minimax", "utility_only", "v4_fallback"):
            for relay in (False, True):
                allocation = allocations[allocator, relay]
                for robust in (False, True):
                    method = f"{allocator}_R{int(relay)}_B{int(robust)}"
                    row, arrays = evaluate(world, cfg, prepared, collector_view(allocation),
                        method, .1, robust=robust, public_only=False)
                    row.update({"seed": seed, "cell": registration["cell"], "role": registration["role"],
                        "allocator": allocator, "relay": relay, "robustification": robust,
                        "route_policy": "deadline6" if relay else "direct",
                        "allocation": allocation.metrics,
                        "strata": outcome_strata_39(world, prepared[0][:len(world.truth)],
                            routes[relay].raw_arrivals, arrays["live"], arrays["reconstructed"],
                            allocation.selected, int(cfg["world"]["burn_in_steps"]))})
                    row["metrics"].update(allocation.metrics)
                    rows.append(row)
        route_records = {}
        for relay, result in routes.items():
            arrival_bytes = canonical_json(dict(sorted(result.raw_arrivals.items())))
            route_records[str(int(relay))] = {**transport_summary(result),
                "raw_arrivals_sha256": hashlib.sha256(arrival_bytes).hexdigest(),
                "raw_arrival_count": len(result.raw_arrivals)}
        record = {"status": "complete", "identity": identity, "seed": seed,
            "source_hash": source_hash, "trace_hash": trace.trace_hash,
            "trajectory_sha256": hashlib.sha256(paths.tobytes()).hexdigest(),
            "rows": rows, "transport_by_relay": route_records,
            "runtime_seconds": time.perf_counter()-started}
    except Exception:
        record = {"status": "failed", "identity": identity, "seed": seed,
            "source_hash": source_hash, "rows": [], "error": traceback.format_exc(),
            "runtime_seconds": time.perf_counter()-started}
    write(existing, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/v6/fairness_validation_v1")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    registration_path = ROOT/"configs/v6/fairness_validation.json"
    registration = json.loads(registration_path.read_text())
    inventory = source_inventory(); digest = config_hash(inventory)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "role": registration["role"], "registration": registration,
        "registration_sha256": hashlib.sha256(registration_path.read_bytes()).hexdigest(),
        "source_hash": digest, "source_files": inventory,
        "expected_worlds": len(registration["seeds"]),
        "expected_evaluations": registration["matrix"]["total_evaluations"],
        "primary_authorized": False}
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["source_hash"] != digest or previous["registration_sha256"] != manifest["registration_sha256"]:
            raise SystemExit("changed source or registration; preserve the old directory and use a new output")
    else:
        write(manifest_path, manifest)
        for name in SOURCE_FILES:
            destination = output/"source_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, destination)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_world, seed, str(output), digest): seed for seed in registration["seeds"]}
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            write(output/"progress.json", {"finished": len(results), "expected": len(registration["seeds"]),
                "failures": sum(item["status"] != "complete" for item in results),
                "complete_seeds": sorted(item["seed"] for item in results if item["status"] == "complete")})
            print(json.dumps({"seed": result["seed"], "status": result["status"],
                              "seconds": result["runtime_seconds"]}), flush=True)
    rows = [row for result in results if result["status"] == "complete" for row in result["rows"]]
    complete = len(rows) == registration["matrix"]["total_evaluations"] and all(
        result["status"] == "complete" for result in results)
    write(output/"outcomes.json", {"complete": complete, "expected_rows": registration["matrix"]["total_evaluations"],
        "rows": rows, "failures": [item for item in results if item["status"] != "complete"],
        "source_hash": digest})
    write(output/"complete_matrix.json", {"complete": complete, "worlds": len(results),
        "rows": len(rows), "source_hash": digest})
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

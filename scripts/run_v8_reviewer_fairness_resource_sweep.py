#!/usr/bin/env python3
"""Reviewer-directed processing-resource sensitivity on the locked shared path."""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.metrics import coverage_metrics  # noqa: E402
from airproof.simulator import generate_world  # noqa: E402
from airproof.v6_covariance_forcing_inputs import sha256_file, write_json  # noqa: E402
from airproof.v6_mobility import coupled_public_trace  # noqa: E402
from airproof.v7_shared_execution import integrated_transport_reviewer  # noqa: E402
from scripts.run_v7_shared_resource_confirmation import _config  # noqa: E402


REGISTRATION = ROOT / "configs/v8/reviewer_fairness_resource_sweep_v1.json"
SOURCE_REGISTRATION = ROOT / "configs/v7/reviewer_shared_resource_core_confirmation_v3.json"
SOURCE_FILES = (
    REGISTRATION,
    SOURCE_REGISTRATION,
    ROOT / "scripts/run_v8_reviewer_fairness_resource_sweep.py",
    ROOT / "scripts/run_v7_shared_resource_confirmation.py",
    ROOT / "airproof/v7_shared_execution.py",
    ROOT / "airproof/v6_experiment.py",
    ROOT / "airproof/v6_transport.py",
    ROOT / "airproof/v6_mobility.py",
    ROOT / "airproof/scheduler.py",
    ROOT / "airproof/simulator.py",
)


def _mean(values):
    values = list(values)
    return float(statistics.fmean(values)) if values else 0.0


def _budget_row(config, trace, observations, registration, budget):
    started = time.perf_counter()
    varied = deepcopy(config)
    varied["scheduler"]["budget_bytes_per_epoch"] = int(budget)
    transport, collector = integrated_transport_reviewer(
        trace,
        observations,
        varied,
        policy=registration["execution"]["relay_policy"],
        fairness=True,
        allocation_policy=registration["execution"]["allocation_policy"],
    )
    burn = int(varied["world"]["burn_in_steps"])
    steps = int(varied["world"]["steps"])
    groups = int(varied["world"]["groups"])
    allocations = [row for row in collector.allocations if burn <= int(row["epoch"]) < steps]
    scored = [record for record in collector.selected if burn <= record.epoch < steps]
    max_avoidable = [max(row["avoidable"].values(), default=0.0) for row in allocations]
    max_unavoidable = [max(row["unavoidable"].values(), default=0.0) for row in allocations]
    counts = np.asarray(
        [[int(row["counts"].get(group, 0)) for group in range(groups)] for row in allocations],
        dtype=int,
    )
    zero = np.zeros(groups, dtype=int)
    worst_zero_streak = 0
    for values in counts:
        zero = np.where(values == 0, zero + 1, 0)
        worst_zero_streak = max(worst_zero_streak, int(zero.max(initial=0)))
    metrics = transport.metrics
    return {
        "budget_bytes_per_epoch": int(budget),
        "trace_hash": trace.trace_hash,
        "scored_epochs": len(allocations),
        "globally_feasible_epochs": sum(bool(row["feasible"]) for row in allocations),
        "feasible_floor_violations": sum(
            bool(row["feasible"]) and not bool(row.get("constraint_satisfied", True))
            for row in allocations
        ),
        "epochs_with_avoidable_deficit": sum(value > 0 for value in max_avoidable),
        "epochs_with_unavoidable_deficit": sum(value > 0 for value in max_unavoidable),
        "mean_max_avoidable_deficit": _mean(max_avoidable),
        "mean_max_unavoidable_deficit": _mean(max_unavoidable),
        "worst_zero_service_streak": worst_zero_streak,
        "selected_count": len(scored),
        "coverage": coverage_metrics(scored, groups),
        "raw_timely_delivered": int(metrics["raw_timely_delivered"]),
        "total_wire_bytes": int(metrics["total_wire_bytes"]),
        "control_bytes": int(metrics["control_bytes"]),
        "receipt_issued": int(metrics["receipt_issued"]),
        "receipt_returned": int(metrics["receipt_returned"]),
        "token_violations": int(metrics["token_violations"]),
        "runtime_seconds": time.perf_counter() - started,
    }


def _job(spec):
    output_raw, registration, shared, seed, source_lock_sha = spec
    output = Path(output_raw)
    target = output / "jobs" / str(seed)
    result_path = target / "result.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("source_lock_sha256") != source_lock_sha:
            raise ValueError("stale resource-sweep job")
        return prior
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = _config(shared, registration["cell"])
    world = generate_world(config, int(seed))
    trace, _ = coupled_public_trace(config, int(seed), world.observations)
    rows = [
        _budget_row(config, trace, world.observations, shared, budget)
        for budget in registration["processing_budget_bytes_per_epoch"]
    ]
    if len({row["trace_hash"] for row in rows}) != 1:
        raise AssertionError("resource arms did not share one transport trace")
    result = {
        "schema_version": 1,
        "status": "complete",
        "role": registration["role"],
        "source_lock_sha256": source_lock_sha,
        "seed": int(seed),
        "cell": registration["cell"],
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    return result


def _analyze(registration, results, output):
    rows = [dict(row, seed=result["seed"]) for result in results for row in result["rows"]]
    expected = {
        (int(seed), int(budget))
        for seed in registration["seeds"]
        for budget in registration["processing_budget_bytes_per_epoch"]
    }
    observed = {(row["seed"], row["budget_bytes_per_epoch"]) for row in rows}
    if expected != observed or len(rows) != len(expected):
        raise ValueError("missing, duplicate, or extra resource-sweep arm")
    summaries = []
    for budget in registration["processing_budget_bytes_per_epoch"]:
        subset = [row for row in rows if row["budget_bytes_per_epoch"] == int(budget)]
        epochs = sum(row["scored_epochs"] for row in subset)
        summaries.append(
            {
                "budget_bytes_per_epoch": int(budget),
                "worlds": len(subset),
                "scored_epochs": epochs,
                "complete_floor_feasibility_rate": sum(
                    row["globally_feasible_epochs"] for row in subset
                ) / epochs,
                "feasible_floor_violations": sum(
                    row["feasible_floor_violations"] for row in subset
                ),
                "epochs_with_avoidable_deficit_rate": sum(
                    row["epochs_with_avoidable_deficit"] for row in subset
                ) / epochs,
                "epochs_with_unavoidable_deficit_rate": sum(
                    row["epochs_with_unavoidable_deficit"] for row in subset
                ) / epochs,
                "mean_max_avoidable_deficit": _mean(
                    row["mean_max_avoidable_deficit"] for row in subset
                ),
                "mean_max_unavoidable_deficit": _mean(
                    row["mean_max_unavoidable_deficit"] for row in subset
                ),
                "mean_coverage_gap": _mean(row["coverage"]["coverage_gap"] for row in subset),
                "mean_coverage_p10": _mean(row["coverage"]["coverage_p10"] for row in subset),
                "mean_selected_count": _mean(row["selected_count"] for row in subset),
                "mean_total_wire_bytes": _mean(row["total_wire_bytes"] for row in subset),
                "mean_control_bytes": _mean(row["control_bytes"] for row in subset),
                "mean_receipt_issued": _mean(row["receipt_issued"] for row in subset),
                "mean_receipt_returned": _mean(row["receipt_returned"] for row in subset),
                "token_violations": sum(row["token_violations"] for row in subset),
            }
        )
    analysis = {
        "schema_version": 1,
        "role": registration["role"],
        "status": "complete",
        "worlds": len(registration["seeds"]),
        "arms": len(rows),
        "same_trace_within_world": all(
            len({row["trace_hash"] for row in rows if row["seed"] == int(seed)}) == 1
            for seed in registration["seeds"]
        ),
        "all_registered_budgets_reported": True,
        "all_invariants_pass": all(
            summary["feasible_floor_violations"] == 0 and summary["token_violations"] == 0
            for summary in summaries
        ),
        "summaries": summaries,
        "interpretation_scope": registration["analysis_rule"],
    }
    write_json(output / "per_world.json", {"rows": rows})
    write_json(output / "analysis.json", analysis)
    return analysis


def execute(*, workers):
    registration_bytes = REGISTRATION.read_bytes()
    registration = json.loads(registration_bytes)
    shared = json.loads(SOURCE_REGISTRATION.read_text(encoding="utf-8"))
    if registration["status"] != "reviewer-directed descriptive sensitivity frozen before outcomes":
        raise ValueError("unexpected registration status")
    if not (ROOT / registration["source_confirmation"] / "completion.json").is_file():
        raise FileNotFoundError("locked shared-resource confirmation is incomplete")
    output = ROOT / registration["output"]
    if output.exists():
        raise FileExistsError("refusing to overwrite resource sweep")
    inventory = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCE_FILES}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    for path in SOURCE_FILES:
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    source_lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "source_inventory": inventory,
        "outcomes_before_lock": 0,
    }
    write_json(output / "source_lock.json", source_lock)
    source_lock_sha = sha256_file(output / "source_lock.json")
    started = time.perf_counter()
    specs = [
        (str(output), registration, shared, int(seed), source_lock_sha)
        for seed in registration["seeds"]
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, spec) for spec in specs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            elapsed = time.perf_counter() - started
            write_json(
                output / "progress.json",
                {
                    "completed_worlds": len(results),
                    "expected_worlds": len(specs),
                    "failures": 0,
                    "elapsed_seconds": elapsed,
                },
            )
            print(
                json.dumps(
                    {
                        "seed": result["seed"],
                        "status": result["status"],
                        "world_seconds": result["runtime_seconds"],
                        "elapsed_seconds": elapsed,
                    }
                ),
                flush=True,
            )
    elapsed = time.perf_counter() - started
    limit = float(registration["runtime_contract"]["total_wall_time_limit_seconds"])
    if elapsed >= limit:
        raise RuntimeError(f"experiment exceeded registered {limit}-second wall-time limit")
    analysis = _analyze(registration, results, output)
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "worlds": len(results),
            "arms": len(results) * len(registration["processing_budget_bytes_per_epoch"]),
            "wall_seconds": elapsed,
            "analysis_sha256": sha256_file(output / "analysis.json"),
        },
    )
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must lie in [1,8]")
    print(json.dumps(execute(workers=args.workers), indent=2))


if __name__ == "__main__":
    main()

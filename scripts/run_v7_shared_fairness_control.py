#!/usr/bin/env python3
"""Run the no-fairness arm on the exact v2 shared-resource worlds and traces."""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from airproof.metrics import coverage_metrics, prediction_metrics
from airproof.simulator import generate_world
from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_covariance_forcing_runner import _predict_one
from airproof.v6_lifetime_exposure import causal_lifetime_exposure_weights
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_numerical_experiment import prepare_public
from airproof.v6_residual_dynamics import public_residual_forcing
from airproof.v7_shared_execution import integrated_transport_reviewer

try:
    from scripts.run_v7_shared_resource_confirmation import _config, _estimator
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    from run_v7_shared_resource_confirmation import _config, _estimator

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v7/reviewer_shared_fairness_control_v1.json"
SHARED_REGISTRATION = ROOT / "configs/v7/reviewer_shared_resource_confirmation_v2.json"


def _job(spec):
    output_raw, shared, seed, cell, source_lock_sha = spec
    output = Path(output_raw)
    target = output / "jobs" / str(seed) / cell
    result_path = target / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result["source_lock_sha256"] != source_lock_sha:
            raise ValueError("stale no-fairness result")
        return result
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = _config(shared, cell)
    world = generate_world(config, int(seed))
    trace, _ = coupled_public_trace(config, int(seed), world.observations)
    transport, collector = integrated_transport_reviewer(
        trace,
        world.observations,
        config,
        policy=shared["execution"]["relay_policy"],
        fairness=False,
        allocation_policy=shared["execution"]["allocation_policy"],
    )
    public, _, _, public_provenance = prepare_public(world, config)
    burn = int(config["world"]["burn_in_steps"])
    steps = int(config["world"]["steps"])
    lag = int(shared["estimator"]["lag"])
    scored = [record for record in collector.selected if burn <= record.epoch < steps]
    shifted = tuple(replace(record, epoch=record.epoch - burn) for record in scored)
    arrivals = {
        record.nullifier: collector.selection_times[record.nullifier] - burn
        for record in scored
    }
    effective, exposure = causal_lifetime_exposure_weights(
        shifted,
        arrivals,
        lag=lag,
        per_user_exposure_budget=float(shared["estimator"]["per_user_exposure_budget"]),
    )
    calibration = json.loads(
        (ROOT / shared["sources"]["innovation_calibration"]).read_text()
    )
    scale = float(calibration["innovation_scales"]["world_cross_fitted_covariance"])
    mechanism = world.physical_mechanism
    center = public[burn : steps + lag]
    forcing = public_residual_forcing(
        center,
        [mechanism.transition] * len(center),
        mechanism.exogenous_forcing[burn : steps + lag],
        previous_public=public[burn - 1],
    )
    live, reconstructed, diagnostics = _predict_one(
        center,
        effective,
        arrivals,
        mechanism.transition,
        forcing,
        scale,
        _estimator(shared),
        "CANDIDATE",
    )
    truth = world.truth[burn:steps]
    metrics = prediction_metrics(truth, reconstructed[: len(truth)], world.cell_groups)
    live_metrics = prediction_metrics(truth, live[: len(truth)], world.cell_groups)
    result = {
        "schema_version": 1,
        "status": "complete",
        "source_lock_sha256": source_lock_sha,
        "seed": int(seed),
        "cell": cell,
        "trace_hash": trace.trace_hash,
        "selected_count": len(scored),
        "coverage": coverage_metrics(scored, int(config["world"]["groups"])),
        "prediction": {**metrics, **{f"live_{key}": value for key, value in live_metrics.items()}},
        "lifetime_exposure": exposure,
        "transport": {
            key: value for key, value in transport.metrics.items() if key != "local_decisions"
        },
        "solver_failure_rate": float(diagnostics["solver_failure_rate"]),
        "public_provenance": public_provenance,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    return result


def _bootstrap(values, *, seed, replicates, alpha):
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = data[rng.integers(0, len(data), size=(replicates, len(data)))].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "lower_simultaneous": float(np.quantile(sampled, alpha)),
        "upper_simultaneous": float(np.quantile(sampled, 1 - alpha)),
    }


def _analyze(registration, results, output):
    source = ROOT / registration["source_confirmation"]
    fair = {}
    for seed in registration["seeds"]:
        for cell in registration["cells"]:
            path = source / "jobs" / str(seed) / cell / "result.json"
            fair[seed, cell] = json.loads(path.read_text())
    nofair = {(row["seed"], row["cell"]): row for row in results}
    rows = []
    for seed in registration["seeds"]:
        for cell in registration["cells"]:
            fair_row = fair[seed, cell]
            nofair_row = nofair[seed, cell]
            fair_ap = next(row for row in fair_row["method_rows"] if row["method"] == "AP_LIFETIME7")
            fair_gap = float(fair_row["fairness"]["coverage_gap"])
            nofair_gap = float(nofair_row["coverage"]["coverage_gap"])
            rows.append(
                {
                    "seed": seed,
                    "cell": cell,
                    "trace_match": fair_row["trace_hash"] == nofair_row["trace_hash"],
                    "fair_gap": fair_gap,
                    "no_fair_gap": nofair_gap,
                    "gap_reduction": 1 - fair_gap / nofair_gap if nofair_gap > 0 else None,
                    "fair_worst_group_rmse": fair_ap["worst_group_rmse"],
                    "no_fair_worst_group_rmse": nofair_row["prediction"]["worst_group_rmse"],
                    "worst_group_rmse_reduction": 1
                    - fair_ap["worst_group_rmse"]
                    / nofair_row["prediction"]["worst_group_rmse"],
                }
            )
    clean = [row for row in rows if row["cell"] == "severe_clean"]
    rules = registration["decision_rules"]
    alpha = 0.05 / int(rules["one_sided_bonferroni_family_size"])
    gap = _bootstrap(
        [row["gap_reduction"] for row in clean],
        seed=int(rules["bootstrap_seed"]),
        replicates=int(rules["bootstrap_replicates"]),
        alpha=alpha,
    )
    error = _bootstrap(
        [row["worst_group_rmse_reduction"] for row in clean],
        seed=int(rules["bootstrap_seed"]) + 1,
        replicates=int(rules["bootstrap_replicates"]),
        alpha=alpha,
    )
    analysis = {
        "schema_version": 1,
        "worlds": len(registration["seeds"]),
        "jobs": len(results),
        "all_trace_pairs_match": all(row["trace_match"] for row in rows),
        "clean_gap_reduction": gap,
        "clean_worst_group_rmse_reduction": error,
        "gates": {
            "H6": gap["lower_simultaneous"]
            >= rules["clean_contributor_gap_reduction_lower"],
            "H7": error["lower_simultaneous"]
            >= rules["clean_worst_group_rmse_reduction_lower"],
        },
    }
    analysis["all_gates_pass"] = all(analysis["gates"].values())
    (output / "paired_worlds.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    write_json(output / "analysis.json", analysis)
    return analysis


def execute(*, workers):
    registration_bytes = REGISTRATION.read_bytes()
    registration = json.loads(registration_bytes)
    shared = json.loads(SHARED_REGISTRATION.read_text())
    if registration["status"] != "frozen before reading shared-resource v2 outcomes":
        raise ValueError("unexpected fairness control registration")
    source = ROOT / registration["source_confirmation"]
    if not (source / "analysis.json").is_file():
        raise FileNotFoundError("shared-resource v2 confirmation is incomplete")
    output = ROOT / registration["output"]
    if output.exists():
        raise FileExistsError("refusing to overwrite fairness control")
    sources = [
        REGISTRATION,
        SHARED_REGISTRATION,
        ROOT / "scripts/run_v7_shared_fairness_control.py",
        ROOT / "airproof/v7_shared_execution.py",
        ROOT / "airproof/v6_transport.py",
        ROOT / "airproof/v6_estimator.py",
        ROOT / "airproof/v6_lifetime_exposure.py",
    ]
    inventory = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sources}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    for path in sources:
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    lock = {
        "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "source_inventory": inventory,
        "outcomes_before_lock": 0,
    }
    write_json(output / "source_lock.json", lock)
    source_lock_sha = sha256_file(output / "source_lock.json")
    specs = [
        (str(output), shared, seed, cell, source_lock_sha)
        for seed in registration["seeds"]
        for cell in registration["cells"]
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, spec) for spec in specs]
        for future in as_completed(futures):
            results.append(future.result())
            write_json(
                output / "progress.json",
                {"completed_jobs": len(results), "expected_jobs": len(specs), "failures": 0},
            )
    analysis = _analyze(registration, results, output)
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "jobs": len(results),
            "analysis_sha256": sha256_file(output / "analysis.json"),
        },
    )
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must lie in [1,4]")
    print(json.dumps(execute(workers=args.workers), indent=2))


if __name__ == "__main__":
    main()

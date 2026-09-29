#!/usr/bin/env python3
"""Diagnose a causal horizon-paced B=7 grant on exposed shared-v2 worlds."""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from airproof.metrics import prediction_metrics
from airproof.simulator import generate_world
from airproof.v6_covariance_forcing_runner import _predict_one
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_numerical_experiment import prepare_public
from airproof.v6_residual_dynamics import public_residual_forcing
from airproof.v7_paced_lifetime import causal_paced_lifetime_exposure_weights
from airproof.v7_shared_execution import integrated_transport_reviewer

try:
    from scripts.run_v7_shared_resource_confirmation import _config, _estimator
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    from run_v7_shared_resource_confirmation import _config, _estimator

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v7/reviewer_paced_lifetime_development_v1.json"
SHARED = ROOT / "configs/v7/reviewer_shared_resource_confirmation_v2.json"


def _job(spec: tuple[int, str, dict, dict]) -> dict:
    seed, cell, registration, shared = spec
    config = _config(shared, cell)
    world = generate_world(config, seed)
    trace, _ = coupled_public_trace(config, seed, world.observations)
    transport, collector = integrated_transport_reviewer(
        trace,
        world.observations,
        config,
        policy=shared["execution"]["relay_policy"],
        fairness=True,
        allocation_policy=shared["execution"]["allocation_policy"],
    )
    public, _, _, _ = prepare_public(world, config)
    burn = int(config["world"]["burn_in_steps"])
    steps = int(config["world"]["steps"])
    lag = int(shared["estimator"]["lag"])
    scored = [record for record in collector.selected if burn <= record.epoch < steps]
    shifted = tuple(replace(record, epoch=record.epoch - burn) for record in scored)
    arrivals = {
        record.nullifier: collector.selection_times[record.nullifier] - burn
        for record in scored
    }
    candidate = registration["candidate"]
    effective, exposure = causal_paced_lifetime_exposure_weights(
        shifted,
        arrivals,
        lag=lag,
        per_user_exposure_budget=float(candidate["lifetime_budget"]),
        horizon_start=int(candidate["horizon_start"]),
        horizon_end=int(candidate["horizon_end"]),
    )
    calibration = json.loads((ROOT / shared["sources"]["innovation_calibration"]).read_text())
    scale = float(calibration["innovation_scales"]["world_cross_fitted_covariance"])
    mechanism = world.physical_mechanism
    if mechanism is None:
        raise ValueError("physical mechanism unavailable")
    center = public[burn : steps + lag]
    forcing = public_residual_forcing(
        center,
        [mechanism.transition] * len(center),
        mechanism.exogenous_forcing[burn : steps + lag],
        previous_public=public[burn - 1],
    )
    truth = world.truth[burn:steps]
    threshold = float(np.quantile(world.truth[:burn], 0.95))
    event = truth >= threshold
    rows = []
    for method in ("AP_LIFETIME7_PACED", "PUBLIC", "SQ"):
        records = () if method == "PUBLIC" else effective
        runner_method = "CANDIDATE" if method == "AP_LIFETIME7_PACED" else method
        live, reconstructed, diagnostics = _predict_one(
            center,
            records,
            arrivals,
            mechanism.transition,
            forcing,
            scale,
            _estimator(shared),
            runner_method,
        )
        metrics = prediction_metrics(truth, reconstructed[: len(truth)], world.cell_groups)
        rows.append(
            {
                "method": method,
                "rmse": metrics["rmse"],
                "event_recall": float(np.mean(live[: len(truth)][event] >= threshold)),
                "solver_failure_rate": diagnostics["solver_failure_rate"],
                "maximum_correction": diagnostics["maximum_correction"],
            }
        )
    return {
        "seed": seed,
        "cell": cell,
        "trace_hash": trace.trace_hash,
        "raw_timely_delivered": transport.metrics["raw_timely_delivered"],
        "selected_count": len(scored),
        "exposure": exposure,
        "methods": rows,
    }


def _analyze(rows: list[dict], registration: dict) -> dict:
    indexed = {(row["seed"], row["cell"]): row for row in rows}
    paired = []
    for seed in registration["seeds"]:
        clean = indexed[(seed, "severe_clean")]
        attack = indexed[(seed, "severe_hotspot")]
        cm = {row["method"]: row for row in clean["methods"]}
        am = {row["method"]: row for row in attack["methods"]}
        ap_growth = am["AP_LIFETIME7_PACED"]["rmse"] - cm["AP_LIFETIME7_PACED"]["rmse"]
        sq_growth = am["SQ"]["rmse"] - cm["SQ"]["rmse"]
        paired.append(
            {
                "seed": seed,
                "clean_ap_to_sq": cm["AP_LIFETIME7_PACED"]["rmse"] / cm["SQ"]["rmse"],
                "clean_ap_to_public": cm["AP_LIFETIME7_PACED"]["rmse"] / cm["PUBLIC"]["rmse"],
                "event_recall_loss_vs_sq": cm["SQ"]["event_recall"] - cm["AP_LIFETIME7_PACED"]["event_recall"],
                "hotspot_growth_ap": ap_growth,
                "hotspot_growth_sq": sq_growth,
                "hotspot_attenuation": 1.0 - ap_growth / sq_growth if sq_growth > 0 else None,
            }
        )
    gates = registration["descriptive_gates"]
    finite_attenuation = [row["hotspot_attenuation"] for row in paired if row["hotspot_attenuation"] is not None]
    decisions = {
        "clean_noninferiority": bool(np.mean([row["clean_ap_to_sq"] for row in paired]) <= gates["clean_AP_to_SQ_ratio_upper"]),
        "citizen_value": bool(np.mean([row["clean_ap_to_public"] for row in paired]) <= gates["clean_AP_to_PUBLIC_ratio_upper"]),
        "positive_sq_growth_every_world": all(row["hotspot_growth_sq"] > 0 for row in paired),
        "hotspot_attenuation": bool(len(finite_attenuation) == len(paired) and np.mean(finite_attenuation) >= gates["hotspot_growth_attenuation_lower"]),
        "event_recall": bool(np.mean([row["event_recall_loss_vs_sq"] for row in paired]) <= gates["event_recall_loss_vs_SQ_upper_percentage_points"] / 100),
    }
    return {
        "schema_version": 1,
        "role": registration["role"],
        "paired": paired,
        "means": {
            key: float(np.mean([row[key] for row in paired]))
            for key in ("clean_ap_to_sq", "clean_ap_to_public", "event_recall_loss_vs_sq", "hotspot_growth_ap", "hotspot_growth_sq")
        },
        "mean_hotspot_attenuation": float(np.mean(finite_attenuation)) if finite_attenuation else None,
        "decisions": decisions,
        "all_descriptive_gates_pass": all(decisions.values()),
    }


def execute(*, workers: int) -> dict:
    registration = json.loads(REGISTRATION.read_text())
    shared = json.loads(SHARED.read_text())
    output = ROOT / registration["output"]
    output.mkdir(parents=True, exist_ok=True)
    specs = [(seed, cell, registration, shared) for seed in registration["seeds"] for cell in registration["cells"]]
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, spec) for spec in specs]
        for future in as_completed(futures):
            rows.append(future.result())
            (output / "progress.json").write_text(json.dumps({"completed": len(rows), "expected": len(specs)}))
    (output / "results.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    analysis = _analyze(rows, registration)
    analysis["wall_seconds"] = time.perf_counter() - started
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True))
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(execute(workers=args.workers), indent=2))

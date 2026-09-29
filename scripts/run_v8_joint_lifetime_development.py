#!/usr/bin/env python3
"""Bounded exposed development for v8 joint lifetime allocation."""
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
from airproof.v7_shared_execution import integrated_transport_reviewer
from airproof.v8_joint_allocation import causal_joint_lifetime_exposure_weights

try:
    from scripts.run_v7_shared_resource_confirmation import _config, _estimator
except ModuleNotFoundError:
    from run_v7_shared_resource_confirmation import _config, _estimator


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v8/joint_lifetime_allocation_development_v1.json"


def _method_row(method, truth, event, threshold, groups, live, reconstructed, diagnostics):
    metrics = prediction_metrics(truth, reconstructed[: len(truth)], groups)
    return {
        "method": method,
        "rmse": metrics["rmse"],
        "worst_group_rmse": metrics["worst_group_rmse"],
        "event_recall": float(np.mean(live[: len(truth)][event] >= threshold)),
        "solver_failure_rate": diagnostics["solver_failure_rate"],
        "maximum_correction": diagnostics["maximum_correction"],
    }


def _job(spec):
    seed, registration, shared = spec
    started = time.perf_counter()
    clean_config = _config(shared, "severe_clean")
    clean_world = generate_world(clean_config, seed)
    hotspot_config = _config(shared, "severe_hotspot")
    hotspot_world = generate_world(hotspot_config, seed)
    if [item.nullifier for item in clean_world.observations] != [
        item.nullifier for item in hotspot_world.observations
    ]:
        raise AssertionError("paired worlds do not share observation identities")
    trace, _ = coupled_public_trace(clean_config, seed, clean_world.observations)
    transport, collector = integrated_transport_reviewer(
        trace,
        clean_world.observations,
        clean_config,
        policy=shared["execution"]["relay_policy"],
        fairness=True,
        allocation_policy=shared["execution"]["allocation_policy"],
    )
    public, _, _, _ = prepare_public(clean_world, clean_config)
    burn = int(clean_config["world"]["burn_in_steps"])
    steps = int(clean_config["world"]["steps"])
    lag = int(registration["fixed"]["lag"])
    selected_ids = {
        item.nullifier: collector.selection_times[item.nullifier]
        for item in collector.selected
        if burn <= item.epoch < steps
    }
    calibration = json.loads((ROOT / shared["sources"]["innovation_calibration"]).read_text())
    scale = float(calibration["innovation_scales"]["world_cross_fitted_covariance"])
    rows = []
    for cell, world in (("severe_clean", clean_world), ("severe_hotspot", hotspot_world)):
        lookup = {item.nullifier: item for item in world.observations}
        shifted = tuple(
            replace(lookup[nullifier], epoch=lookup[nullifier].epoch - burn)
            for nullifier in selected_ids
        )
        arrivals = {nullifier: arrival - burn for nullifier, arrival in selected_ids.items()}
        mechanism = world.physical_mechanism
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
        for candidate in registration["candidates"]:
            effective, exposure = causal_joint_lifetime_exposure_weights(
                shifted,
                arrivals,
                center,
                lag=lag,
                per_user_exposure_budget=float(registration["fixed"]["lifetime_budget"]),
                horizon_start=0,
                horizon_end=steps - burn,
                decision_interval=int(candidate["decision_interval"]),
                innovation_scale_floor=float(registration["fixed"]["innovation_scale_floor"]),
                innovation_clip=float(registration["fixed"]["innovation_clip"]),
                fairness_strength=float(candidate["fairness_strength"]),
                outcome_strength=float(registration["fixed"]["outcome_strength"]),
            )
            for method, runner in (("AP", "CANDIDATE"), ("SQ", "SQ")):
                live, reconstructed, diagnostics = _predict_one(
                    center,
                    effective,
                    arrivals,
                    mechanism.transition,
                    forcing,
                    scale,
                    _estimator(shared),
                    runner,
                )
                rows.append(
                    {
                        "seed": seed,
                        "cell": cell,
                        "candidate": candidate["id"],
                        "exposure": exposure,
                        **_method_row(
                            method,
                            truth,
                            event,
                            threshold,
                            world.cell_groups,
                            live,
                            reconstructed,
                            diagnostics,
                        ),
                    }
                )
    return {
        "seed": seed,
        "trace_hash": trace.trace_hash,
        "selected_count": len(selected_ids),
        "raw_timely_delivered": transport.metrics["raw_timely_delivered"],
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }


def _analyze(registration, jobs):
    rows = [row for job in jobs for row in job["rows"]]
    index = {
        (row["seed"], row["cell"], row["candidate"], row["method"]): row
        for row in rows
    }
    outcome_control = {
        candidate["decision_interval"]: candidate["id"]
        for candidate in registration["candidates"]
        if float(candidate["fairness_strength"]) == 0.0
    }
    summaries = []
    gates = registration["descriptive_gates"]
    for candidate in registration["candidates"]:
        cid = candidate["id"]
        control = outcome_control[int(candidate["decision_interval"])]
        paired = []
        for seed in registration["reuse"]["seeds"]:
            clean_ap = index[seed, "severe_clean", cid, "AP"]
            clean_sq = index[seed, "severe_clean", cid, "SQ"]
            attack_ap = index[seed, "severe_hotspot", cid, "AP"]
            attack_sq = index[seed, "severe_hotspot", cid, "SQ"]
            control_ap = index[seed, "severe_clean", control, "AP"]
            ap_growth = attack_ap["rmse"] - clean_ap["rmse"]
            sq_growth = attack_sq["rmse"] - clean_sq["rmse"]
            fair_gap = float(clean_ap["exposure"]["effective_contributor_gap"])
            control_gap = float(control_ap["exposure"]["effective_contributor_gap"])
            paired.append(
                {
                    "seed": seed,
                    "clean_ap_to_sq": clean_ap["rmse"] / clean_sq["rmse"],
                    "event_recall_loss_vs_sq": clean_sq["event_recall"] - clean_ap["event_recall"],
                    "hotspot_attenuation": 1.0 - ap_growth / sq_growth if sq_growth > 0 else None,
                    "contributor_gap_reduction": (
                        1.0 - fair_gap / control_gap if control_gap > 0 else 0.0
                    ),
                    "worst_group_rmse_gain": 1.0
                    - clean_ap["worst_group_rmse"] / control_ap["worst_group_rmse"],
                    "maximum_exposure": clean_ap["exposure"]["maximum_user_lifetime_exposure"],
                    "solver_failure_rate": max(
                        clean_ap["solver_failure_rate"],
                        clean_sq["solver_failure_rate"],
                        attack_ap["solver_failure_rate"],
                        attack_sq["solver_failure_rate"],
                    ),
                }
            )
        means = {
            key: float(np.mean([row[key] for row in paired]))
            for key in (
                "clean_ap_to_sq",
                "event_recall_loss_vs_sq",
                "contributor_gap_reduction",
                "worst_group_rmse_gain",
                "maximum_exposure",
                "solver_failure_rate",
            )
        }
        attenuations = [row["hotspot_attenuation"] for row in paired]
        means["hotspot_attenuation"] = (
            float(np.mean(attenuations)) if all(value is not None for value in attenuations) else None
        )
        decisions = {
            "clean_noninferiority": means["clean_ap_to_sq"] <= gates["clean_AP_to_SQ_ratio_upper"],
            "hotspot_attenuation": means["hotspot_attenuation"] is not None
            and means["hotspot_attenuation"] >= gates["hotspot_attenuation_lower"],
            "event_recall": means["event_recall_loss_vs_sq"]
            <= gates["event_recall_loss_vs_SQ_upper_percentage_points"] / 100,
            "contributor_gap": means["contributor_gap_reduction"]
            >= gates["contributor_gap_reduction_lower"],
            "worst_group_rmse": means["worst_group_rmse_gain"]
            >= gates["worst_group_RMSE_gain_lower"],
            "lifetime": max(row["maximum_exposure"] for row in paired)
            <= gates["maximum_user_lifetime_exposure_upper"] + 1e-10,
            "solver": means["solver_failure_rate"] <= gates["solver_failure_rate"],
        }
        summaries.append(
            {
                "candidate": cid,
                "control": control,
                "paired": paired,
                "means": means,
                "decisions": decisions,
                "eligible": all(decisions.values()),
            }
        )
    eligible = [row for row in summaries if row["eligible"]]
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (
                -min(
                    row["means"]["hotspot_attenuation"] / gates["hotspot_attenuation_lower"],
                    row["means"]["contributor_gap_reduction"] / gates["contributor_gap_reduction_lower"],
                    row["means"]["worst_group_rmse_gain"] / gates["worst_group_RMSE_gain_lower"],
                ),
                next(c["decision_interval"] for c in registration["candidates"] if c["id"] == row["candidate"]),
                next(c["fairness_strength"] for c in registration["candidates"] if c["id"] == row["candidate"]),
                row["candidate"],
            ),
        )[0]["candidate"]
    return {
        "schema_version": 1,
        "role": registration["role"],
        "jobs": len(jobs),
        "candidates": summaries,
        "selected_candidate": selected,
        "all_candidates_retained": len(summaries) == len(registration["candidates"]),
    }


def execute(workers):
    registration = json.loads(REGISTRATION.read_text())
    shared = json.loads((ROOT / registration["source_registration"]).read_text())
    output = ROOT / registration["output"]
    if output.exists():
        raise FileExistsError("refusing to overwrite v8 development outcomes")
    output.mkdir(parents=True)
    started = time.perf_counter()
    jobs = []
    specs = [(seed, registration, shared) for seed in registration["reuse"]["seeds"]]
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        futures = [pool.submit(_job, spec) for spec in specs]
        for future in as_completed(futures):
            jobs.append(future.result())
            (output / "progress.json").write_text(
                json.dumps({"completed": len(jobs), "expected": len(specs)})
            )
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs), encoding="utf-8"
    )
    analysis = _analyze(registration, jobs)
    analysis["wall_seconds"] = time.perf_counter() - started
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must lie in [1,4]")
    print(json.dumps(execute(args.workers), indent=2))

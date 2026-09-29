#!/usr/bin/env python3
"""Run clip-only and cap-only attribution controls on shared-core v3 evidence."""
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

from airproof.metrics import prediction_metrics
from airproof.records import canonical_json
from airproof.simulator import generate_world
from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_covariance_forcing_runner import _predict_one
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_numerical_experiment import prepare_public
from airproof.v6_residual_dynamics import public_residual_forcing
from airproof.v7_shared_execution import integrated_transport_reviewer

try:
    from scripts.run_v7_shared_resource_confirmation import _config, _estimator
except ModuleNotFoundError:  # direct ``python scripts/...py`` execution
    from run_v7_shared_resource_confirmation import _config, _estimator

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v7/reviewer_shared_core_component_controls_v1.json"
SHARED_REGISTRATION = ROOT / "configs/v7/reviewer_shared_resource_core_confirmation_v3.json"


def _selected_hash(records, arrivals) -> str:
    return hashlib.sha256(
        canonical_json(
            [
                {
                    "nullifier": record.nullifier,
                    "epoch": record.epoch,
                    "arrival": arrivals[record.nullifier],
                    "quality": record.quality,
                }
                for record in records
            ]
        )
    ).hexdigest()


def _job(spec: tuple[str, dict, dict, int, str, str]) -> dict:
    output_raw, registration, shared, seed, cell, source_lock_sha = spec
    output = Path(output_raw)
    target = output / "jobs" / str(seed) / cell
    result_path = target / "result.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior["source_lock_sha256"] != source_lock_sha:
            raise ValueError("stale component-control result")
        return prior
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
        record.nullifier: collector.selection_times[record.nullifier] - burn for record in scored
    }
    selected_hash = _selected_hash(shifted, arrivals)
    source_row = json.loads(
        (
            ROOT
            / registration["source_confirmation"]
            / "jobs"
            / str(seed)
            / cell
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    if trace.trace_hash != source_row["trace_hash"] or selected_hash != source_row["selected_hash"]:
        raise ValueError("component control did not reproduce the registered evidence set")

    calibration = json.loads(
        (ROOT / shared["sources"]["innovation_calibration"]).read_text(encoding="utf-8")
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
    base = _estimator(shared)
    truth = world.truth[burn:steps]
    threshold = float(np.quantile(world.truth[:burn], 0.95))
    events = truth >= threshold
    method_rows = []
    for name, definition in registration["methods"].items():
        arm = replace(
            base,
            loss=definition["loss"],
            input_clip=bool(definition["input_clip"]),
            output_cap=bool(definition["output_cap"]),
            clip_delta=float(definition.get("clip_delta", base.clip_delta)),
            cap=float(definition.get("cap", base.cap)),
        )
        live, reconstructed, diagnostics = _predict_one(
            center,
            shifted,
            arrivals,
            mechanism.transition,
            forcing,
            scale,
            arm,
            "CANDIDATE",
        )
        metrics = prediction_metrics(truth, reconstructed[: len(truth)], world.cell_groups)
        live_metrics = prediction_metrics(truth, live[: len(truth)], world.cell_groups)
        method_rows.append(
            {
                "method": name,
                **metrics,
                **{f"live_{key}": value for key, value in live_metrics.items()},
                "event_recall": float(np.mean(live[: len(truth)][events] >= threshold)),
                "maximum_correction": float(diagnostics["maximum_correction"]),
                "solver_failure_rate": float(diagnostics["solver_failure_rate"]),
            }
        )
    result = {
        "schema_version": 1,
        "status": "complete",
        "source_lock_sha256": source_lock_sha,
        "seed": int(seed),
        "cell": cell,
        "trace_hash": trace.trace_hash,
        "selected_hash": selected_hash,
        "selected_count": len(scored),
        "method_rows": method_rows,
        "runtime_seconds": time.perf_counter() - started,
        "transport_identity_only": {
            "raw_timely_delivered": transport.metrics["raw_timely_delivered"],
            "receipt_issued": transport.metrics["receipt_issued"],
        },
    }
    write_json(result_path, result)
    return result


def _bootstrap_ratio(left, right, *, seed: int, replicates: int) -> list[float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(left), size=(replicates, len(left)))
    values = left[index].mean(axis=1) / right[index].mean(axis=1)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _analyze(registration: dict, results: list[dict], output: Path) -> dict:
    source = ROOT / registration["source_confirmation"]
    controls = {(row["seed"], row["cell"]): row for row in results}
    methods = ["AP_SHARED_CORE", "SQ", "HUBER", *registration["methods"]]
    per_seed = []
    for seed in registration["seeds"]:
        rows = {}
        for cell in registration["cells"]:
            source_row = json.loads(
                (source / "jobs" / str(seed) / cell / "result.json").read_text(encoding="utf-8")
            )
            rows[cell] = {
                item["method"]: item for item in source_row["method_rows"]
            } | {
                item["method"]: item for item in controls[seed, cell]["method_rows"]
            }
        per_seed.append(
            {
                "seed": seed,
                **{
                    f"{cell}_{method}_rmse": rows[cell][method]["rmse"]
                    for cell in registration["cells"]
                    for method in methods
                },
                **{
                    f"{method}_growth": (
                        rows["severe_hotspot"][method]["rmse"]
                        - rows["severe_clean"][method]["rmse"]
                    )
                    for method in methods
                },
            }
        )
    means = {
        method: {
            "clean_rmse": float(
                np.mean([row[f"severe_clean_{method}_rmse"] for row in per_seed])
            ),
            "hotspot_rmse": float(
                np.mean([row[f"severe_hotspot_{method}_rmse"] for row in per_seed])
            ),
            "hotspot_growth": float(np.mean([row[f"{method}_growth"] for row in per_seed])),
        }
        for method in methods
    }
    sq_growth = means["SQ"]["hotspot_growth"]
    for method in methods:
        means[method]["attenuation_vs_sq_growth"] = (
            float(1 - means[method]["hotspot_growth"] / sq_growth) if sq_growth > 0 else None
        )
    reps = int(registration["analysis"]["bootstrap_replicates"])
    seed = int(registration["analysis"]["bootstrap_seed"])
    ratios = {
        method: {
            "clean_ratio_to_sq": float(means[method]["clean_rmse"] / means["SQ"]["clean_rmse"]),
            "clean_ratio_to_sq_ci95": _bootstrap_ratio(
                [row[f"severe_clean_{method}_rmse"] for row in per_seed],
                [row["severe_clean_SQ_rmse"] for row in per_seed],
                seed=seed + offset,
                replicates=reps,
            ),
        }
        for offset, method in enumerate(methods)
    }
    analysis = {
        "schema_version": 1,
        "role": registration["analysis"]["role"],
        "worlds": len(registration["seeds"]),
        "jobs": len(results),
        "all_source_evidence_matches": all(
            row["trace_hash"]
            == json.loads(
                (
                    source / "jobs" / str(row["seed"]) / row["cell"] / "result.json"
                ).read_text(encoding="utf-8")
            )["trace_hash"]
            for row in results
        ),
        "means": means,
        "clean_ratios": ratios,
        "interpretation": (
            "INPUT_CLIP_ONLY and OUTPUT_CAP_ONLY change one bounded interface at a time; "
            "HUBER isolates bounded score and AP_SHARED_CORE adds the final public-centered cap."
        ),
    }
    (output / "per_seed.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_seed), encoding="utf-8"
    )
    write_json(output / "analysis.json", analysis)
    return analysis


def execute(*, workers: int) -> dict:
    registration_bytes = REGISTRATION.read_bytes()
    registration = json.loads(registration_bytes)
    shared = json.loads(SHARED_REGISTRATION.read_text(encoding="utf-8"))
    if registration["status"] != (
        "frozen after partial core-v3 progress and before any component-control outcome"
    ):
        raise ValueError("unexpected component-control registration status")
    source = ROOT / registration["source_confirmation"]
    if not (source / "completion.json").is_file():
        raise FileNotFoundError("shared-resource core v3 confirmation is incomplete")
    output = ROOT / registration["output"]
    if output.exists():
        raise FileExistsError("refusing to overwrite component controls")
    sources = [
        REGISTRATION,
        SHARED_REGISTRATION,
        ROOT / "scripts/run_v7_shared_core_component_controls.py",
        ROOT / "scripts/run_v7_shared_resource_confirmation.py",
        ROOT / "airproof/v7_shared_execution.py",
        ROOT / "airproof/v6_estimator.py",
    ]
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    inventory = {}
    for path in sources:
        inventory[path.relative_to(ROOT).as_posix()] = sha256_file(path)
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
        (str(output), registration, shared, seed, cell, source_lock_sha)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must lie in [1,4]")
    print(json.dumps(execute(workers=args.workers), indent=2))


if __name__ == "__main__":
    main()

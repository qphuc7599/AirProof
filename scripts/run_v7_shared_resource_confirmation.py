#!/usr/bin/env python3
"""Run the registered shared-resource reviewer confirmation."""
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
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import psutil

from airproof.config import config_hash, load_config
from airproof.metrics import coverage_metrics, prediction_metrics
from airproof.records import canonical_json
from airproof.simulator import generate_world
from airproof.v5_experiment import cell_configuration
from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_covariance_forcing_runner import _predict_one
from airproof.v6_estimator import EstimatorConfig
from airproof.v6_lifetime_exposure import causal_lifetime_exposure_weights
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_numerical_experiment import prepare_public
from airproof.v6_residual_dynamics import public_residual_forcing
from airproof.v7_privacy_controls import reviewer_release_evidence
from airproof.v7_shared_execution import integrated_transport_reviewer

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = Path("configs/v7/reviewer_shared_resource_confirmation_v2.json")
SEED_REGISTRY = Path("configs/v7/seed_registry.json")


def _source_paths(registration: dict, registration_path: Path = REGISTRATION) -> list[Path]:
    relatives = [
        registration_path.as_posix(),
        SEED_REGISTRY.as_posix(),
        registration["sources"]["base_configuration"],
        registration["sources"]["resource_protocol"],
        registration["sources"]["innovation_calibration"],
        "scripts/run_v7_shared_resource_confirmation.py",
        "airproof/v7_shared_execution.py",
        "airproof/v7_privacy_controls.py",
        "airproof/v6_experiment.py",
        "airproof/v6_transport.py",
        "airproof/v6_resources.py",
        "airproof/v6_receipts.py",
        "airproof/v6_audit_return.py",
        "airproof/v6_public_checkpoint.py",
        "airproof/v6_fairness.py",
        "airproof/v6_estimator.py",
        "airproof/v6_lifetime_exposure.py",
        "airproof/v6_residual_dynamics.py",
        "airproof/v6_covariance_forcing_runner.py",
        "airproof/v6_numerical_experiment.py",
        "airproof/v6_mobility.py",
        "airproof/v5_experiment.py",
        "airproof/scheduler.py",
        "airproof/simulator.py",
        "airproof/experiment.py",
        "airproof/metrics.py",
        "airproof/records.py",
    ]
    paths = [ROOT / relative for relative in relatives]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered source missing: {missing}")
    return paths


def _source_inventory(
    registration: dict, registration_path: Path = REGISTRATION
) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in _source_paths(registration, registration_path)
    }


def _load_registration(registration_path: Path = REGISTRATION) -> dict:
    registration = json.loads((ROOT / registration_path).read_text(encoding="utf-8"))
    seeds = registration["worlds"]["seeds"]
    registry = json.loads((ROOT / SEED_REGISTRY).read_text(encoding="utf-8"))
    if (
        registration.get("schema_version") != 1
        or "frozen" not in registration.get("status", "")
        or seeds != registry["namespaces"][registration["worlds"]["seed_namespace"]]
        or len(seeds) != 12
        or len(set(seeds)) != 12
        or registration["worlds"]["cells"] != ["severe_clean", "severe_hotspot"]
    ):
        raise ValueError("shared-resource registration identity changed")
    return registration


def _config(registration: dict, cell: str) -> dict:
    config = load_config(ROOT / registration["sources"]["base_configuration"])
    protocol = load_config(ROOT / registration["sources"]["resource_protocol"])
    contract = registration["execution"]
    config["transport"] = dict(protocol["transport"])
    config["transport"].update(
        drain_epochs=int(contract["ttl_epochs"]),
        useful_lag_epochs=int(contract["raw_useful_lag_epochs"]),
    )
    config["privacy"]["nonnegative_raw_lower"] = float(
        registration["privacy"]["nonnegative_raw_domain"][0]
    )
    config["privacy"]["nonnegative_raw_upper"] = float(
        registration["privacy"]["nonnegative_raw_domain"][1]
    )
    config["scheduler"]["algorithm"] = contract["allocation_policy"]
    account = registration["accountability"]
    public_checkpoint = dict(account.get("public_checkpoint", {"enabled": False}))
    if bool(account.get("public_checkpoint_in_common_execution", False)):
        public_checkpoint["enabled"] = True
    config["audit"] = {
        "return_mode": account["return_mode"],
        "max_records": int(account["max_records"]),
        "max_content_bytes": int(account["max_content_bytes"]),
        "durable_log_capacity_bytes": int(account["durable_log_capacity_bytes"]),
        "collector_return_buffer_bytes": int(account["collector_return_buffer_bytes"]),
        "inclusion_deadline_epochs": int(account["inclusion_deadline_epochs"]),
        "max_epoch": int(account["max_epoch"]),
        "public_checkpoint": public_checkpoint,
    }
    config = cell_configuration(config, cell)
    worlds = registration["worlds"]
    observed = [
        int(config["world"]["agents"]),
        int(config["world"]["grid_side"]),
        int(config["world"]["steps"]),
        int(config["world"]["burn_in_steps"]),
        int(config["transport"]["drain_epochs"]),
    ]
    expected = [
        worlds["agents"],
        worlds["grid_side"],
        worlds["acquisition_epochs"],
        worlds["burn_in_epochs"],
        worlds["drain_epochs"],
    ]
    if observed != expected:
        raise ValueError("executable configuration differs from registered scale")
    return config


def _estimator(registration: dict) -> EstimatorConfig:
    fixed = registration["estimator"]
    return EstimatorConfig(
        cap=float(fixed["cap"]),
        lag=int(fixed["lag"]),
        huber_delta=float(fixed["huber_delta"]),
        lambda_temporal=float(fixed["lambda_temporal"]),
        lambda_spatial=float(fixed["lambda_spatial"]),
        lambda_zero=float(fixed["lambda_zero"]),
        time_step=float(fixed["time_step"]),
        tolerance=float(fixed["tolerance"]),
        max_iterations=int(fixed["max_iterations"]),
        numerical_refinements=int(fixed["numerical_refinements"]),
        loss="huber",
        output_cap=True,
        input_clip=False,
    )


def _finite_transport_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "local_decisions"}


def _prediction_identity(
    source_lock_sha: str,
    seed: int,
    cell: str,
    method: str,
    trace_hash: str,
    selected_hash: str,
    estimator: EstimatorConfig,
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "source_lock_sha256": source_lock_sha,
                "seed": seed,
                "cell": cell,
                "method": method,
                "trace_hash": trace_hash,
                "selected_hash": selected_hash,
                "estimator": asdict(estimator),
            }
        )
    ).hexdigest()


def _job(spec: tuple[str, dict, int, str, str]) -> dict:
    output_raw, registration, seed, cell, source_lock_sha = spec
    output = Path(output_raw)
    target = output / "jobs" / str(seed) / cell
    result_path = target / "result.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("source_lock_sha256") != source_lock_sha:
            raise ValueError("stale shared-resource job")
        return prior
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cpu_started = time.process_time()
    process = psutil.Process()
    config = _config(registration, cell)
    world = generate_world(config, seed)
    trace, _ = coupled_public_trace(config, seed, world.observations)
    transport, collector = integrated_transport_reviewer(
        trace,
        world.observations,
        config,
        policy=registration["execution"]["relay_policy"],
        fairness=bool(registration["execution"]["fairness"]),
        allocation_policy=registration["execution"]["allocation_policy"],
    )
    public, _, _, public_provenance = prepare_public(world, config)
    burn = int(config["world"]["burn_in_steps"])
    steps = int(config["world"]["steps"])
    lag = int(registration["estimator"]["lag"])
    scored = [record for record in collector.selected if burn <= record.epoch < steps]
    shifted = tuple(replace(record, epoch=record.epoch - burn) for record in scored)
    arrivals = {
        record.nullifier: collector.selection_times[record.nullifier] - burn
        for record in scored
    }
    selected_hash = hashlib.sha256(
        canonical_json(
            [
                {
                    "nullifier": record.nullifier,
                    "epoch": record.epoch,
                    "arrival": arrivals[record.nullifier],
                    "quality": record.quality,
                }
                for record in shifted
            ]
        )
    ).hexdigest()
    exposure_policy = registration["estimator"].get(
        "lifetime_exposure_policy", "causal_first_arrival_B7"
    )
    if exposure_policy == "causal_first_arrival_B7":
        effective, exposure = causal_lifetime_exposure_weights(
            shifted,
            arrivals,
            lag=lag,
            per_user_exposure_budget=float(
                registration["estimator"]["per_user_exposure_budget"]
            ),
        )
        exposure["allocation_policy"] = exposure_policy
    elif exposure_policy == "retained_v4_no_lifetime_reweighting":
        effective = shifted
        exposure = {
            "allocation_policy": exposure_policy,
            "input_records": len(shifted),
            "effective_records": len(shifted),
            "scaled_records": 0,
            "dropped_exhausted_records": 0,
            "dropped_unavailable_records": 0,
            "users_with_exposure": len({record.user_id for record in shifted}),
            "maximum_user_lifetime_exposure": 0.0,
            "total_lifetime_exposure": 0.0,
            "per_user_exposure_budget": 0.0,
            "lag": lag,
        }
    else:
        raise ValueError("unknown lifetime exposure policy")
    base = _estimator(registration)
    calibration = json.loads(
        (ROOT / registration["sources"]["innovation_calibration"]).read_text(
            encoding="utf-8"
        )
    )
    scale = float(calibration["innovation_scales"]["world_cross_fitted_covariance"])
    mechanism = world.physical_mechanism
    if mechanism is None or len(mechanism.exogenous_forcing) < steps + lag:
        raise ValueError("registered physical mechanism is unavailable")
    center = public[burn : steps + lag]
    operator = mechanism.transition
    forcing = public_residual_forcing(
        center,
        [operator] * len(center),
        mechanism.exogenous_forcing[burn : steps + lag],
        previous_public=public[burn - 1],
    )
    ap_method = registration["estimator"]["method"]
    methods = (ap_method, "PUBLIC", "SQ", "HUBER")
    prediction_manifests = []
    for method in methods:
        records = shifted if method == "PUBLIC" else effective
        runner_method = "CANDIDATE" if method == ap_method else method
        live, reconstructed, diagnostics = _predict_one(
            center,
            records,
            arrivals,
            operator,
            forcing,
            scale,
            base,
            runner_method,
        )
        arm = target / method
        arm.mkdir(parents=True, exist_ok=True)
        prediction_path = arm / "prediction.npz"
        np.savez_compressed(prediction_path, live=live, reconstructed=reconstructed)
        identity = _prediction_identity(
            source_lock_sha,
            seed,
            cell,
            method,
            trace.trace_hash,
            selected_hash,
            base,
        )
        epoch_terms = diagnostics.get("epoch_terms", [])
        manifest = {
            "identity": identity,
            "seed": seed,
            "cell": cell,
            "method": method,
            "prediction_sha256": sha256_file(prediction_path),
            "trace_hash": trace.trace_hash,
            "selected_hash": selected_hash,
            "scoring_loaded_during_prediction": False,
            "solver_failure_rate": float(diagnostics["solver_failure_rate"]),
            "projected_gradient_inf": max(
                (float(row["projected_gradient_inf"]) for row in epoch_terms), default=0.0
            ),
            "maximum_correction": float(diagnostics["maximum_correction"]),
            "lifetime_exposure_matched": method != "PUBLIC",
        }
        write_json(arm / "prediction_manifest.json", manifest)
        prediction_manifests.append(manifest)

    truth = world.truth[burn:steps]
    threshold = float(np.quantile(world.truth[:burn], 0.95))
    events = truth >= threshold
    method_rows = []
    for manifest in prediction_manifests:
        arm = target / manifest["method"]
        with np.load(arm / "prediction.npz", allow_pickle=False) as arrays:
            live = arrays["live"][: len(truth)]
            reconstructed = arrays["reconstructed"][: len(truth)]
        metrics = prediction_metrics(truth, reconstructed, world.cell_groups)
        live_metrics = prediction_metrics(truth, live, world.cell_groups)
        method_rows.append(
            {
                "method": manifest["method"],
                **metrics,
                **{f"live_{key}": value for key, value in live_metrics.items()},
                "event_recall": float(np.mean(live[events] >= threshold)),
                "event_support": int(events.sum()),
                "event_threshold": threshold,
                "prediction_identity": manifest["identity"],
                "prediction_sha256": manifest["prediction_sha256"],
                "solver_failure_rate": manifest["solver_failure_rate"],
                "projected_gradient_inf": manifest["projected_gradient_inf"],
                "maximum_correction": manifest["maximum_correction"],
            }
        )
    release_summary, release_arrays = reviewer_release_evidence(
        world, public[:steps], dict(transport.release_arrivals), config
    )
    np.savez_compressed(target / "protected_releases.npz", **release_arrays)
    allocations = [
        row for row in collector.allocations if burn <= int(row["epoch"]) < steps
    ]
    feasible_floor_violations = sum(
        bool(row["feasible"]) and not bool(row.get("constraint_satisfied", True))
        for row in allocations
    )
    counts = np.asarray(
        [
            [int(row["counts"].get(group, 0)) for group in range(config["world"]["groups"])]
            for row in allocations
        ],
        dtype=int,
    )
    zero = np.zeros(config["world"]["groups"], dtype=int)
    worst_zero_streak = 0
    for values in counts:
        zero = np.where(values == 0, zero + 1, 0)
        worst_zero_streak = max(worst_zero_streak, int(zero.max(initial=0)))
    fairness = {
        **coverage_metrics(scored, int(config["world"]["groups"])),
        "allocation_policy": registration["execution"]["allocation_policy"],
        "scored_epochs": len(allocations),
        "globally_feasible_epochs": sum(bool(row["feasible"]) for row in allocations),
        "feasible_floor_violations": feasible_floor_violations,
        "mean_max_avoidable_deficit": float(
            np.mean([max(row["avoidable"].values(), default=0.0) for row in allocations])
        ),
        "mean_max_unavoidable_deficit": float(
            np.mean([max(row["unavoidable"].values(), default=0.0) for row in allocations])
        ),
        "worst_zero_service_streak": worst_zero_streak,
    }
    transport_metrics = _finite_transport_metrics(transport.metrics)
    result = {
        "schema_version": 1,
        "status": "complete",
        "role": registration["role"],
        "source_lock_sha256": source_lock_sha,
        "seed": seed,
        "cell": cell,
        "config_hash": config_hash(config),
        "trace_hash": trace.trace_hash,
        "selected_hash": selected_hash,
        "observation_count": len(world.observations),
        "selected_count": len(scored),
        "method_rows": method_rows,
        "fairness": fairness,
        "lifetime_exposure": exposure,
        "privacy_release": release_summary,
        "protected_release_sha256": sha256_file(target / "protected_releases.npz"),
        "transport": transport_metrics,
        "public_provenance": public_provenance,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_bytes": int(getattr(process.memory_info(), "peak_wset", process.memory_info().rss)),
        },
    }
    write_json(target / "allocations.json", {"rows": allocations})
    write_json(result_path, result)
    return result


def _bootstrap_mean(values: np.ndarray, *, seed: int, replicates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    return values[indices].mean(axis=1)


def _analyze(registration: dict, results: list[dict], output: Path) -> dict:
    indexed = {(row["seed"], row["cell"]): row for row in results}
    seeds = registration["worlds"]["seeds"]
    paired = []
    ap_method = registration["estimator"]["method"]
    for seed in seeds:
        clean = indexed[(seed, "severe_clean")]
        attack = indexed[(seed, "severe_hotspot")]
        clean_methods = {row["method"]: row for row in clean["method_rows"]}
        attack_methods = {row["method"]: row for row in attack["method_rows"]}
        growth_ap = attack_methods[ap_method]["rmse"] - clean_methods[ap_method]["rmse"]
        growth_sq = attack_methods["SQ"]["rmse"] - clean_methods["SQ"]["rmse"]
        paired.append(
            {
                "seed": seed,
                "trace_hash_clean": clean["trace_hash"],
                "trace_hash_attack": attack["trace_hash"],
                "trace_match": clean["trace_hash"] == attack["trace_hash"],
                "clean_ap_rmse": clean_methods[ap_method]["rmse"],
                "clean_sq_rmse": clean_methods["SQ"]["rmse"],
                "clean_public_rmse": clean_methods["PUBLIC"]["rmse"],
                "clean_noninferiority_contrast": (
                    clean_methods[ap_method]["rmse"]
                    - 1.05 * clean_methods["SQ"]["rmse"]
                ),
                "citizen_value_contrast": (
                    clean_methods[ap_method]["rmse"]
                    - clean_methods["PUBLIC"]["rmse"]
                ),
                "event_recall_loss_vs_sq": (
                    clean_methods["SQ"]["event_recall"]
                    - clean_methods[ap_method]["event_recall"]
                ),
                "hotspot_growth_ap": growth_ap,
                "hotspot_growth_sq": growth_sq,
                "hotspot_attenuation": (
                    1.0 - growth_ap / growth_sq if growth_sq > 0 else None
                ),
                "timely_raw_arrivals": clean["transport"]["raw_timely_delivered"],
                "selected_count": clean["selected_count"],
                "release_rmse": clean["privacy_release"]["residual"]["rmse"],
                "release_nonnegative_raw_rmse": clean["privacy_release"][
                    "raw_nonnegative"
                ]["rmse"],
                "release_public_rmse": clean["privacy_release"]["residual"][
                    "same_support_public_rmse"
                ],
                "receipt_issued": clean["transport"]["receipt_issued"],
                "receipt_verified": clean["transport"]["audit_verified_return_pairs"],
                "publicly_checkpointed": clean["transport"]["audit_publicly_checkpointed"],
            }
        )
    (output / "paired_worlds.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in paired),
        encoding="utf-8",
    )
    alpha = 0.05 / 4
    reps = int(registration["evaluation"]["bootstrap_replicates"])
    seed = int(registration["evaluation"]["bootstrap_seed"])
    contrasts = {
        "clean_noninferiority": np.asarray(
            [row["clean_noninferiority_contrast"] for row in paired], dtype=float
        ),
        "citizen_value": np.asarray(
            [row["citizen_value_contrast"] for row in paired], dtype=float
        ),
        "event_recall_loss": np.asarray(
            [row["event_recall_loss_vs_sq"] for row in paired], dtype=float
        ),
        "hotspot_attenuation": np.asarray(
            [row["hotspot_attenuation"] for row in paired], dtype=float
        ),
    }
    bounds = {}
    for offset, (name, values) in enumerate(contrasts.items()):
        if not np.isfinite(values).all():
            bounds[name] = {"valid": False, "reason": "nonfinite paired contrast"}
            continue
        bootstrap = _bootstrap_mean(values, seed=seed + offset, replicates=reps)
        bounds[name] = {
            "valid": True,
            "worlds": len(values),
            "mean": float(values.mean()),
            "lower_simultaneous": float(np.quantile(bootstrap, alpha)),
            "upper_simultaneous": float(np.quantile(bootstrap, 1 - alpha)),
        }
    rules = registration["decision_rules"]
    gates = {
        "clean_noninferiority": (
            bounds["clean_noninferiority"].get("valid") is True
            and bounds["clean_noninferiority"]["upper_simultaneous"] <= 0
        ),
        "citizen_value": (
            bounds["citizen_value"].get("valid") is True
            and bounds["citizen_value"]["upper_simultaneous"] <= 0
        ),
        "event_recall": (
            bounds["event_recall_loss"].get("valid") is True
            and bounds["event_recall_loss"]["upper_simultaneous"]
            <= rules["event_recall_loss_vs_SQ_upper_percentage_points"] / 100
        ),
        "hotspot_attenuation": (
            bounds["hotspot_attenuation"].get("valid") is True
            and bounds["hotspot_attenuation"]["lower_simultaneous"]
            >= rules["hotspot_growth_attenuation_lower"]
        ),
    }
    invariant_failures = []
    for row in results:
        metrics = row["transport"]
        checks = {
            "trace_pair": next(item for item in paired if item["seed"] == row["seed"])[
                "trace_match"
            ],
            "contact_capacity": metrics["max_contact_direction_bytes"]
            <= registration["execution"]["contact_capacity_bytes_per_direction"],
            "control_capacity": metrics["max_control_direction_bytes"]
            <= registration["execution"]["control_budget_bytes_per_direction"],
            "raw_buffer": metrics["max_raw_buffer_bytes"]
            <= registration["execution"]["raw_buffer_bytes_per_node"],
            "release_buffer": metrics["max_release_buffer_bytes"]
            <= registration["execution"]["release_buffer_bytes_per_node"],
            "copy_tokens": metrics["token_violations"] == 0,
            "floor": row["fairness"]["feasible_floor_violations"] == 0,
            "privacy": row["privacy_release"]["epsilon_history"]
            <= registration["privacy"]["epsilon_history"] + 1e-12
            and row["privacy_release"]["privacy_budget_violations"] == 0,
            "lifetime": (
                row["lifetime_exposure"]["maximum_user_lifetime_exposure"]
                <= float(registration["estimator"].get("per_user_exposure_budget", 0.0))
                + 1e-10
            ),
            "solver": all(
                method["solver_failure_rate"] == 0 for method in row["method_rows"]
            ),
        }
        checkpoint_cfg = registration.get("accountability", {}).get(
            "public_checkpoint", {}
        )
        if registration.get("accountability", {}).get(
            "public_checkpoint_in_common_execution", False
        ):
            checks.update(
                public_checkpoint_complete=(
                    metrics["audit_publicly_checkpointed"]
                    == metrics["receipt_issued"]
                    and metrics["audit_public_checkpoint_heads_verified"] >= 1
                ),
                public_checkpoint_queue=(
                    metrics["audit_public_checkpoint_peak_buffer_bytes"]
                    <= int(checkpoint_cfg["queue_buffer_bytes"])
                    and metrics["audit_public_checkpoint_reassembly_peak_bytes"]
                    <= int(checkpoint_cfg["auditor_reassembly_buffer_bytes"])
                ),
                public_checkpoint_integrity=(
                    metrics["audit_public_checkpoint_signature_failures"] == 0
                    and metrics["audit_public_checkpoint_consistency_failures"] == 0
                ),
            )
        for check, passed in checks.items():
            if not passed:
                invariant_failures.append(
                    {"seed": row["seed"], "cell": row["cell"], "check": check}
                )
    descriptive = {
        "timely_raw_arrivals_mean": float(
            np.mean([row["timely_raw_arrivals"] for row in paired])
        ),
        "selected_count_mean": float(np.mean([row["selected_count"] for row in paired])),
        "release_rmse_mean": float(np.mean([row["release_rmse"] for row in paired])),
        "release_nonnegative_raw_rmse_mean": float(
            np.mean([row["release_nonnegative_raw_rmse"] for row in paired])
        ),
        "release_public_rmse_mean": float(
            np.mean([row["release_public_rmse"] for row in paired])
        ),
        "receipt_issued_mean": float(np.mean([row["receipt_issued"] for row in paired])),
        "receipt_verified_mean": float(
            np.mean([row["receipt_verified"] for row in paired])
        ),
        "publicly_checkpointed_mean": float(
            np.mean([row["publicly_checkpointed"] for row in paired])
        ),
    }
    analysis = {
        "schema_version": 1,
        "role": registration["role"],
        "worlds": len(seeds),
        "jobs": len(results),
        "paired_bounds": bounds,
        "efficacy_gates": gates,
        "all_efficacy_gates_pass": all(gates.values()),
        "invariant_failures": invariant_failures,
        "all_execution_invariants_pass": not invariant_failures,
        "descriptive_common_workload": descriptive,
        "claim_authorized": all(gates.values()) and not invariant_failures,
    }
    write_json(output / "analysis.json", analysis)
    return analysis


def _prepare_output(
    registration: dict, output: Path, registration_path: Path = REGISTRATION
) -> str:
    inventory = _source_inventory(registration, registration_path)
    registration_sha = sha256_file(ROOT / registration_path)
    source_identity = hashlib.sha256(canonical_json(inventory)).hexdigest()
    lock_path = output / "source_lock.json"
    if output.exists():
        if not lock_path.is_file():
            raise FileExistsError("existing output lacks source lock")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            lock.get("registration_sha256") != registration_sha
            or lock.get("source_inventory") != inventory
            or lock.get("outcomes_before_lock") != 0
        ):
            raise ValueError("shared-resource source lock mismatch")
        return sha256_file(lock_path)
    output.mkdir(parents=True)
    lock = {
        "schema_version": 1,
        "role": "immutable source/config lock created before registered outcomes",
        "registration_sha256": registration_sha,
        "source_identity": source_identity,
        "source_inventory": inventory,
        "outcomes_before_lock": 0,
    }
    write_json(lock_path, lock)
    snapshot = output / "source_snapshot"
    for relative in inventory:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "registration": registration["registration"],
            "registration_sha256": registration_sha,
            "source_lock_sha256": sha256_file(lock_path),
            "expected_jobs": len(registration["worlds"]["seeds"])
            * len(registration["worlds"]["cells"]),
            "expected_method_rows": len(registration["worlds"]["seeds"])
            * len(registration["worlds"]["cells"])
            * 4,
            "large_prediction_policy": "stored once and content-addressed; compact artifact indexes hashes",
        },
    )
    return sha256_file(lock_path)


def execute(
    *,
    output: Path,
    smoke: bool,
    workers: int,
    registration_path: Path = REGISTRATION,
) -> dict:
    registration = _load_registration(registration_path)
    if smoke:
        seeds = [6250999]
        cells = ["severe_clean"]
    else:
        seeds = registration["worlds"]["seeds"]
        cells = registration["worlds"]["cells"]
    if not 1 <= workers <= 4:
        raise ValueError("workers must lie in [1,4]")
    source_lock_sha = _prepare_output(registration, output, registration_path)
    specs = [
        (str(output), registration, int(seed), cell, source_lock_sha)
        for seed in seeds
        for cell in cells
    ]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        futures = [pool.submit(_job, spec) for spec in specs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(
                output / "progress.json",
                {
                    "completed_jobs": len(results),
                    "expected_jobs": len(specs),
                    "failures": 0,
                    "smoke": smoke,
                },
            )
    if smoke:
        summary = {
            "status": "complete",
            "role": "engineering smoke outside registered confirmation seeds",
            "jobs": len(results),
            "wall_seconds": time.perf_counter() - started,
            "result": results[0],
        }
        write_json(output / "smoke_summary.json", summary)
        return summary
    analysis = _analyze(registration, results, output)
    completion = {
        "status": "complete",
        "jobs": len(results),
        "method_rows": sum(len(row["method_rows"]) for row in results),
        "source_lock_sha256": source_lock_sha,
        "wall_seconds": time.perf_counter() - started,
        "analysis_sha256": sha256_file(output / "analysis.json"),
        "claim_authorized": analysis["claim_authorized"],
    }
    write_json(output / "completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    registration = _load_registration(args.registration)
    if args.output is None:
        if args.smoke:
            output = ROOT / "reports" / "v7" / "reviewer_revision" / "shared_resource_smoke_v2"
        else:
            output = ROOT / registration["artifacts"]["directory"]
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
    result = execute(
        output=output,
        smoke=args.smoke,
        workers=args.workers,
        registration_path=args.registration,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

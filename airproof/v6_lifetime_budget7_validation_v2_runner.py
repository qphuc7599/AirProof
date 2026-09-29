"""Fail-closed runner for independent five-cell lifetime-estimator validation."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .metrics import prediction_metrics
from .records import canonical_json
from .v6_covariance_forcing_inputs import sha256_file, write_json
from .v6_covariance_forcing_runner import _predict_one, information_fingerprint
from .v6_estimator import EstimatorConfig
from .v6_lifetime_exposure import causal_lifetime_exposure_weights
from .v6_lifetime_budget7_validation_v2_inputs import (
    load_prediction_view,
    load_scoring_view,
    verify_bundle,
)
from .v6_lifetime_budget7_validation_v2_registration import assert_registration

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = "configs/v6/lifetime_budget7_validation_v2.json"


def planned_jobs(registration: Mapping[str, Any], *, smoke: bool) -> list[tuple[int, str]]:
    """Return the exact registered job matrix without opening any input capability."""
    assert_registration(registration)
    seeds = (registration["runtime"]["runtime_smoke_seeds"] if smoke
             else registration["worlds"]["seeds"])
    jobs = [(int(seed), cell) for seed in seeds for cell in registration["worlds"]["cells"]]
    expected = len(seeds) * len(registration["worlds"]["cells"])
    if len(jobs) != expected or len(set(jobs)) != expected:
        raise ValueError("validation job matrix is incomplete or duplicated")
    if not smoke and expected * len(registration["methods"]) != 240:
        raise ValueError("full validation must contain exactly 240 method rows")
    return jobs


def load_registration(root: str | Path = ROOT) -> dict[str, Any]:
    value = json.loads((Path(root) / REGISTRATION).read_text(encoding="utf-8"))
    assert_registration(value)
    return value


def fixed_estimator(registration: Mapping[str, Any]) -> EstimatorConfig:
    fixed = registration["fixed_estimator"]
    return EstimatorConfig(
        cap=float(fixed["cap"]), lag=int(fixed["lag"]),
        huber_delta=float(fixed["huber_delta"]),
        lambda_temporal=float(fixed["lambda_temporal"]),
        lambda_spatial=float(fixed["lambda_spatial"]),
        lambda_zero=float(fixed["lambda_zero"]),
        time_step=float(fixed["time_step"]), tolerance=float(fixed["tolerance"]),
        max_iterations=int(fixed["max_iterations"]),
        numerical_refinements=int(fixed["numerical_refinements"]),
        loss="huber", output_cap=True, input_clip=False,
        history_normalized_huber=False,
    )


def verify_source_lock(root: str | Path, registration: Mapping[str, Any]) -> dict:
    root = Path(root)
    path = root / registration["input_producer"]["source_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (lock.get("registration_sha256") != sha256_file(root / REGISTRATION)
            or lock.get("validation_outcomes_before_lock") != 0):
        raise ValueError("validation source lock does not bind zero outcomes")
    mismatches = [relative for relative, expected in lock["source_sha256"].items()
                  if not (root / relative).is_file()
                  or sha256_file(root / relative) != expected]
    if mismatches:
        raise ValueError(f"validation source lock mismatches: {mismatches}")
    development = root / registration["nomination"]["development_analysis"]
    if sha256_file(development) != lock["development_analysis_sha256"]:
        raise ValueError("development nomination changed after validation lock")
    calibration = root / registration["fixed_estimator"]["innovation_calibration"]
    if sha256_file(calibration) != lock["innovation_calibration_sha256"]:
        raise ValueError("innovation calibration changed after validation lock")
    seed_registry = root / registration["worlds"]["seed_registry"]
    if sha256_file(seed_registry) != lock["seed_registry_sha256"]:
        raise ValueError("validation seed registry changed after source lock")
    return lock


def _identity(source_lock_sha: str, input_lock_sha: str, seed: int, cell: str,
              method: str, fingerprint: str, estimator: EstimatorConfig,
              budget: float) -> str:
    return hashlib.sha256(canonical_json({
        "source_lock_sha256": source_lock_sha,
        "input_lock_sha256": input_lock_sha,
        "seed": seed, "cell": cell, "method": method,
        "information_fingerprint": fingerprint,
        "estimator": asdict(estimator), "lifetime_budget": budget,
    })).hexdigest()


def _scoring_path(bundle: Path, seed: int, cell: str) -> Path:
    base = "severe_clean" if cell in ("severe_drift", "severe_hotspot") else cell
    return bundle / "scoring" / str(seed) / f"scoring_{base}.npz"


def _job(spec: tuple[str, dict[str, Any], int, str, str, str]) -> dict[str, Any]:
    root_raw, registration, seed, cell, source_lock_sha, input_lock_sha = spec
    root = Path(root_raw)
    bundle = root / registration["input_producer"]["bundle"]
    input_lock = json.loads((root / registration["input_producer"]["input_lock"])
                            .read_text(encoding="utf-8"))
    view = load_prediction_view(root, bundle, seed, cell,
                                expected_manifest_sha256=input_lock["manifest_sha256"])
    calibration = json.loads((bundle / "global_calibration.json")
                             .read_text(encoding="utf-8"))
    covariance = registration["fixed_estimator"]["innovation_covariance"]
    scale = float(calibration["innovation_scales"][covariance])
    variance = float(calibration["innovation_variances"][covariance])
    forcing = view["residual_forcing"]
    fingerprint = information_fingerprint(
        view, innovation_variance=variance, residual_forcing=forcing)
    summary = json.loads((bundle / "prediction" / str(seed) / "summary.json")
                         .read_text(encoding="utf-8"))
    resource_invariants = summary["cell_summary"][cell]["resource_invariants_pass"] is True
    target = root / registration["artifacts"]["outcomes"] / "jobs" / str(seed) / cell
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / "result.json"
    methods = registration["methods"]
    pairs = [((target / method / "prediction.npz").is_file(),
              (target / method / "prediction_manifest.json").is_file())
             for method in methods]
    if any(prediction != manifest for prediction, manifest in pairs):
        raise ValueError("one-sided validation resume artifact")
    if result_path.is_file() and not all(prediction and manifest
                                         for prediction, manifest in pairs):
        raise ValueError("validation result exists without all predictions")
    estimator = fixed_estimator(registration)
    budget = float(registration["fixed_estimator"]["per_user_exposure_budget"])
    manifests = []
    process = psutil.Process()
    started, cpu_started_job = time.perf_counter(), time.process_time()
    peak_rss = process.memory_info().rss
    for method in methods:
        arm = target / method
        arm.mkdir(parents=True, exist_ok=True)
        prediction_path = arm / "prediction.npz"
        manifest_path = arm / "prediction_manifest.json"
        identity = _identity(source_lock_sha, input_lock_sha, seed, cell, method,
                             fingerprint, estimator, budget)
        if prediction_path.is_file() and manifest_path.is_file():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (prior.get("identity") != identity
                    or prior.get("prediction_sha256") != sha256_file(prediction_path)):
                raise ValueError("stale or tampered validation prediction")
            manifests.append(prior)
            continue
        arm_started, cpu_started = time.perf_counter(), time.process_time()
        exposure = {"maximum_user_lifetime_exposure": 0.0,
                    "effective_records": len(view["observations"]),
                    "scaled_records": 0, "dropped_exhausted_records": 0,
                    "total_lifetime_exposure": 0.0}
        records = view["observations"]
        runner_method = method
        if method == "AP_LIFETIME7":
            records, exposure = causal_lifetime_exposure_weights(
                records, view["arrival_map"], lag=estimator.lag,
                per_user_exposure_budget=budget)
            runner_method = "CANDIDATE"
        live, reconstructed, diagnostics = _predict_one(
            view["public_center"], records, view["arrival_map"], view["operator"],
            forcing, scale, estimator, runner_method)
        if (not np.isfinite(live).all() or not np.isfinite(reconstructed).all()
                or len(live) != len(view["public_center"])):
            raise FloatingPointError("nonfinite or incomplete validation prediction")
        np.savez_compressed(prediction_path, live=live, reconstructed=reconstructed)
        maximum_kkt = max((float(row["projected_gradient_inf"])
                           for row in diagnostics.get("epoch_terms", [])), default=0.0)
        manifest = {
            "schema_version": 1, "role": "validation prediction before scoring",
            "identity": identity, "seed": seed, "cell": cell, "method": method,
            "information_fingerprint": fingerprint,
            "innovation_variance": variance, "innovation_scale": scale,
            "residual_forcing": "producer_exact_causal",
            "prediction_sha256": sha256_file(prediction_path),
            "scoring_loaded_during_prediction": False,
            "solver_failure_rate": float(diagnostics["solver_failure_rate"]),
            "projected_gradient_inf": maximum_kkt,
            "maximum_absolute_correction": float(diagnostics["maximum_correction"]),
            "maximum_user_lifetime_exposure": exposure[
                "maximum_user_lifetime_exposure"],
            "effective_records": exposure["effective_records"],
            "scaled_records": exposure["scaled_records"],
            "dropped_exhausted_records": exposure["dropped_exhausted_records"],
            "total_lifetime_exposure": exposure["total_lifetime_exposure"],
            "wall_seconds": time.perf_counter() - arm_started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_bytes": int(process.memory_info().rss),
        }
        write_json(manifest_path, manifest)
        manifests.append(manifest)
        peak_rss = max(peak_rss, process.memory_info().rss)
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        prior_rows = {row.get("method"): row for row in prior.get("rows", [])}
        if (prior.get("status") != "complete" or prior.get("seed") != seed
                or prior.get("cell") != cell
                or prior.get("source_lock_sha256") != source_lock_sha
                or prior.get("input_lock_sha256") != input_lock_sha
                or prior.get("row_count") != len(methods) or set(prior_rows) != set(methods)
                or any(prior_rows[row["method"]].get("prediction_identity")
                       != row["identity"] or prior_rows[row["method"]].get(
                           "prediction_sha256") != row["prediction_sha256"]
                       for row in manifests)):
            raise ValueError("stale or tampered validation result")
        return prior
    scoring = load_scoring_view(bundle, seed, cell)
    truth, events = scoring["evaluation_truth"], scoring["event_mask"]
    threshold, groups = float(scoring["event_threshold"]), scoring["cell_groups"]
    scoring_path = _scoring_path(bundle, seed, cell)
    rows = []
    for manifest in manifests:
        with np.load(target / manifest["method"] / "prediction.npz",
                     allow_pickle=False) as arrays:
            live = arrays["live"][:len(truth)]
            reconstructed = arrays["reconstructed"][:len(truth)]
        metrics = prediction_metrics(truth, reconstructed, groups)
        live_metrics = prediction_metrics(truth, live, groups)
        rows.append({
            "schema_version": 1, "seed": seed, "scenario": cell,
            "method": manifest["method"], "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "worst_group_rmse": float(metrics["worst_group_rmse"]),
            "live_rmse": float(live_metrics["rmse"]),
            "event_recall": float(np.mean(live[events] >= threshold)),
            "event_support": int(events.sum()), "event_threshold": threshold,
            "prediction_horizon_half_open": [
                int(view["prediction_epochs"][0]),
                int(view["prediction_epochs"][-1]) + 1],
            "scoring_horizon_half_open": [
                int(view["prediction_epochs"][0]),
                int(view["prediction_epochs"][0]) + len(truth)],
            "rmse_clock": "fixed_lag_reconstructed_after_registered_lag",
            "live_rmse_clock": "immutable_prediction_at_epoch",
            "event_recall_clock": "immutable_live_prediction_at_epoch",
            "information_fingerprint": fingerprint,
            "solver_failure_rate": manifest["solver_failure_rate"],
            "projected_gradient_inf": manifest["projected_gradient_inf"],
            "maximum_absolute_correction": manifest["maximum_absolute_correction"],
            "maximum_user_lifetime_exposure": manifest[
                "maximum_user_lifetime_exposure"],
            "wall_seconds": manifest["wall_seconds"],
            "process_cpu_seconds": manifest["process_cpu_seconds"],
            "peak_rss_bytes": manifest["peak_rss_bytes"],
            "prediction_identity": manifest["identity"],
            "prediction_sha256": manifest["prediction_sha256"],
            "scoring_sha256": sha256_file(scoring_path),
            "resource_invariants_pass": resource_invariants,
            "finite_outputs": True,
        })
    result = {
        "schema_version": 1, "status": "complete", "seed": seed, "cell": cell,
        "source_lock_sha256": source_lock_sha, "input_lock_sha256": input_lock_sha,
        "rows": rows, "row_count": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started_job,
        "peak_rss_bytes": int(peak_rss),
    }
    write_json(result_path, result)
    return result


def execute(*, smoke: bool) -> dict[str, Any]:
    registration = load_registration(ROOT)
    verify_source_lock(ROOT, registration)
    source_lock = ROOT / registration["input_producer"]["source_lock"]
    input_lock_path = ROOT / registration["input_producer"]["input_lock"]
    input_lock = json.loads(input_lock_path.read_text(encoding="utf-8"))
    source_lock_sha, input_lock_sha = sha256_file(source_lock), sha256_file(input_lock_path)
    if (input_lock.get("source_lock_sha256") != source_lock_sha
            or input_lock.get("registration_sha256") != sha256_file(ROOT / REGISTRATION)
            or input_lock.get("scientific_outcomes_before_lock") != 0):
        raise RuntimeError("validation input lock mismatch")
    bundle = ROOT / registration["input_producer"]["bundle"]
    verify_bundle(ROOT, bundle, expected_manifest_sha256=input_lock["manifest_sha256"])
    readiness = json.loads((ROOT / registration["input_producer"]["readiness"])
                           .read_text(encoding="utf-8"))
    if (readiness.get("pass") is not True
            or readiness.get("source_lock_sha256") != source_lock_sha
            or readiness.get("input_lock_sha256") != input_lock_sha):
        raise RuntimeError("validation readiness mismatch")
    keys = planned_jobs(registration, smoke=smoke)
    specs = [(str(ROOT), registration, seed, cell, source_lock_sha, input_lock_sha)
             for seed, cell in keys]
    output = ROOT / registration["artifacts"]["outcomes"]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(registration["runtime"]["workers"],
                                             len(specs))) as pool:
        futures = [pool.submit(_job, spec) for spec in specs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(output / "progress.json", {
                "complete_jobs": len(results), "expected_jobs": len(specs),
                "failures": 0, "runtime_smoke": smoke,
            })
    summary = {
        "status": "complete", "runtime_smoke": smoke,
        "complete_jobs": len(results), "expected_jobs": len(specs),
        "complete_rows": sum(row["row_count"] for row in results),
        "wall_seconds": time.perf_counter() - started,
        "source_lock_sha256": source_lock_sha, "input_lock_sha256": input_lock_sha,
    }
    write_json(output / ("runtime_smoke.json" if smoke else "execution.json"), summary)
    return summary

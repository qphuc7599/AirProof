"""Fail-closed candidate-only runner for lifetime-budget repair v5."""
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
from .v6_lifetime_budget_repair_registration import assert_registration
from .v6_lifetime_exposure import causal_lifetime_exposure_weights
from .v6_lifetime_validation_inputs import load_prediction_view, load_scoring_view, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = "configs/v6/lifetime_budget_repair_development_v5.json"


def load_registration(root: str | Path = ROOT) -> dict[str, Any]:
    value = json.loads((Path(root) / REGISTRATION).read_text(encoding="utf-8"))
    assert_registration(value)
    return value


def planned_jobs(registration: Mapping[str, Any], *, smoke: bool) -> list[tuple[int, str]]:
    assert_registration(registration)
    seeds = (registration["runtime"]["runtime_smoke_seeds"] if smoke
             else registration["input_reuse"]["seeds"])
    jobs = [(int(seed), cell) for seed in seeds for cell in registration["input_reuse"]["cells"]]
    if len(jobs) != len(seeds) * 5 or len(set(jobs)) != len(jobs):
        raise ValueError("repair job matrix changed")
    return jobs


def fixed_estimator(registration: Mapping[str, Any]) -> EstimatorConfig:
    fixed = registration["fixed_estimator"]
    return EstimatorConfig(
        cap=float(fixed["cap"]), lag=int(fixed["lag"]),
        huber_delta=float(fixed["huber_delta"]),
        lambda_temporal=float(fixed["lambda_temporal"]),
        lambda_spatial=float(fixed["lambda_spatial"]),
        lambda_zero=float(fixed["lambda_zero"]), time_step=float(fixed["time_step"]),
        tolerance=float(fixed["tolerance"]), max_iterations=int(fixed["max_iterations"]),
        numerical_refinements=int(fixed["numerical_refinements"]), loss="huber",
        output_cap=True, input_clip=False, history_normalized_huber=False)


def verify_source_lock(root: str | Path, registration: Mapping[str, Any]) -> dict:
    root = Path(root)
    path = root / registration["artifacts"]["source_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (lock.get("registration_sha256") != sha256_file(root / REGISTRATION)
            or lock.get("candidate_outcomes_before_lock") != 0
            or lock.get("controls_recomputed") is not False):
        raise ValueError("repair source lock identity changed")
    mismatches = [relative for relative, expected in lock["source_sha256"].items()
                  if not (root / relative).is_file() or sha256_file(root / relative) != expected]
    if mismatches:
        raise ValueError(f"repair source lock mismatches: {mismatches}")
    for relative, expected in lock["upstream_sha256"].items():
        if not (root / relative).is_file() or sha256_file(root / relative) != expected:
            raise ValueError(f"repair upstream changed: {relative}")
    return lock


def _identity(lock_sha: str, input_sha: str, seed: int, cell: str,
              candidate: Mapping[str, Any], fingerprint: str,
              estimator: EstimatorConfig) -> str:
    return hashlib.sha256(canonical_json({"source_lock_sha256": lock_sha,
        "input_lock_sha256": input_sha, "seed": seed, "cell": cell,
        "candidate": dict(candidate), "information_fingerprint": fingerprint,
        "estimator": asdict(estimator)})).hexdigest()


def _scoring_path(bundle: Path, seed: int, cell: str) -> Path:
    base = "severe_clean" if cell in ("severe_drift", "severe_hotspot") else cell
    return bundle / "scoring" / str(seed) / f"scoring_{base}.npz"


def _job(spec: tuple[str, dict[str, Any], int, str, str, str]) -> dict[str, Any]:
    root_raw, registration, seed, cell, lock_sha, input_sha = spec
    root = Path(root_raw)
    bundle = root / registration["input_reuse"]["bundle"]
    input_lock = json.loads((root / registration["input_reuse"]["input_lock"])
                            .read_text(encoding="utf-8"))
    view = load_prediction_view(root, bundle, seed, cell,
                                expected_manifest_sha256=input_lock["manifest_sha256"])
    calibration = json.loads((bundle / "global_calibration.json").read_text(encoding="utf-8"))
    covariance = registration["fixed_estimator"]["innovation_covariance"]
    scale = float(calibration["innovation_scales"][covariance])
    variance = float(calibration["innovation_variances"][covariance])
    fingerprint = information_fingerprint(view, innovation_variance=variance,
                                          residual_forcing=view["residual_forcing"])
    parent = root / registration["input_reuse"]["source_validation_outcomes"] / "jobs" / str(seed) / cell
    parent_result = json.loads((parent / "result.json").read_text(encoding="utf-8"))
    if (parent_result.get("status") != "complete"
            or {row.get("method") for row in parent_result.get("rows", [])}
            != set(registration["controls"]["reused_methods"])
            or {row.get("information_fingerprint") for row in parent_result["rows"]}
            != {fingerprint}):
        raise ValueError("reused validation controls or information fingerprint changed")
    resource_ok = all(row.get("resource_invariants_pass") is True
                      for row in parent_result["rows"])
    target = root / registration["artifacts"]["outcomes"] / "jobs" / str(seed) / cell
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / "result.json"
    candidates = registration["candidates"]
    pairs = [((target / c["id"] / "prediction.npz").is_file(),
              (target / c["id"] / "prediction_manifest.json").is_file()) for c in candidates]
    if any(a != b for a, b in pairs):
        raise ValueError("one-sided repair resume artifact")
    if result_path.is_file() and not all(a and b for a, b in pairs):
        raise ValueError("repair result exists without every candidate prediction")
    estimator = fixed_estimator(registration)
    process = psutil.Process()
    started, cpu_started = time.perf_counter(), time.process_time()
    peak = process.memory_info().rss
    manifests = []
    for candidate in candidates:
        arm = target / candidate["id"]
        arm.mkdir(parents=True, exist_ok=True)
        prediction_path, manifest_path = arm / "prediction.npz", arm / "prediction_manifest.json"
        identity = _identity(lock_sha, input_sha, seed, cell, candidate, fingerprint, estimator)
        if prediction_path.is_file() and manifest_path.is_file():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior.get("identity") != identity or prior.get("prediction_sha256") != sha256_file(prediction_path):
                raise ValueError("stale or tampered repair prediction")
            manifests.append(prior)
            continue
        arm_started, arm_cpu = time.perf_counter(), time.process_time()
        records, exposure = causal_lifetime_exposure_weights(
            view["observations"], view["arrival_map"], lag=estimator.lag,
            per_user_exposure_budget=float(candidate["per_user_exposure_budget"]))
        live, reconstructed, diagnostics = _predict_one(
            view["public_center"], records, view["arrival_map"], view["operator"],
            view["residual_forcing"], scale, estimator, "CANDIDATE")
        if not np.isfinite(live).all() or not np.isfinite(reconstructed).all():
            raise FloatingPointError("nonfinite repair prediction")
        np.savez_compressed(prediction_path, live=live, reconstructed=reconstructed)
        maximum_kkt = max((float(row["projected_gradient_inf"])
                           for row in diagnostics.get("epoch_terms", [])), default=0.0)
        manifest = {"schema_version": 1, "role": "exposed repair candidate prediction before scoring",
            "identity": identity, "seed": seed, "cell": cell,
            "candidate": candidate["id"], "per_user_exposure_budget": candidate["per_user_exposure_budget"],
            "information_fingerprint": fingerprint, "innovation_variance": variance,
            "innovation_scale": scale, "residual_forcing": "producer_exact_causal",
            "prediction_sha256": sha256_file(prediction_path),
            "scoring_loaded_during_prediction": False,
            "solver_failure_rate": float(diagnostics["solver_failure_rate"]),
            "projected_gradient_inf": maximum_kkt,
            "maximum_absolute_correction": float(diagnostics["maximum_correction"]),
            **{key: exposure[key] for key in ("maximum_user_lifetime_exposure",
                "effective_records", "scaled_records", "dropped_exhausted_records",
                "total_lifetime_exposure")},
            "wall_seconds": time.perf_counter() - arm_started,
            "process_cpu_seconds": time.process_time() - arm_cpu,
            "peak_rss_bytes": int(process.memory_info().rss)}
        write_json(manifest_path, manifest)
        manifests.append(manifest)
        peak = max(peak, process.memory_info().rss)
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if (prior.get("status") != "complete" or prior.get("source_lock_sha256") != lock_sha
                or prior.get("input_lock_sha256") != input_sha or prior.get("row_count") != 2):
            raise ValueError("stale repair result")
        return prior
    scoring = load_scoring_view(bundle, seed, cell)
    truth, events = scoring["evaluation_truth"], scoring["event_mask"]
    threshold, groups = float(scoring["event_threshold"]), scoring["cell_groups"]
    rows = []
    for manifest in manifests:
        with np.load(target / manifest["candidate"] / "prediction.npz", allow_pickle=False) as arrays:
            live, reconstructed = arrays["live"][:len(truth)], arrays["reconstructed"][:len(truth)]
        metrics, live_metrics = prediction_metrics(truth, reconstructed, groups), prediction_metrics(truth, live, groups)
        rows.append({"schema_version": 1, "seed": seed, "scenario": cell,
            "candidate": manifest["candidate"], "method": "CANDIDATE",
            "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
            "worst_group_rmse": float(metrics["worst_group_rmse"]),
            "live_rmse": float(live_metrics["rmse"]),
            "event_recall": float(np.mean(live[events] >= threshold)),
            "event_support": int(events.sum()), "event_threshold": threshold,
            "information_fingerprint": fingerprint,
            **{key: manifest[key] for key in ("solver_failure_rate", "projected_gradient_inf",
                "maximum_absolute_correction", "maximum_user_lifetime_exposure",
                "wall_seconds", "process_cpu_seconds", "peak_rss_bytes")},
            "prediction_identity": manifest["identity"],
            "prediction_sha256": manifest["prediction_sha256"],
            "scoring_sha256": sha256_file(_scoring_path(bundle, seed, cell)),
            "resource_invariants_pass": resource_ok, "finite_outputs": True})
    result = {"schema_version": 1, "status": "complete", "seed": seed, "cell": cell,
        "source_lock_sha256": lock_sha, "input_lock_sha256": input_sha,
        "parent_control_result_sha256": sha256_file(parent / "result.json"),
        "controls_recomputed": False, "rows": rows, "row_count": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_bytes": int(peak)}
    write_json(result_path, result)
    return result


def execute(*, smoke: bool) -> dict[str, Any]:
    registration = load_registration(ROOT)
    verify_source_lock(ROOT, registration)
    source_path = ROOT / registration["artifacts"]["source_lock"]
    input_path = ROOT / registration["input_reuse"]["input_lock"]
    source_sha, input_sha = sha256_file(source_path), sha256_file(input_path)
    input_lock = json.loads(input_path.read_text(encoding="utf-8"))
    verify_bundle(ROOT, ROOT / registration["input_reuse"]["bundle"],
                  expected_manifest_sha256=input_lock["manifest_sha256"])
    readiness = json.loads((ROOT / registration["artifacts"]["readiness"]).read_text(encoding="utf-8"))
    if (readiness.get("pass") is not True or readiness.get("source_lock_sha256") != source_sha
            or readiness.get("input_lock_sha256") != input_sha):
        raise RuntimeError("repair readiness is absent, stale, or failed")
    jobs = planned_jobs(registration, smoke=smoke)
    specs = [(str(ROOT), registration, seed, cell, source_sha, input_sha) for seed, cell in jobs]
    output = ROOT / registration["artifacts"]["outcomes"]
    output.mkdir(parents=True, exist_ok=True)
    results, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(registration["runtime"]["workers"], len(specs))) as pool:
        futures = [pool.submit(_job, spec) for spec in specs]
        for future in as_completed(futures):
            results.append(future.result())
            write_json(output / "progress.json", {"complete_jobs": len(results),
                "expected_jobs": len(specs), "failures": 0, "runtime_smoke": smoke})
    summary = {"status": "complete", "runtime_smoke": smoke,
        "complete_jobs": len(results), "expected_jobs": len(specs),
        "complete_candidate_rows": sum(result["row_count"] for result in results),
        "controls_recomputed": False, "new_world_or_transport_generation": False,
        "wall_seconds": time.perf_counter() - started,
        "source_lock_sha256": source_sha, "input_lock_sha256": input_sha}
    write_json(output / ("runtime_smoke.json" if smoke else "execution.json"), summary)
    return summary

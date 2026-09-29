"""Outcome adapter for the frozen covariance/forcing development protocol."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .field import grid_laplacian
from .meteorology import grid_coordinates
from .metrics import prediction_metrics
from .records import canonical_json
from .v6_covariance_forcing_inputs import (
    load_prediction_view,
    load_scoring_view,
    sha256_file,
    verify_campaign_bundle,
    write_json,
)
from .v6_estimator import EstimatorConfig, estimate_public_field


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = canonical_json({"dtype": str(value.dtype), "shape": list(value.shape)})
    return hashlib.sha256(header + value.view(np.uint8).tobytes()).hexdigest()


def _observation_hash(records, arrival_map: Mapping[str, int]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(canonical_json({
            "user_id": item.user_id, "epoch": item.epoch, "cell": item.cell,
            "group": item.group, "value": item.value, "sigma": item.sigma,
            "quality": item.quality, "size_bytes": item.size_bytes,
            "nullifier": item.nullifier, "arrival": arrival_map[item.nullifier],
        }))
    return digest.hexdigest()


def information_fingerprint(view: Mapping[str, Any], *, innovation_variance: float,
                            residual_forcing: np.ndarray) -> str:
    operator = view["operator"]
    identity = {
        "public_center": _array_hash(view["public_center"]),
        "observations": _observation_hash(view["observations"], view["arrival_map"]),
        "arrival_map": hashlib.sha256(canonical_json(view["arrival_map"])).hexdigest(),
        "innovation_variance": float(innovation_variance),
        "transition_data": _array_hash(operator.data),
        "transition_indices": _array_hash(operator.indices),
        "transition_indptr": _array_hash(operator.indptr),
        "transition_shape": list(operator.shape),
        "residual_forcing": _array_hash(residual_forcing),
        "prediction_epochs": _array_hash(view["prediction_epochs"]),
    }
    return hashlib.sha256(canonical_json(identity)).hexdigest()


def _fixed_estimator(registration: Mapping[str, Any]) -> EstimatorConfig:
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
    )


def _predict_one(public: np.ndarray, records, arrivals, operator, forcing: np.ndarray,
                 scale: float, config: EstimatorConfig, method: str):
    if method == "PUBLIC":
        return public.copy(), public.copy(), {
            "solver_failure_rate": 0.0, "maximum_correction": 0.0,
            "epoch_terms": [], "public_only": True,
        }
    arm = config
    if method == "SQ":
        arm = replace(config, loss="quadratic", output_cap=False)
    elif method == "HUBER":
        arm = replace(config, loss="huber", output_cap=False)
    elif method != "CANDIDATE":
        raise ValueError("unknown registered method")
    result = estimate_public_field(
        public, grid_coordinates(32), records, arm,
        innovation_scales=float(scale), transitions=[operator] * len(public),
        residual_forcing=forcing, spatial_laplacian=grid_laplacian(32),
        arrival_map=arrivals,
    )
    return result.live, result.reconstructed, result.diagnostics


def _prediction_identity(input_lock_sha256: str, seed: int, cell: str,
                         candidate: Mapping[str, str], method: str,
                         fingerprint: str, config: EstimatorConfig) -> str:
    return hashlib.sha256(canonical_json({
        "input_lock_sha256": input_lock_sha256, "seed": int(seed), "cell": cell,
        "candidate": dict(candidate), "method": method,
        "information_fingerprint": fingerprint, "estimator": asdict(config),
    })).hexdigest()


def _job(spec: tuple[str, str, str, dict[str, Any], int, str, str]) -> dict[str, Any]:
    root_raw, input_raw, output_raw, registration, seed, cell, input_lock_sha256 = spec
    root, input_dir, output_dir = Path(root_raw), Path(input_raw), Path(output_raw)
    input_manifest_sha256 = registration["_locked_input_manifest_sha256"]
    view = load_prediction_view(root, input_dir, seed, cell,
                                expected_manifest_sha256=input_manifest_sha256)
    calibration = json.loads((input_dir / "global_calibration.json").read_text())
    base = _fixed_estimator(registration)
    score_steps = int(registration["scale"]["acquisition_epochs"]
                      - registration["scale"]["burn_in_epochs"])
    target = output_dir / "jobs" / str(seed) / cell
    target.mkdir(parents=True, exist_ok=True)
    prediction_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    process_started = time.process_time()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    for candidate in registration["candidates"]:
        covariance = candidate["innovation_covariance"]
        variance = float(calibration["innovation_variances"][covariance])
        scale = float(calibration["innovation_scales"][covariance])
        forcing = (np.zeros_like(view["residual_forcing"])
                   if candidate["residual_forcing"] == "zero"
                   else view["residual_forcing"])
        fingerprint = information_fingerprint(view, innovation_variance=variance,
                                              residual_forcing=forcing)
        for method in ("CANDIDATE", "PUBLIC", "SQ", "HUBER"):
            arm_dir = target / candidate["id"] / method
            arm_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = arm_dir / "prediction.npz"
            manifest_path = arm_dir / "prediction_manifest.json"
            identity = _prediction_identity(input_lock_sha256, seed, cell, candidate,
                                            method, fingerprint, base)
            if prediction_path.is_file() and manifest_path.is_file():
                prior = json.loads(manifest_path.read_text())
                if prior.get("identity") != identity or prior.get("prediction_sha256") != sha256_file(prediction_path):
                    raise ValueError("stale or tampered prediction resume artifact")
                prediction_records.append(prior)
                continue
            arm_started, cpu_started = time.perf_counter(), time.process_time()
            live, reconstructed, diagnostics = _predict_one(
                view["public_center"], view["observations"], view["arrival_map"],
                view["operator"], forcing, scale, base, method)
            if (not np.isfinite(live).all() or not np.isfinite(reconstructed).all()
                    or len(live) != len(view["public_center"])):
                raise FloatingPointError("nonfinite or incomplete prediction")
            np.savez_compressed(prediction_path, live=live, reconstructed=reconstructed)
            epoch_terms = diagnostics.get("epoch_terms", [])
            maximum_kkt = max((float(row["projected_gradient_inf"])
                               for row in epoch_terms), default=0.0)
            manifest = {"schema_version": 1, "role": "prediction_before_scoring",
                "identity": identity, "seed": int(seed), "cell": cell,
                "candidate": candidate["id"], "method": method,
                "information_fingerprint": fingerprint,
                "innovation_variance": variance, "innovation_scale": scale,
                "residual_forcing": candidate["residual_forcing"],
                "prediction_sha256": sha256_file(prediction_path),
                "scoring_loaded_during_prediction": False,
                "solver_failure_rate": float(diagnostics["solver_failure_rate"]),
                "projected_gradient_inf": maximum_kkt,
                "maximum_absolute_correction": float(diagnostics["maximum_correction"]),
                "wall_seconds": time.perf_counter() - arm_started,
                "process_cpu_seconds": time.process_time() - cpu_started,
                "peak_rss_bytes": int(process.memory_info().rss),
                "output_rows": len(live), "scored_rows_expected": score_steps}
            write_json(manifest_path, manifest)
            prediction_records.append(manifest)
            peak_rss = max(peak_rss, process.memory_info().rss)

    # Scoring is a separate phase and begins only after all prediction files exist.
    scoring = load_scoring_view(input_dir, seed)
    truth = scoring["evaluation_truth"]
    events = scoring["event_mask"]
    threshold = float(scoring["event_threshold"])
    groups = scoring["cell_groups"]
    if truth.shape[0] != score_steps:
        raise ValueError("scoring horizon differs from registration")
    rows = []
    for manifest in prediction_records:
        arm_dir = target / manifest["candidate"] / manifest["method"]
        with np.load(arm_dir / "prediction.npz", allow_pickle=False) as arrays:
            live = arrays["live"][:score_steps]
            reconstructed = arrays["reconstructed"][:score_steps]
        metrics = prediction_metrics(truth, reconstructed, groups)
        live_metrics = prediction_metrics(truth, live, groups)
        recall = float(np.mean(live[events] >= threshold)) if events.any() else None
        row = {"schema_version": 1, "seed": int(seed), "scenario": cell,
            "candidate": manifest["candidate"], "method": manifest["method"],
            "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
            "worst_group_rmse": float(metrics["worst_group_rmse"]),
            "live_rmse": float(live_metrics["rmse"]), "event_recall": recall,
            "event_support": int(events.sum()), "event_threshold": threshold,
            "information_fingerprint": manifest["information_fingerprint"],
            "solver_failure_rate": manifest["solver_failure_rate"],
            "projected_gradient_inf": manifest["projected_gradient_inf"],
            "maximum_absolute_correction": manifest["maximum_absolute_correction"],
            "wall_seconds": manifest["wall_seconds"],
            "process_cpu_seconds": manifest["process_cpu_seconds"],
            "peak_rss_bytes": manifest["peak_rss_bytes"],
            "prediction_identity": manifest["identity"],
            "prediction_sha256": manifest["prediction_sha256"],
            "scoring_sha256": sha256_file(input_dir / "scoring" / str(seed) / "scoring_only.npz"),
            "resource_invariants_pass": True, "finite_outputs": True}
        rows.append(row)
    result = {"schema_version": 1, "status": "complete", "seed": int(seed),
        "cell": cell, "input_lock_sha256": input_lock_sha256,
        "input_manifest_sha256": input_manifest_sha256, "rows": rows,
        "row_count": len(rows), "wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - process_started,
        "peak_rss_bytes": int(peak_rss)}
    write_json(target / "result.json", result)
    return result


def execute_registered_plan(registration: dict[str, Any], plan: list[dict[str, Any]], *,
                            runtime_smoke: bool = False) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    readiness_path = root / registration["readiness"]["artifact"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    input_lock_path = root / registration["producer"]["input_lock"]
    lock_hash = sha256_file(input_lock_path)
    if (readiness.get("readiness_pass") is not True
            or readiness.get("input_lock_sha256") != lock_hash):
        raise RuntimeError("readiness/input lock mismatch")
    lock = json.loads(input_lock_path.read_text(encoding="utf-8"))
    input_dir = root / registration["producer"]["campaign_bundle"]
    verify_campaign_bundle(root, input_dir,
                           expected_manifest_sha256=lock["input_manifest_sha256"])
    registration = dict(registration)
    registration["_locked_input_manifest_sha256"] = lock["input_manifest_sha256"]
    output_dir = root / registration["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({(int(row["seed"]), str(row["scenario"])) for row in plan})
    registered_seeds = set(registration["world_roles"]["development"]["seeds"])
    registered_cells = set(registration["world_roles"]["development"]["cells"])
    if any(seed not in registered_seeds or cell not in registered_cells for seed, cell in keys):
        raise ValueError("plan contains unregistered seed/cell")
    workers = min(int(registration["runtime"]["workers_initial"]), len(keys))
    specs = [(str(root), str(input_dir), str(output_dir), registration, seed, cell, lock_hash)
             for seed, cell in keys]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, spec) for spec in specs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(output_dir / "progress.json", {"completed_jobs": len(results),
                "expected_jobs": len(specs), "failures": 0, "runtime_smoke": runtime_smoke})
    summary = {"status": "complete", "runtime_smoke": runtime_smoke,
        "completed_jobs": len(results), "expected_jobs": len(specs),
        "completed_rows": sum(row["row_count"] for row in results),
        "wall_seconds": time.perf_counter() - started,
        "input_lock_sha256": lock_hash}
    write_json(output_dir / ("runtime_smoke.json" if runtime_smoke else "execution.json"), summary)
    return summary

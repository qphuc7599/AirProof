"""Runner for the frozen v3 history-normalized Huber development family."""
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
from .v6_covariance_forcing_inputs import (
    load_prediction_view,
    load_scoring_view,
    sha256_file,
    write_json,
)
from .v6_covariance_forcing_runner import (
    _predict_one,
    information_fingerprint,
)
from .v6_estimator import EstimatorConfig
from .v6_history_normalized_registration import assert_registration

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = "configs/v6/history_normalized_huber_development_v3.json"


def load_registration(root: str | Path = ROOT) -> dict[str, Any]:
    value = json.loads((Path(root) / REGISTRATION).read_text(encoding="utf-8"))
    assert_registration(value)
    return value


def fixed_estimator(registration: Mapping[str, Any], budget: float) -> EstimatorConfig:
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
        history_normalized_huber=True, per_user_weight_budget=float(budget),
    )


def verify_source_lock(root: str | Path, registration: Mapping[str, Any]) -> dict:
    root = Path(root)
    path = root / registration["artifacts"]["source_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (lock.get("registration_sha256") != sha256_file(root / REGISTRATION)
            or lock.get("new_outcomes_before_lock") != 0):
        raise ValueError("v3 source lock does not bind the frozen zero-outcome registration")
    mismatches = [relative for relative, expected in lock["source_sha256"].items()
                  if not (root / relative).is_file()
                  or sha256_file(root / relative) != expected]
    if mismatches:
        raise ValueError(f"v3 source lock mismatches: {mismatches}")
    v2_lock_path = root / registration["inputs"]["input_lock"]
    if sha256_file(v2_lock_path) != lock["v2_input_lock_sha256"]:
        raise ValueError("v2 input lock changed after v3 registration")
    v2_analysis = root / registration["development_history"]["v2_analysis"]
    if sha256_file(v2_analysis) != lock["v2_analysis_sha256"]:
        raise ValueError("v2 analysis changed after v3 registration")
    feasibility = root / registration["development_history"]["input_feasibility_audit"]
    if sha256_file(feasibility) != lock["history_budget_input_audit_sha256"]:
        raise ValueError("history-budget feasibility audit changed after v3 registration")
    for relative, expected in lock["reused_control_sha256"].items():
        if not (root / relative).is_file() or sha256_file(root / relative) != expected:
            raise ValueError(f"reused v2 control changed: {relative}")
    return lock


def _identity(lock_sha: str, seed: int, cell: str, candidate: Mapping[str, Any],
              fingerprint: str, estimator: EstimatorConfig) -> str:
    return hashlib.sha256(canonical_json({
        "v3_source_lock_sha256": lock_sha,
        "seed": int(seed), "cell": cell, "candidate": dict(candidate),
        "information_fingerprint": fingerprint,
        "estimator": asdict(estimator),
    })).hexdigest()


def _job(spec: tuple[str, dict[str, Any], int, str, str]) -> dict[str, Any]:
    root_raw, registration, seed, cell, source_lock_sha = spec
    root = Path(root_raw)
    input_dir = root / registration["inputs"]["campaign_bundle"]
    v2_lock = json.loads((root / registration["inputs"]["input_lock"])
                         .read_text(encoding="utf-8"))
    view = load_prediction_view(root, input_dir, seed, cell,
                                expected_manifest_sha256=v2_lock["input_manifest_sha256"])
    calibration = json.loads((input_dir / "global_calibration.json")
                             .read_text(encoding="utf-8"))
    covariance = registration["fixed_estimator"]["innovation_covariance"]
    scale = float(calibration["innovation_scales"][covariance])
    variance = float(calibration["innovation_variances"][covariance])
    forcing = view["residual_forcing"]
    fingerprint = information_fingerprint(
        view, innovation_variance=variance, residual_forcing=forcing)
    output_dir = root / registration["artifacts"]["outcomes"]
    target = output_dir / "jobs" / str(seed) / cell
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / "result.json"
    v2_result_path = (root / registration["controls"]["source_campaign"] / "jobs"
                      / str(seed) / cell / "result.json")
    v2_result = json.loads(v2_result_path.read_text(encoding="utf-8"))
    v2_controls = [row for row in v2_result["rows"]
                   if row["candidate"] == registration["controls"]["source_candidate"]]
    resource_invariants = (len(v2_controls) == 4
        and all(row.get("resource_invariants_pass") is True for row in v2_controls)
        and {row["information_fingerprint"] for row in v2_controls} == {fingerprint})
    if not resource_invariants:
        raise ValueError("locked v2 resource/fingerprint inheritance failed")
    artifact_pairs = []
    for candidate in registration["candidates"]:
        arm = target / candidate["id"]
        artifact_pairs.append(((arm / "prediction.npz").is_file(),
                               (arm / "prediction_manifest.json").is_file()))
    if any(prediction != manifest for prediction, manifest in artifact_pairs):
        raise ValueError("one-sided v3 resume artifact")
    if result_path.is_file() and not all(prediction and manifest
                                         for prediction, manifest in artifact_pairs):
        raise ValueError("v3 result exists without every prediction artifact")
    manifests = []
    process = psutil.Process()
    started = time.perf_counter()
    process_started = time.process_time()
    peak_rss = process.memory_info().rss
    for candidate in registration["candidates"]:
        estimator = fixed_estimator(registration, candidate["per_user_weight_budget"])
        arm = target / candidate["id"]
        arm.mkdir(parents=True, exist_ok=True)
        prediction_path = arm / "prediction.npz"
        manifest_path = arm / "prediction_manifest.json"
        identity = _identity(source_lock_sha, seed, cell, candidate, fingerprint, estimator)
        if prediction_path.is_file() and manifest_path.is_file():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (prior.get("identity") != identity
                    or prior.get("prediction_sha256") != sha256_file(prediction_path)):
                raise ValueError("stale or tampered v3 prediction artifact")
            manifests.append(prior)
            continue
        arm_started, cpu_started = time.perf_counter(), time.process_time()
        live, reconstructed, diagnostics = _predict_one(
            view["public_center"], view["observations"], view["arrival_map"],
            view["operator"], forcing, scale, estimator, "CANDIDATE")
        if (not np.isfinite(live).all() or not np.isfinite(reconstructed).all()
                or len(live) != len(view["public_center"])):
            raise FloatingPointError("nonfinite or incomplete v3 prediction")
        np.savez_compressed(prediction_path, live=live, reconstructed=reconstructed)
        epoch_terms = diagnostics.get("epoch_terms", [])
        maximum_kkt = max((float(row["projected_gradient_inf"])
                           for row in epoch_terms), default=0.0)
        maximum_user_weight = max((float(row.get("maximum_user_window_weight", 0.0))
                                   for row in epoch_terms), default=0.0)
        manifest = {
            "schema_version": 1, "role": "v3 prediction before scoring",
            "identity": identity, "seed": int(seed), "cell": cell,
            "candidate": candidate["id"],
            "per_user_weight_budget": candidate["per_user_weight_budget"],
            "information_fingerprint": fingerprint,
            "innovation_variance": variance, "innovation_scale": scale,
            "residual_forcing": "producer_exact_causal",
            "prediction_sha256": sha256_file(prediction_path),
            "scoring_loaded_during_prediction": False,
            "solver_failure_rate": float(diagnostics["solver_failure_rate"]),
            "projected_gradient_inf": maximum_kkt,
            "maximum_absolute_correction": float(diagnostics["maximum_correction"]),
            "maximum_user_window_weight": maximum_user_weight,
            "wall_seconds": time.perf_counter() - arm_started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_bytes": int(process.memory_info().rss),
        }
        write_json(manifest_path, manifest)
        manifests.append(manifest)
        peak_rss = max(peak_rss, process.memory_info().rss)

    if result_path.is_file():
        prior_result = json.loads(result_path.read_text(encoding="utf-8"))
        prior_rows = {row.get("candidate"): row for row in prior_result.get("rows", [])}
        if (prior_result.get("status") != "complete"
                or prior_result.get("seed") != seed
                or prior_result.get("cell") != cell
                or prior_result.get("v3_source_lock_sha256") != source_lock_sha
                or prior_result.get("row_count") != len(registration["candidates"])
                or set(prior_rows) != {row["id"] for row in registration["candidates"]}
                or any(prior_rows[row["candidate"]].get("prediction_identity")
                       != row["identity"]
                       or prior_rows[row["candidate"]].get("prediction_sha256")
                       != row["prediction_sha256"] for row in manifests)):
            raise ValueError("stale or tampered v3 result resume artifact")
        return prior_result

    scoring = load_scoring_view(input_dir, seed)
    truth, events = scoring["evaluation_truth"], scoring["event_mask"]
    threshold, groups = float(scoring["event_threshold"]), scoring["cell_groups"]
    rows = []
    for manifest in manifests:
        prediction_path = target / manifest["candidate"] / "prediction.npz"
        with np.load(prediction_path, allow_pickle=False) as arrays:
            live = arrays["live"][:len(truth)]
            reconstructed = arrays["reconstructed"][:len(truth)]
        metrics = prediction_metrics(truth, reconstructed, groups)
        live_metrics = prediction_metrics(truth, live, groups)
        rows.append({
            "schema_version": 1, "seed": int(seed), "scenario": cell,
            "candidate": manifest["candidate"], "method": "CANDIDATE",
            "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
            "worst_group_rmse": float(metrics["worst_group_rmse"]),
            "live_rmse": float(live_metrics["rmse"]),
            "event_recall": float(np.mean(live[events] >= threshold)),
            "event_support": int(events.sum()), "event_threshold": threshold,
            "information_fingerprint": fingerprint,
            "solver_failure_rate": manifest["solver_failure_rate"],
            "projected_gradient_inf": manifest["projected_gradient_inf"],
            "maximum_absolute_correction": manifest["maximum_absolute_correction"],
            "maximum_user_window_weight": manifest["maximum_user_window_weight"],
            "per_user_weight_budget": manifest["per_user_weight_budget"],
            "wall_seconds": manifest["wall_seconds"],
            "process_cpu_seconds": manifest["process_cpu_seconds"],
            "peak_rss_bytes": manifest["peak_rss_bytes"],
            "prediction_identity": manifest["identity"],
            "prediction_sha256": manifest["prediction_sha256"],
            "scoring_sha256": sha256_file(
                input_dir / "scoring" / str(seed) / "scoring_only.npz"),
            "resource_invariants_pass": resource_invariants,
            "resource_invariants_source": str(v2_result_path.relative_to(root)).replace(
                "\\", "/"),
            "finite_outputs": True,
        })
    result = {
        "schema_version": 1, "status": "complete", "seed": int(seed),
        "cell": cell, "v3_source_lock_sha256": source_lock_sha,
        "v2_input_lock_sha256": sha256_file(root / registration["inputs"]["input_lock"]),
        "rows": rows, "row_count": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - process_started,
        "peak_rss_bytes": int(peak_rss),
    }
    write_json(result_path, result)
    return result


def execute(*, smoke: bool) -> dict[str, Any]:
    registration = load_registration(ROOT)
    lock = verify_source_lock(ROOT, registration)
    readiness = json.loads((ROOT / registration["artifacts"]["readiness"])
                           .read_text(encoding="utf-8"))
    lock_path = ROOT / registration["artifacts"]["source_lock"]
    lock_sha = sha256_file(lock_path)
    if (readiness.get("pass") is not True
            or readiness.get("source_lock_sha256") != lock_sha):
        raise RuntimeError("v3 readiness/source lock mismatch")
    seeds = (registration["runtime"]["runtime_smoke_seeds"] if smoke
             else registration["inputs"]["development_seeds"])
    keys = [(int(seed), cell) for seed in seeds for cell in registration["inputs"]["cells"]]
    specs = [(str(ROOT), registration, seed, cell, lock_sha) for seed, cell in keys]
    output = ROOT / registration["artifacts"]["outcomes"]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(registration["runtime"]["workers"], len(specs))) as pool:
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
        "complete_candidate_rows": sum(row["row_count"] for row in results),
        "wall_seconds": time.perf_counter() - started,
        "source_lock_sha256": lock_sha,
        "v2_controls_reused": len(lock["reused_control_sha256"]),
    }
    write_json(output / ("runtime_smoke.json" if smoke else "execution.json"), summary)
    return summary

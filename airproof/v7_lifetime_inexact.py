"""Hash-bound inexact-solver certificate for the retained lifetime study.

This module only audits frozen inputs and prediction manifests.  It does not
invoke the estimator or change any scientific outcome.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import linalg as sparse_linalg

from .field import grid_laplacian
from .records import canonical_json
from .v6_covariance_forcing_inputs import load_csr, restore_records, sha256_file
from .v6_covariance_forcing_runner import information_fingerprint
from .v6_lifetime_budget7_validation_v2_runner import (
    _identity,
    fixed_estimator,
    verify_source_lock,
)
from .v6_lifetime_exposure import (
    causal_lifetime_exposure_weights,
    lifetime_recursive_sensitivity_certificate,
)

REGISTRATION = Path("configs/v6/lifetime_budget7_validation_v2.json")
SCRIPT = Path("scripts/certify_v7_lifetime_inexact_solver.py")


def audit_input_bundle(
    root: str | Path,
    bundle_path: str | Path,
    input_lock_path: str | Path,
) -> dict[str, Any]:
    """Rehash every registered payload and verify the immutable bundle root."""
    root = Path(root)
    bundle = root / Path(bundle_path)
    lock_file = root / Path(input_lock_path)
    manifest_file = bundle / "manifest.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(manifest_file)
    if manifest_sha256 != lock.get("manifest_sha256"):
        raise ValueError("input manifest differs from its immutable lock")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = manifest.get("payload_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("input manifest has no payload hash map")
    if int(manifest.get("expected_payload_count", -1)) != len(expected):
        raise ValueError("input manifest payload count is inconsistent")
    actual_paths = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != manifest_file
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise ValueError(f"input bundle membership mismatch: missing={missing}, extra={extra}")
    actual = {relative: sha256_file(bundle / relative) for relative in sorted(expected)}
    mismatches = [relative for relative in actual if actual[relative] != expected[relative]]
    if mismatches:
        raise ValueError(f"input bundle payload hash mismatch: {mismatches}")
    payload_root = hashlib.sha256(canonical_json(actual)).hexdigest()
    if payload_root != manifest.get("payload_root_sha256"):
        raise ValueError("input bundle root differs from its manifest")
    if payload_root != lock.get("payload_root_sha256"):
        raise ValueError("input bundle root differs from its immutable lock")
    return {
        "bundle": Path(bundle_path).as_posix(),
        "manifest": (Path(bundle_path) / "manifest.json").as_posix(),
        "manifest_sha256": manifest_sha256,
        "input_lock": Path(input_lock_path).as_posix(),
        "input_lock_sha256": sha256_file(lock_file),
        "payload_count": len(actual),
        "payload_bytes": int(sum((bundle / relative).stat().st_size for relative in actual)),
        "payload_root_sha256": payload_root,
        "payload_sha256": actual,
        "all_payload_hashes_verified": True,
    }


def maximum_effective_coordinate_weight(
    records,
    arrival_map,
    *,
    lag: int,
    per_user_exposure_budget: float,
) -> tuple[float, dict[str, int]]:
    """Return a safe curvature weight bound for every rolling-window solve."""
    effective, _ = causal_lifetime_exposure_weights(
        records,
        arrival_map,
        lag=lag,
        per_user_exposure_budget=per_user_exposure_budget,
    )
    totals: defaultdict[tuple[int, int], float] = defaultdict(float)
    for item in effective:
        totals[(int(item.epoch), int(item.cell))] += float(item.quality)
    if not totals:
        return 0.0, {"epoch": -1, "cell": -1}
    location, value = max(totals.items(), key=lambda pair: (pair[1], pair[0]))
    return float(value), {"epoch": location[0], "cell": location[1]}


def inexact_replacement_bounds(
    *,
    strong_convexity_mu: float,
    gradient_lipschitz_upper: float,
    projected_gradient_step: float,
    projected_gradient_inf: float,
    maximum_dimension: int,
    solve_count: int,
    recurrence_rho: float,
    exact_uniform_bound: float,
    exact_summed_bound: float,
) -> dict[str, float | int]:
    """Convert a shared natural-residual ceiling into recursive L2 bounds.

    For this separable Huber objective, a gradient difference equals ``H d``
    for a symmetric secant matrix with spectrum in ``[mu, L]``.  Projection on
    the common box is nonexpansive, so ``P(x-alpha grad f(x))`` contracts by
    ``max(|1-alpha mu|, |1-alpha L|)``. The paired replacement bounds require
    this residual ceiling for both numerical executions. The one-path bounds
    compare one numerical execution with its recursively exact counterpart.
    """
    values = (
        strong_convexity_mu,
        gradient_lipschitz_upper,
        projected_gradient_step,
        projected_gradient_inf,
        recurrence_rho,
        exact_uniform_bound,
        exact_summed_bound,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("finite inexact-solver constants required")
    if (
        strong_convexity_mu <= 0
        or gradient_lipschitz_upper < strong_convexity_mu
        or projected_gradient_step <= 0
        or projected_gradient_inf < 0
        or maximum_dimension < 1
        or solve_count < 1
        or not 0 <= recurrence_rho < 1
        or exact_uniform_bound < 0
        or exact_summed_bound < 0
    ):
        raise ValueError("invalid inexact-solver constants")
    contraction = max(
        abs(1.0 - projected_gradient_step * strong_convexity_mu),
        abs(1.0 - projected_gradient_step * gradient_lipschitz_upper),
    )
    if contraction >= 1:
        raise ValueError("implemented projected-gradient map is not contractive")
    residual_l2 = math.sqrt(maximum_dimension) * projected_gradient_inf
    per_solve_error = residual_l2 / (1.0 - contraction)
    one_path_uniform_remainder = per_solve_error / (1.0 - recurrence_rho)
    one_path_summed_remainder = (
        per_solve_error * float(solve_count) / (1.0 - recurrence_rho)
    )
    uniform_remainder = 2.0 * per_solve_error / (1.0 - recurrence_rho)
    summed_remainder = (
        2.0 * per_solve_error * float(solve_count) / (1.0 - recurrence_rho)
    )
    return {
        "projected_gradient_step": float(projected_gradient_step),
        "gradient_map_contraction_upper": float(contraction),
        "maximum_projected_gradient_inf": float(projected_gradient_inf),
        "maximum_window_dimension": int(maximum_dimension),
        "maximum_projected_gradient_l2": float(residual_l2),
        "per_solve_l2_error_upper": float(per_solve_error),
        "solve_count": int(solve_count),
        "one_path_uniform_deviation_from_recursive_exact_upper": float(
            one_path_uniform_remainder
        ),
        "one_path_sum_deviation_from_recursive_exact_upper": float(
            one_path_summed_remainder
        ),
        "uniform_numerical_remainder": float(uniform_remainder),
        "summed_numerical_remainder": float(summed_remainder),
        "inexact_uniform_active_window_l2_bound": float(
            exact_uniform_bound + uniform_remainder
        ),
        "inexact_sum_active_window_l2_bound": float(
            exact_summed_bound + summed_remainder
        ),
    }


def build_inexact_solver_certificate(
    root: str | Path,
    registration_path: str | Path = REGISTRATION,
) -> dict[str, Any]:
    """Audit the locked B=7 study and derive its inexact-solver corollary."""
    root = Path(root)
    registration_path = Path(registration_path)
    registration_file = root / registration_path
    registration = json.loads(registration_file.read_text(encoding="utf-8"))
    fixed = registration["fixed_estimator"]
    estimator = fixed_estimator(registration)
    source_lock_path = Path(registration["input_producer"]["source_lock"])
    source_lock_file = root / source_lock_path
    verify_source_lock(root, registration)
    source_lock_sha256 = sha256_file(source_lock_file)
    bundle_path = Path(registration["input_producer"]["bundle"])
    input_lock_path = Path(registration["input_producer"]["input_lock"])
    input_audit = audit_input_bundle(root, bundle_path, input_lock_path)
    input_lock_file = root / input_lock_path
    input_lock = json.loads(input_lock_file.read_text(encoding="utf-8"))
    input_lock_sha256 = sha256_file(input_lock_file)
    if (
        input_lock.get("registration_sha256") != sha256_file(registration_file)
        or input_lock.get("source_lock_sha256") != source_lock_sha256
        or input_lock.get("scientific_outcomes_before_lock") != 0
    ):
        raise ValueError("validation input lock does not bind the frozen zero-outcome source")

    bundle = root / bundle_path
    operator_path = bundle / "mechanism" / "physical_operator.npz"
    calibration_path = bundle / "global_calibration.json"
    operator = load_csr(operator_path)
    operator_norm_observed = float(
        np.max(
            np.abs(
                sparse_linalg.svds(
                    operator,
                    k=1,
                    which="LM",
                    return_singular_vectors=False,
                    solver="arpack",
                    random_state=0,
                    tol=1e-12,
                    maxiter=100_000,
                )
            )
        )
    )
    maximum_absolute_row_sum = float(np.asarray(abs(operator).sum(axis=1)).max())
    maximum_absolute_column_sum = float(np.asarray(abs(operator).sum(axis=0)).max())
    operator_norm = math.nextafter(
        math.sqrt(maximum_absolute_row_sum * maximum_absolute_column_sum),
        math.inf,
    )
    if operator_norm_observed > operator_norm:
        raise ValueError("observed operator norm exceeds induced-norm upper bound")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    scale_floor = float(
        calibration["innovation_scales"][fixed["innovation_covariance"]]
    )
    innovation_variance = float(
        calibration["innovation_variances"][fixed["innovation_covariance"]]
    )

    seeds = [int(seed) for seed in registration["worlds"]["seeds"]]
    cells = list(registration["worlds"]["cells"])
    lag = int(estimator.lag)
    budget = float(fixed["per_user_exposure_budget"])
    maximum_weight = -1.0
    maximum_weight_location: dict[str, int | str] = {}
    information_fingerprints: dict[tuple[int, str], str] = {}
    observation_files = 0
    input_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for seed in seeds:
        for cell in cells:
            relative = Path("prediction") / str(seed) / f"observations_{cell}.npz"
            records, arrivals = restore_records(bundle / relative)
            context_base = input_manifest["context_base"][cell]
            context_path = (
                bundle / "prediction" / str(seed) / f"context_{context_base}.npz"
            )
            with np.load(context_path, allow_pickle=False) as arrays:
                view = {
                    "public_center": arrays["public_center"],
                    "prediction_epochs": arrays["prediction_epochs"],
                    "operator": operator,
                    "residual_forcing": arrays["residual_forcing"],
                    "observations": records,
                    "arrival_map": arrivals,
                }
                information_fingerprints[(seed, cell)] = information_fingerprint(
                    view,
                    innovation_variance=innovation_variance,
                    residual_forcing=view["residual_forcing"],
                )
            value, location = maximum_effective_coordinate_weight(
                records,
                arrivals,
                lag=lag,
                per_user_exposure_budget=budget,
            )
            observation_files += 1
            if value > maximum_weight:
                maximum_weight = value
                maximum_weight_location = {
                    "seed": seed,
                    "scenario": cell,
                    **location,
                }

    outcomes = root / registration["artifacts"]["outcomes"]
    expected_manifests = {
        outcomes
        / "jobs"
        / str(seed)
        / cell
        / "AP_LIFETIME7"
        / "prediction_manifest.json"
        for seed in seeds
        for cell in cells
    }
    actual_manifests = set(
        outcomes.glob("jobs/*/*/AP_LIFETIME7/prediction_manifest.json")
    )
    if actual_manifests != expected_manifests:
        raise ValueError("AP lifetime prediction-manifest matrix is incomplete or duplicated")
    solver_manifest_sha256 = {}
    prediction_sha256 = {}
    projected = []
    failures = []
    for path in sorted(expected_manifests):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != 1
            or value.get("role") != "validation prediction before scoring"
            or value.get("method") != "AP_LIFETIME7"
            or int(value.get("seed", -1)) not in seeds
            or value.get("cell") not in cells
        ):
            raise ValueError(f"invalid AP prediction manifest identity: {path}")
        seed = int(value["seed"])
        cell = str(value["cell"])
        expected_fingerprint = information_fingerprints[(seed, cell)]
        expected_identity = _identity(
            source_lock_sha256,
            input_lock_sha256,
            seed,
            cell,
            "AP_LIFETIME7",
            expected_fingerprint,
            estimator,
            budget,
        )
        if (
            value.get("identity") != expected_identity
            or value.get("information_fingerprint") != expected_fingerprint
            or value.get("scoring_loaded_during_prediction") is not False
            or value.get("residual_forcing") != "producer_exact_causal"
            or float(value.get("innovation_scale", math.nan)) != scale_floor
            or float(value.get("innovation_variance", math.nan))
            != innovation_variance
        ):
            raise ValueError(f"AP prediction manifest is not source/input bound: {path}")
        prediction_path = path.with_name("prediction.npz")
        observed_prediction_hash = sha256_file(prediction_path)
        if observed_prediction_hash != value.get("prediction_sha256"):
            raise ValueError(f"AP prediction hash mismatch: {prediction_path}")
        manifest_relative = path.relative_to(root).as_posix()
        prediction_relative = prediction_path.relative_to(root).as_posix()
        solver_manifest_sha256[manifest_relative] = sha256_file(path)
        prediction_sha256[prediction_relative] = observed_prediction_hash
        residual = float(value["projected_gradient_inf"])
        failure = float(value["solver_failure_rate"])
        if (
            not math.isfinite(residual)
            or residual < 0
            or not math.isfinite(failure)
            or not 0 <= failure <= 1
        ):
            raise ValueError(f"nonfinite solver audit value: {path}")
        projected.append(residual)
        failures.append(failure)

    time_step = float(estimator.time_step)
    mu = time_step * float(estimator.lambda_zero)
    temporal_weight = float(estimator.lambda_temporal) / time_step
    beta = temporal_weight * operator_norm
    exact = lifetime_recursive_sensitivity_certificate(
        per_user_exposure_budget=budget,
        huber_delta=float(estimator.huber_delta),
        scale_floor=scale_floor,
        strong_convexity_mu=mu,
        boundary_coupling_norm=beta,
    )
    side = int(registration["worlds"]["grid_side"])
    grid_laplacian_norm_observed = 4.0 + 4.0 * math.cos(math.pi / side)
    grid_laplacian_norm = 8.0
    observed_grid_norm = float(
        sparse_linalg.eigsh(
            grid_laplacian(side),
            k=1,
            which="LA",
            return_eigenvectors=False,
            tol=1e-12,
            maxiter=100_000,
        )[0]
    )
    if not math.isclose(
        observed_grid_norm, grid_laplacian_norm_observed, abs_tol=1e-9
    ):
        raise ValueError("registered grid Laplacian differs from the analytic norm")
    quadratic_lipschitz = temporal_weight * (1.0 + operator_norm) ** 2 + time_step * (
        float(estimator.lambda_spatial) * grid_laplacian_norm
        + float(estimator.lambda_zero)
    )
    observed_observation_lipschitz = (
        maximum_weight / (float(estimator.precision_normalizer) * scale_floor**2)
    )
    replacement_coordinate_weight_upper = maximum_weight + budget
    replacement_observation_lipschitz = replacement_coordinate_weight_upper / (
        float(estimator.precision_normalizer) * scale_floor**2
    )
    gradient_lipschitz = quadratic_lipschitz + replacement_observation_lipschitz
    prediction_start, prediction_stop = map(
        int, registration["evaluation_clock"]["prediction_half_open"]
    )
    solve_count = prediction_stop - prediction_start
    maximum_dimension = (lag + 1) * side**2
    inexact = inexact_replacement_bounds(
        strong_convexity_mu=mu,
        gradient_lipschitz_upper=gradient_lipschitz,
        projected_gradient_step=1.0,
        projected_gradient_inf=max(projected),
        maximum_dimension=maximum_dimension,
        solve_count=solve_count,
        recurrence_rho=float(exact["normalized_boundary_contraction"]),
        exact_uniform_bound=float(exact["uniform_active_window_l2_bound"]),
        exact_summed_bound=float(exact["sum_active_window_l2_bound"]),
    )

    source_paths = (
        Path("airproof/field.py"),
        Path("airproof/v6_estimator.py"),
        Path("airproof/v6_lifetime_exposure.py"),
        Path("airproof/v6_lifetime_budget7_validation_v2_runner.py"),
        Path("airproof/v7_lifetime_inexact.py"),
        SCRIPT,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "role": (
            "post-hoc hash-bound numerical certificate for the retained B=7 "
            "validation; no estimator rerun or scientific-outcome change"
        ),
        "residual_definition": (
            "||z-P_box(z-gradient F_t(z))||_infinity with projected-gradient step 1"
        ),
        "proof_obligation": (
            "For the separable Huber objective, every gradient secant matrix is "
            "symmetric with spectrum in [mu,L]. The gradient step contracts by "
            "max(|1-mu|,|1-L|), box projection is nonexpansive, and a fixed point "
            "is the exact constrained minimizer. A paired replacement bound "
            "requires the stated residual ceiling for both executions; triangle "
            "inequality then adds one per-solve error for each execution."
        ),
        "input_bundle_audit": input_audit,
        "solver_outcome_audit": {
            "manifest_count": len(expected_manifests),
            "prediction_count": len(prediction_sha256),
            "solver_manifest_sha256": solver_manifest_sha256,
            "prediction_sha256": prediction_sha256,
            "all_prediction_hashes_verified": True,
            "all_manifest_identities_verified": True,
            "maximum_solver_failure_rate": max(failures),
        },
        "smoothness_certificate": {
            "observation_files": observation_files,
            "maximum_effective_quality_at_one_epoch_cell": maximum_weight,
            "maximum_effective_quality_location": maximum_weight_location,
            "innovation_scale_floor": scale_floor,
            "operator_spectral_norm_observed": operator_norm_observed,
            "operator_spectral_norm_upper": operator_norm,
            "operator_maximum_absolute_row_sum": maximum_absolute_row_sum,
            "operator_maximum_absolute_column_sum": maximum_absolute_column_sum,
            "grid_laplacian_spectral_norm_observed": grid_laplacian_norm_observed,
            "grid_laplacian_spectral_norm_upper": grid_laplacian_norm,
            "temporal_quadratic_norm_upper": temporal_weight
            * (1.0 + operator_norm) ** 2,
            "spatial_ridge_norm_upper": time_step
            * (
                float(estimator.lambda_spatial) * grid_laplacian_norm
                + float(estimator.lambda_zero)
            ),
            "quadratic_gradient_lipschitz_upper": quadratic_lipschitz,
            "observed_observation_gradient_lipschitz_upper": (
                observed_observation_lipschitz
            ),
            "replacement_coordinate_quality_upper": (
                replacement_coordinate_weight_upper
            ),
            "replacement_observation_gradient_lipschitz_upper": (
                replacement_observation_lipschitz
            ),
            "replacement_smoothness_scope": (
                "The unchanged users contribute no more than the observed all-user "
                "coordinate maximum; an arbitrary replacement user contributes no "
                "more than its entire lifetime exposure budget at one coordinate."
            ),
            "gradient_lipschitz_upper": gradient_lipschitz,
            "strong_convexity_mu": mu,
        },
        "exact_minimizer_certificate": exact,
        "inexact_solver_certificate": inexact,
        "corollary_scope": (
            "The one-path numerical deviation bounds apply to each of the 60 "
            "hash- and identity-verified AP_LIFETIME7 paths, across its 630 rolling "
            "solves. The paired replacement bounds are conditional: both compared "
            "executions must satisfy the reported residual ceiling at every solve, "
            "in addition to the fixed-context hypotheses of the exact theorem. The "
            "retained study did not run arbitrary one-user replacement histories."
        ),
        "provenance": {
            "registration": registration_path.as_posix(),
            "registration_sha256": sha256_file(registration_file),
            "source_lock": source_lock_path.as_posix(),
            "source_lock_sha256": source_lock_sha256,
            "source_lock_verified": True,
            "input_lock": input_lock_path.as_posix(),
            "input_lock_sha256": input_lock_sha256,
            "operator": operator_path.relative_to(root).as_posix(),
            "operator_sha256": sha256_file(operator_path),
            "calibration": calibration_path.relative_to(root).as_posix(),
            "calibration_sha256": sha256_file(calibration_path),
            "source_sha256": {
                path.as_posix(): sha256_file(root / path) for path in source_paths
            },
        },
    }
    result["certificate_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result

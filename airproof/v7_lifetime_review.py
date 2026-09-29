"""Reviewer-facing lifetime-stability and exposure-profile audits.

This module is deliberately additive: it reads the frozen v6 registration and
artifacts without changing the source identity of the validated estimator.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .records import Observation, canonical_json
from .v6_covariance_forcing_inputs import load_csr, sha256_file
from .v6_lifetime_exposure import (
    causal_lifetime_exposure_weights,
    lifetime_recursive_sensitivity_certificate,
)

REGISTRATION = Path("configs/v6/lifetime_budget7_validation_v2.json")


def _operator_spectral_norm(operator: sparse.spmatrix) -> float:
    matrix = sparse.csr_matrix(operator, dtype=float)
    if min(matrix.shape) == 1:
        return float(np.linalg.norm(matrix.toarray(), ord=2))
    value = sparse_linalg.svds(
        matrix,
        k=1,
        which="LM",
        return_singular_vectors=False,
        solver="arpack",
        random_state=0,
        tol=1e-12,
        maxiter=100_000,
    )
    return float(np.max(np.abs(value)))


def registered_lifetime_certificate(
    root: str | Path,
    registration_path: str | Path = REGISTRATION,
) -> dict[str, Any]:
    """Derive every theorem constant from the registered executable artifacts."""
    root = Path(root)
    registration_path = Path(registration_path)
    registration_file = root / registration_path
    registration = json.loads(registration_file.read_text(encoding="utf-8"))
    fixed = registration["fixed_estimator"]
    bundle = root / registration["input_producer"]["bundle"]
    operator_path = bundle / "mechanism" / "physical_operator.npz"
    calibration_path = bundle / "global_calibration.json"
    operator = load_csr(operator_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    covariance = fixed["innovation_covariance"]
    scale_floor = float(calibration["innovation_scales"][covariance])
    time_step = float(fixed["time_step"])
    strong_convexity_mu = time_step * float(fixed["lambda_zero"])
    temporal_weight = float(fixed["lambda_temporal"]) / time_step
    operator_norm = _operator_spectral_norm(operator)
    boundary_coupling_norm = temporal_weight * operator_norm
    certificate = lifetime_recursive_sensitivity_certificate(
        per_user_exposure_budget=float(fixed["per_user_exposure_budget"]),
        huber_delta=float(fixed["huber_delta"]),
        scale_floor=scale_floor,
        strong_convexity_mu=strong_convexity_mu,
        boundary_coupling_norm=boundary_coupling_norm,
    )
    solver_manifests = sorted(
        (root / registration["artifacts"]["outcomes"]).glob(
            "jobs/*/*/AP_LIFETIME7/prediction_manifest.json"
        )
    )
    projected = []
    failures = []
    for path in solver_manifests:
        value = json.loads(path.read_text(encoding="utf-8"))
        projected.append(float(value["projected_gradient_inf"]))
        failures.append(float(value["solver_failure_rate"]))
    result: dict[str, Any] = {
        "schema_version": 1,
        "role": "machine-derived lifetime replacement-stability certificate",
        "objective": (
            "sum_i w_i Huber_delta((z_i-r_i)/sigma_i) + "
            "0.5*(lambda_temporal/time_step)*sum_t ||z_t-A_t z_{t-1}-f_t||_2^2 + "
            "0.5*time_step*sum_t z_t^T(lambda_spatial L+lambda_zero I)z_t, "
            "subject to max(-cap,-public)<=z<=cap"
        ),
        "grant_rule": (
            "causal first-arrival order; requested exposure equals declared quality "
            "times the number of remaining lag-window uses; grant is the minimum of "
            "requested and the user's remaining lifetime budget"
        ),
        "boundary_mapping": "the first state in a solve receives A_t z_{t-1}",
        "boundary_norm": "induced matrix two-norm",
        "parameter_derivation": {
            "strong_convexity_mu": "time_step*lambda_zero",
            "temporal_weight": "lambda_temporal/time_step",
            "operator_spectral_norm": "largest singular value of registered CSR operator",
            "boundary_coupling_norm": "temporal_weight*operator_spectral_norm",
            "scale_floor": (
                "registered world-cross-fitted innovation scale shared by all rows"
            ),
        },
        "registered_values": {
            "lambda_zero": float(fixed["lambda_zero"]),
            "lambda_temporal": float(fixed["lambda_temporal"]),
            "lambda_spatial": float(fixed["lambda_spatial"]),
            "time_step": time_step,
            "lag": int(fixed["lag"]),
            "cap": float(fixed["cap"]),
            "operator_spectral_norm": operator_norm,
            "temporal_weight": temporal_weight,
            "innovation_scale_floor": scale_floor,
        },
        "exact_minimizer_certificate": certificate,
        "numerical_solver_audit": {
            "manifest_count": len(solver_manifests),
            "maximum_projected_gradient_inf": max(projected, default=None),
            "maximum_solver_failure_rate": max(failures, default=None),
            "registered_tolerance": float(fixed["tolerance"]),
            "scope": (
                "The theorem is stated for exact minimizers. Projected-gradient "
                "residuals quantify numerical termination; no unproved conversion "
                "from that residual to an additional recursive state bound is claimed."
            ),
        },
        "provenance": {
            "registration": registration_path.as_posix(),
            "registration_sha256": sha256_file(registration_file),
            "operator": operator_path.relative_to(root).as_posix(),
            "operator_sha256": sha256_file(operator_path),
            "calibration": calibration_path.relative_to(root).as_posix(),
            "calibration_sha256": sha256_file(calibration_path),
        },
    }
    result["certificate_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result


def lifetime_exposure_profile(
    observations: Iterable[Observation],
    arrival_map: Mapping[str, int | None],
    *,
    lag: int,
    per_user_exposure_budget: float,
    horizon: int | None = None,
) -> tuple[tuple[Observation, ...], list[dict[str, float | int]], dict[str, Any]]:
    """Reconstruct causal grant decisions and a complete epoch-level budget profile."""
    items = tuple(observations)
    effective, base_audit = causal_lifetime_exposure_weights(
        items,
        arrival_map,
        lag=lag,
        per_user_exposure_budget=per_user_exposure_budget,
    )
    effective_by_id = {row.nullifier: row for row in effective}
    if len(effective_by_id) != len(effective):
        raise ValueError("effective records require unique nullifiers")
    ordered = sorted(
        items,
        key=lambda item: (
            np.inf if arrival_map.get(item.nullifier) is None else int(arrival_map[item.nullifier]),
            int(item.epoch),
            int(item.cell),
            int(item.user_id),
            item.nullifier,
        ),
    )
    if horizon is None:
        finite_arrivals = [
            int(arrival_map[row.nullifier])
            for row in items
            if arrival_map.get(row.nullifier) is not None
        ]
        horizon = max(finite_arrivals, default=-1) + 1
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")

    decisions: defaultdict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    unavailable = 0
    for item in ordered:
        arrival = arrival_map.get(item.nullifier)
        if arrival is None or int(arrival) > item.epoch + lag:
            unavailable += 1
            continue
        arrival = int(arrival)
        multiplicity = item.epoch + lag - arrival + 1
        accepted = effective_by_id.get(item.nullifier)
        granted = 0.0 if accepted is None else float(accepted.quality) * multiplicity
        decisions[arrival].append(
            {
                "nullifier": item.nullifier,
                "user_id": int(item.user_id),
                "requested_exposure": float(item.quality) * multiplicity,
                "granted_exposure": granted,
                "effective_quality": 0.0 if accepted is None else float(accepted.quality),
                "multiplicity": multiplicity,
            }
        )

    remaining: defaultdict[int, float] = defaultdict(
        lambda: float(per_user_exposure_budget)
    )
    seen: set[int] = set()
    profile = []
    cumulative_requested = cumulative_granted = 0.0
    cumulative_input = cumulative_effective = cumulative_dropped = 0
    for epoch in range(horizon):
        epoch_rows = decisions.get(epoch, [])
        active = {int(row["user_id"]) for row in epoch_rows}
        granted_users: set[int] = set()
        epoch_quality = []
        for row in epoch_rows:
            user = int(row["user_id"])
            seen.add(user)
            requested = float(row["requested_exposure"])
            granted = float(row["granted_exposure"])
            cumulative_requested += requested
            cumulative_granted += granted
            cumulative_input += 1
            remaining[user] = max(0.0, remaining[user] - granted)
            if granted > 0:
                cumulative_effective += 1
                granted_users.add(user)
                epoch_quality.append(float(row["effective_quality"]))
            else:
                cumulative_dropped += 1
        remaining_values = np.asarray([remaining[user] for user in sorted(seen)], dtype=float)
        profile.append(
            {
                "epoch": epoch,
                "input_records": len(epoch_rows),
                "effective_records": sum(float(row["granted_exposure"]) > 0 for row in epoch_rows),
                "dropped_exhausted_records": sum(
                    float(row["granted_exposure"]) <= 0 for row in epoch_rows
                ),
                "active_users": len(active),
                "granted_active_users": len(granted_users),
                "active_user_fraction": (
                    float(len(granted_users) / len(active)) if active else float("nan")
                ),
                "mean_effective_quality": (
                    float(np.mean(epoch_quality)) if epoch_quality else float("nan")
                ),
                "users_seen": len(seen),
                "mean_remaining_budget": (
                    float(np.mean(remaining_values)) if len(remaining_values) else float("nan")
                ),
                "median_remaining_budget": (
                    float(np.median(remaining_values)) if len(remaining_values) else float("nan")
                ),
                "exhausted_user_fraction": (
                    float(np.mean(remaining_values <= 1e-12))
                    if len(remaining_values)
                    else float("nan")
                ),
                "cumulative_input_records": cumulative_input,
                "cumulative_effective_records": cumulative_effective,
                "cumulative_dropped_exhausted_records": cumulative_dropped,
                "cumulative_requested_exposure": cumulative_requested,
                "cumulative_granted_exposure": cumulative_granted,
            }
        )
    audit = {
        **base_audit,
        "profile_horizon": horizon,
        "profile_unavailable_records": unavailable,
        "profile_final_granted_exposure": cumulative_granted,
        "profile_final_effective_records": cumulative_effective,
        "profile_final_dropped_exhausted_records": cumulative_dropped,
    }
    if not np.isclose(
        cumulative_granted,
        float(base_audit["total_lifetime_exposure"]),
        atol=1e-9,
        rtol=1e-12,
    ):
        raise AssertionError("profile does not reproduce registered lifetime exposure")
    return effective, profile, audit


def uniform_total_exposure_oracle(
    observations: Iterable[Observation],
    arrival_map: Mapping[str, int | None],
    *,
    lag: int,
    per_user_exposure_budget: float,
) -> tuple[tuple[Observation, ...], dict[str, Any]]:
    """Noncausal attribution control spreading the same total exposure per user.

    This control reads all usable records for a user, so it is never a deployable
    policy. It tests whether front-loading, rather than total exposure, drives a
    lifetime result.
    """
    items = tuple(observations)
    requested_by_user: defaultdict[int, float] = defaultdict(float)
    multiplicity: dict[str, int] = {}
    usable = []
    for item in items:
        arrival = arrival_map.get(item.nullifier)
        if arrival is None or int(arrival) > item.epoch + lag:
            continue
        uses = item.epoch + lag - int(arrival) + 1
        multiplicity[item.nullifier] = uses
        requested_by_user[int(item.user_id)] += float(item.quality) * uses
        usable.append(item)
    scale_by_user = {
        user: min(1.0, float(per_user_exposure_budget) / requested)
        for user, requested in requested_by_user.items()
        if requested > 0
    }
    effective = tuple(
        sorted(
            (
                replace(item, quality=float(item.quality) * scale_by_user[int(item.user_id)])
                for item in usable
                if scale_by_user[int(item.user_id)] > 0
            ),
            key=lambda item: (
                int(item.epoch), int(item.cell), int(item.user_id), item.nullifier
            ),
        )
    )
    exposure_by_user: defaultdict[int, float] = defaultdict(float)
    for item in effective:
        exposure_by_user[int(item.user_id)] += (
            float(item.quality) * multiplicity[item.nullifier]
        )
    expected_by_user = {
        user: min(float(per_user_exposure_budget), requested)
        for user, requested in requested_by_user.items()
    }
    if any(
        not np.isclose(exposure_by_user[user], expected, atol=1e-9, rtol=1e-12)
        for user, expected in expected_by_user.items()
    ):
        raise AssertionError("uniform control changed per-user total exposure")
    return effective, {
        "role": "noncausal equal-total-exposure attribution control",
        "input_records": len(items),
        "effective_records": len(effective),
        "users_with_exposure": len(exposure_by_user),
        "maximum_user_lifetime_exposure": max(exposure_by_user.values(), default=0.0),
        "total_lifetime_exposure": float(sum(exposure_by_user.values())),
        "per_user_exposure_budget": float(per_user_exposure_budget),
        "lag": int(lag),
    }

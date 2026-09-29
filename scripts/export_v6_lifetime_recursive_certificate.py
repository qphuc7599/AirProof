from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_lifetime_exposure import lifetime_recursive_sensitivity_certificate

REGISTRATION = ROOT / "configs/v6/causal_lifetime_exposure_validation_v1.json"
CALIBRATION = (
    ROOT
    / "reports/v6/innovation_covariance_forcing_development_v2/inputs/global_calibration.json"
)
OPERATOR = (
    ROOT
    / "reports/v6/innovation_covariance_forcing_development_v2/inputs/mechanism/physical_operator.npz"
)
SOURCE_LOCK = ROOT / "reports/v6/causal_lifetime_exposure_validation_v1/source_lock.json"
OUTPUT = (
    ROOT
    / "reports/v6/causal_lifetime_exposure_development_v4/recursive_sensitivity_certificate.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def load_operator(path: Path) -> sparse.csr_matrix:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"data", "indices", "indptr", "shape"}:
            raise ValueError("unexpected physical-operator schema")
        return sparse.csr_matrix(
            (arrays["data"], arrays["indices"], arrays["indptr"]),
            shape=tuple(int(value) for value in arrays["shape"]),
        )


def main() -> None:
    registration = load_json(REGISTRATION)
    calibration = load_json(CALIBRATION)
    source_lock = load_json(SOURCE_LOCK)
    source_relative = "airproof/v6_lifetime_exposure.py"
    if source_lock["source_sha256"].get(source_relative) != sha256(ROOT / source_relative):
        raise RuntimeError("lifetime mechanism differs from validation source lock")
    operator = load_operator(OPERATOR)
    asymmetry = operator - operator.T
    maximum_asymmetry = (
        float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.0
    )
    if maximum_asymmetry > 1e-12 or np.any(operator.data < 0):
        raise ValueError("certificate requires the locked symmetric nonnegative operator")
    row_sum = np.asarray(operator.sum(axis=1)).ravel()
    if not np.allclose(row_sum, row_sum[0], atol=1e-12, rtol=0):
        raise ValueError("operator does not have the certified constant row sum")
    infinity_norm = float(np.max(np.abs(operator).sum(axis=1)))
    one_norm = float(np.max(np.abs(operator).sum(axis=0)))
    norm_upper = float(np.sqrt(infinity_norm * one_norm))
    eigenvalue_witness = float(row_sum[0])
    if not np.isclose(norm_upper, eigenvalue_witness, atol=1e-12, rtol=0):
        raise ValueError("operator norm upper bound lacks a matching eigenvalue witness")

    fixed = registration["fixed_estimator"]
    scale = float(calibration["innovation_scales"][fixed["innovation_covariance"]])
    strong_convexity = float(fixed["lambda_zero"] * fixed["time_step"])
    temporal_weight = float(fixed["lambda_temporal"] / fixed["time_step"])
    boundary_coupling = temporal_weight * norm_upper
    certificate = lifetime_recursive_sensitivity_certificate(
        per_user_exposure_budget=float(fixed["per_user_exposure_budget"]),
        huber_delta=float(fixed["huber_delta"]),
        scale_floor=scale,
        strong_convexity_mu=strong_convexity,
        boundary_coupling_norm=boundary_coupling,
    )
    payload = {
        "schema_version": 1,
        "role": "analytical whole-solve-sequence replacement-stability certificate",
        "claim_scope": certificate.pop("scope"),
        "registration": str(REGISTRATION.relative_to(ROOT)).replace("\\", "/"),
        "registration_sha256": sha256(REGISTRATION),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "mechanism_source_sha256": sha256(ROOT / source_relative),
        "calibration_sha256": sha256(CALIBRATION),
        "operator_sha256": sha256(OPERATOR),
        "operator_audit": {
            "shape": list(operator.shape),
            "nnz": int(operator.nnz),
            "maximum_absolute_asymmetry": maximum_asymmetry,
            "nonnegative": True,
            "one_norm": one_norm,
            "infinity_norm": infinity_norm,
            "spectral_norm_upper_bound": norm_upper,
            "matching_eigenvalue_witness": eigenvalue_witness,
            "spectral_norm_certified": norm_upper,
        },
        "derivation": {
            "strong_convexity_lower_bound": "lambda_zero * time_step",
            "boundary_gradient_coupling": "(lambda_temporal / time_step) * ||A||_2",
            "recurrence": "d_t <= direct_gradient_t/mu + rho*d_(t-1)",
            "total_direct_gradient_bound": "2*B*huber_delta/scale_floor",
        },
        "certificate": certificate,
        "scientific_outcomes_read_or_generated": 0,
        "not_claimed": [
            "truth accuracy",
            "clean noninferiority",
            "attack attenuation",
            "event recall",
            "changed scheduler or other-user admission stability",
            "identity-splitting robustness",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()

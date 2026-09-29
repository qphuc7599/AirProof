from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from airproof.records import Observation, canonical_json
from airproof.v6_covariance_forcing_inputs import sha256_file
from airproof.v7_lifetime_inexact import (
    audit_input_bundle,
    inexact_replacement_bounds,
    maximum_effective_coordinate_weight,
)

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    ROOT
    / "reports/v7/reviewer_revision/lifetime_retained_diagnostic/"
    "inexact_solver_certificate.json"
)


def _write_locked_bundle(root: Path) -> None:
    bundle = root / "bundle"
    bundle.mkdir()
    payload = bundle / "payload.bin"
    payload.write_bytes(b"locked input")
    hashes = {"payload.bin": sha256_file(payload)}
    payload_root = hashlib.sha256(canonical_json(hashes)).hexdigest()
    manifest = {
        "expected_payload_count": 1,
        "payload_root_sha256": payload_root,
        "payload_sha256": hashes,
    }
    manifest_file = bundle / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "input_lock.json").write_text(
        json.dumps(
            {
                "manifest_sha256": sha256_file(manifest_file),
                "payload_root_sha256": payload_root,
            }
        ),
        encoding="utf-8",
    )


def _observation(user: int, quality: float, key: str) -> Observation:
    return Observation(user, 0, 0, 0, 10.0, 1.0, quality, 512, key, 0, 0)


def test_bundle_audit_rehashes_every_payload_and_fails_on_tampering(tmp_path):
    _write_locked_bundle(tmp_path)
    audit = audit_input_bundle(tmp_path, "bundle", "input_lock.json")
    assert audit["payload_count"] == 1
    assert audit["all_payload_hashes_verified"] is True

    (tmp_path / "bundle" / "payload.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        audit_input_bundle(tmp_path, "bundle", "input_lock.json")


def test_effective_coordinate_weight_uses_lifetime_adjusted_quality():
    records = (
        _observation(1, 1.0, "first"),
        _observation(1, 1.0, "exhausted"),
        _observation(2, 0.5, "other"),
    )
    arrivals = {row.nullifier: 0 for row in records}
    value, location = maximum_effective_coordinate_weight(
        records,
        arrivals,
        lag=2,
        per_user_exposure_budget=3.0,
    )
    assert value == pytest.approx(1.5)
    assert location == {"epoch": 0, "cell": 0}


def test_inexact_bound_propagates_two_solver_errors_across_the_recurrence():
    result = inexact_replacement_bounds(
        strong_convexity_mu=0.1,
        gradient_lipschitz_upper=0.6,
        projected_gradient_step=1.0,
        projected_gradient_inf=1e-4,
        maximum_dimension=4,
        solve_count=10,
        recurrence_rho=0.5,
        exact_uniform_bound=2.0,
        exact_summed_bound=4.0,
    )
    assert result["gradient_map_contraction_upper"] == pytest.approx(0.9)
    assert result["per_solve_l2_error_upper"] == pytest.approx(0.002)
    assert result[
        "one_path_uniform_deviation_from_recursive_exact_upper"
    ] == pytest.approx(0.004)
    assert result[
        "one_path_sum_deviation_from_recursive_exact_upper"
    ] == pytest.approx(0.04)
    assert result["uniform_numerical_remainder"] == pytest.approx(0.008)
    assert result["summed_numerical_remainder"] == pytest.approx(0.08)
    assert result["inexact_uniform_active_window_l2_bound"] == pytest.approx(2.008)
    assert result["inexact_sum_active_window_l2_bound"] == pytest.approx(4.08)


def test_inexact_bound_rejects_negative_exact_bounds():
    with pytest.raises(ValueError, match="invalid inexact-solver constants"):
        inexact_replacement_bounds(
            strong_convexity_mu=0.1,
            gradient_lipschitz_upper=0.6,
            projected_gradient_step=1.0,
            projected_gradient_inf=1e-4,
            maximum_dimension=4,
            solve_count=10,
            recurrence_rho=0.5,
            exact_uniform_bound=-1.0,
            exact_summed_bound=4.0,
        )


def test_committed_certificate_is_self_consistent_and_source_bound():
    result = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    identity = result.pop("certificate_sha256")
    assert hashlib.sha256(canonical_json(result)).hexdigest() == identity
    assert result["input_bundle_audit"]["all_payload_hashes_verified"] is True
    assert result["input_bundle_audit"]["payload_count"] == 146
    assert result["solver_outcome_audit"]["manifest_count"] == 60
    assert result["solver_outcome_audit"]["all_prediction_hashes_verified"] is True
    assert result["solver_outcome_audit"]["all_manifest_identities_verified"] is True
    assert result["provenance"]["source_lock_verified"] is True
    bounds = result["inexact_solver_certificate"]
    assert bounds["per_solve_l2_error_upper"] < 5e-4
    assert bounds[
        "one_path_uniform_deviation_from_recursive_exact_upper"
    ] < 5e-3
    assert bounds["inexact_uniform_active_window_l2_bound"] < 129.0
    assert bounds["inexact_sum_active_window_l2_bound"] < 1296.0
    assert "both compared executions" in result["corollary_scope"]
    for relative, expected in result["provenance"]["source_sha256"].items():
        assert sha256_file(ROOT / relative) == expected

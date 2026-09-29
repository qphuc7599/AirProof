from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from airproof.records import Observation
from airproof.v7_lifetime_review import (
    lifetime_exposure_profile,
    registered_lifetime_certificate,
    uniform_total_exposure_oracle,
)


def observation(user: int, epoch: int, quality: float, key: str) -> Observation:
    return Observation(
        user,
        epoch,
        0,
        0,
        10.0,
        1.0,
        quality,
        512,
        key,
        epoch,
        epoch,
    )


def test_profile_exactly_reproduces_causal_allocator():
    records = (
        observation(1, 0, 1.0, "a"),
        observation(1, 1, 1.0, "b"),
        observation(2, 1, 0.5, "c"),
    )
    arrivals = {row.nullifier: row.epoch for row in records}
    effective, profile, audit = lifetime_exposure_profile(
        records,
        arrivals,
        lag=2,
        per_user_exposure_budget=3.0,
        horizon=4,
    )
    assert [row.nullifier for row in effective] == ["a", "c"]
    assert profile[-1]["cumulative_granted_exposure"] == pytest.approx(4.5)
    assert audit["profile_final_granted_exposure"] == pytest.approx(
        audit["total_lifetime_exposure"]
    )
    assert profile[1]["active_user_fraction"] == pytest.approx(0.5)


def test_uniform_oracle_preserves_total_and_spreads_exposure():
    records = (
        observation(1, 0, 1.0, "a"),
        observation(1, 1, 1.0, "b"),
    )
    arrivals = {row.nullifier: row.epoch for row in records}
    effective, audit = uniform_total_exposure_oracle(
        records,
        arrivals,
        lag=2,
        per_user_exposure_budget=3.0,
    )
    assert len(effective) == 2
    assert [row.quality for row in effective] == pytest.approx([0.5, 0.5])
    assert audit["total_lifetime_exposure"] == pytest.approx(3.0)
    assert audit["maximum_user_lifetime_exposure"] == pytest.approx(3.0)


def test_registered_certificate_reads_config_operator_and_scale(tmp_path):
    registration = {
        "fixed_estimator": {
            "innovation_covariance": "registered",
            "per_user_exposure_budget": 7.0,
            "huber_delta": 1.0,
            "lambda_zero": 0.5,
            "lambda_temporal": 0.25,
            "lambda_spatial": 0.1,
            "time_step": 1.0,
            "lag": 2,
            "cap": 8.0,
            "tolerance": 1e-7,
        },
        "input_producer": {"bundle": "bundle"},
        "artifacts": {"outcomes": "outcomes"},
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(registration), encoding="utf-8")
    mechanism = tmp_path / "bundle" / "mechanism"
    mechanism.mkdir(parents=True)
    matrix = sparse.eye(2, format="csr") * 0.5
    np.savez_compressed(
        mechanism / "physical_operator.npz",
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape),
    )
    (tmp_path / "bundle" / "global_calibration.json").write_text(
        json.dumps({"innovation_scales": {"registered": 2.0}}),
        encoding="utf-8",
    )
    result = registered_lifetime_certificate(tmp_path, "config.json")
    values = result["exact_minimizer_certificate"]
    assert result["registered_values"]["operator_spectral_norm"] == pytest.approx(0.5)
    assert values["strong_convexity_mu"] == pytest.approx(0.5)
    assert values["boundary_coupling_norm"] == pytest.approx(0.125)
    assert values["normalized_boundary_contraction"] == pytest.approx(0.25)
    assert len(result["certificate_sha256"]) == 64

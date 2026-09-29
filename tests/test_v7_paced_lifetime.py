from __future__ import annotations

import pytest

from airproof.records import Observation
from airproof.v7_paced_lifetime import causal_paced_lifetime_exposure_weights


def obs(user: int, epoch: int, key: str) -> Observation:
    return Observation(user, epoch, 0, 0, 10.0, 1.0, 1.0, 512, key, epoch, epoch)


def test_paced_grants_obey_public_cumulative_release_curve():
    records = tuple(obs(1, epoch, f"r{epoch}") for epoch in range(10))
    arrivals = {row.nullifier: row.epoch for row in records}
    effective, audit = causal_paced_lifetime_exposure_weights(
        records,
        arrivals,
        lag=0,
        per_user_exposure_budget=5.0,
        horizon_start=0,
        horizon_end=10,
    )
    assert len(effective) == 10
    assert [row.quality for row in effective] == pytest.approx([0.5] * 10)
    assert audit["maximum_user_lifetime_exposure"] == pytest.approx(5.0)
    assert audit["maximum_release_curve_violation"] <= 1e-10


def test_pacing_retains_late_influence_under_same_total_budget():
    records = (obs(1, 0, "early"), obs(1, 9, "late"))
    arrivals = {row.nullifier: row.epoch for row in records}
    effective, audit = causal_paced_lifetime_exposure_weights(
        records,
        arrivals,
        lag=0,
        per_user_exposure_budget=1.0,
        horizon_start=0,
        horizon_end=10,
    )
    assert [row.quality for row in effective] == pytest.approx([0.1, 0.9])
    assert audit["total_lifetime_exposure"] == pytest.approx(1.0)


def test_future_arrival_does_not_change_a_closed_prefix_grant():
    early = obs(7, 1, "early")
    late = obs(7, 8, "late")
    first, _ = causal_paced_lifetime_exposure_weights(
        (early,),
        {early.nullifier: 1},
        lag=2,
        per_user_exposure_budget=3.0,
        horizon_start=0,
        horizon_end=10,
    )
    extended, _ = causal_paced_lifetime_exposure_weights(
        (early, late),
        {early.nullifier: 1, late.nullifier: 8},
        lag=2,
        per_user_exposure_budget=3.0,
        horizon_start=0,
        horizon_end=10,
    )
    assert first[0] == extended[0]


def test_invalid_horizon_and_unavailable_records_fail_closed():
    record = obs(1, 0, "x")
    with pytest.raises(ValueError):
        causal_paced_lifetime_exposure_weights(
            (record,), {}, lag=0, per_user_exposure_budget=1.0,
            horizon_start=2, horizon_end=2,
        )
    effective, audit = causal_paced_lifetime_exposure_weights(
        (record,),
        {record.nullifier: None},
        lag=0,
        per_user_exposure_budget=1.0,
        horizon_start=0,
        horizon_end=10,
    )
    assert not effective
    assert audit["dropped_unavailable_records"] == 1

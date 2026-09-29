from __future__ import annotations

import numpy as np
import pytest

from airproof.records import Observation
from airproof.v6_lifetime_exposure import (
    causal_lifetime_exposure_weights,
    lifetime_recursive_sensitivity_certificate,
)


def obs(user: int, epoch: int, *, quality: float = 1.0, key: str = "x") -> Observation:
    return Observation(user, epoch, 0, 0, 10.0, 1.0, quality, 512,
                       f"{key}-{user}-{epoch}", epoch, epoch)


def test_budget_counts_every_possible_rolling_window_use():
    records = (obs(1, 0, key="a"), obs(1, 1, key="b"), obs(2, 0, key="c"))
    arrivals = {row.nullifier: row.epoch for row in records}
    effective, audit = causal_lifetime_exposure_weights(
        records, arrivals, lag=2, per_user_exposure_budget=3.0)
    user1 = [row for row in effective if row.user_id == 1]
    assert len(user1) == 1 and user1[0].quality == pytest.approx(1.0)
    assert audit["maximum_user_lifetime_exposure"] == pytest.approx(3.0)
    assert audit["dropped_exhausted_records"] == 1


def test_partial_grant_and_late_arrival_use_are_exact():
    records = (obs(1, 0, quality=.5, key="a"), obs(1, 1, key="b"))
    arrivals = {records[0].nullifier: 1, records[1].nullifier: 2}
    effective, audit = causal_lifetime_exposure_weights(
        records, arrivals, lag=2, per_user_exposure_budget=2.0)
    # First record has two uses and consumes 1; second has two uses and receives 1/2 quality.
    assert [row.quality for row in effective] == pytest.approx([.5, .5])
    assert audit["total_lifetime_exposure"] == pytest.approx(2.0)
    assert audit["scaled_records"] == 1


def test_future_record_does_not_change_causal_prefix_allocation():
    prefix = (obs(7, 0, key="a"),)
    extended = prefix + (obs(7, 20, key="future"),)
    first, _ = causal_lifetime_exposure_weights(
        prefix, {prefix[0].nullifier: 0}, lag=6, per_user_exposure_budget=7.0)
    second, _ = causal_lifetime_exposure_weights(
        extended, {row.nullifier: row.epoch for row in extended},
        lag=6, per_user_exposure_budget=7.0)
    assert first[0] == second[0]
    assert len(second) == 1


def test_unavailable_and_invalid_inputs_fail_closed():
    record = obs(1, 0)
    effective, audit = causal_lifetime_exposure_weights(
        (record,), {record.nullifier: None}, lag=6, per_user_exposure_budget=7.0)
    assert not effective and audit["dropped_unavailable_records"] == 1
    with pytest.raises(ValueError):
        causal_lifetime_exposure_weights((record,), {}, lag=-1,
                                         per_user_exposure_budget=7.0)
    with pytest.raises(ValueError):
        causal_lifetime_exposure_weights((record,), {}, lag=6,
                                         per_user_exposure_budget=np.nan)


def test_recursive_lifetime_certificate_uses_total_history_budget():
    certificate = lifetime_recursive_sensitivity_certificate(
        per_user_exposure_budget=28.0,
        huber_delta=1.345,
        scale_floor=2.0,
        strong_convexity_mu=0.05,
        boundary_coupling_norm=0.045,
    )
    direct = 2 * 28.0 * 1.345 / (0.05 * 2.0)
    assert certificate["normalized_boundary_contraction"] == pytest.approx(0.9)
    assert certificate["uniform_active_window_l2_bound"] == pytest.approx(direct)
    assert certificate["sum_active_window_l2_bound"] == pytest.approx(10 * direct)


@pytest.mark.parametrize(
    "field,value",
    [
        ("per_user_exposure_budget", 0.0),
        ("scale_floor", 0.0),
        ("strong_convexity_mu", 0.0),
        ("boundary_coupling_norm", -1.0),
        ("boundary_coupling_norm", 0.05),
    ],
)
def test_recursive_lifetime_certificate_rejects_invalid_contract(field, value):
    arguments = {
        "per_user_exposure_budget": 28.0,
        "huber_delta": 1.345,
        "scale_floor": 2.0,
        "strong_convexity_mu": 0.05,
        "boundary_coupling_norm": 0.045,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        lifetime_recursive_sensitivity_certificate(**arguments)

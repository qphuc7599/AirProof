from dataclasses import replace

import numpy as np
import pytest

from airproof.records import Observation
from airproof.v8_joint_allocation import causal_joint_lifetime_exposure_weights


def _record(user, epoch, group, value, name):
    return Observation(
        user_id=user,
        epoch=epoch,
        cell=group,
        group=group,
        value=value,
        sigma=1.0,
        quality=1.0,
        size_bytes=512,
        nullifier=name,
        direct_arrival=epoch,
        relay_arrival=epoch,
    )


def _allocate(records, arrivals, **overrides):
    public_reference = overrides.pop("public_reference", np.zeros((6, 2)))
    options = dict(
        lag=2,
        per_user_exposure_budget=3.0,
        horizon_start=0,
        horizon_end=6,
        decision_interval=3,
        innovation_scale_floor=1.0,
        innovation_clip=4.0,
        fairness_strength=0.0,
    )
    options.update(overrides)
    return causal_joint_lifetime_exposure_weights(
        records, arrivals, public_reference, **options
    )


def test_causal_cadence_selects_only_arrived_active_highest_innovation():
    records = (
        _record(0, 0, 0, 0.2, "early-expired"),
        _record(0, 1, 0, 1.0, "small"),
        _record(0, 2, 1, 3.0, "large"),
        _record(0, 5, 0, 4.0, "future"),
    )
    selected, diagnostics = _allocate(
        records, {record.nullifier: record.epoch for record in records}
    )
    assert [record.nullifier for record in selected] == ["large", "future"]
    assert diagnostics["maximum_user_lifetime_exposure"] == pytest.approx(3.0)
    assert diagnostics["maximum_release_curve_violation"] == 0.0


def test_lifetime_bound_holds_under_history_replacement_and_late_arrival():
    records = tuple(
        _record(user, epoch, epoch % 2, float(epoch), f"{user}-{epoch}")
        for user in range(3)
        for epoch in range(6)
    )
    arrivals = {record.nullifier: record.epoch for record in records}
    selected, diagnostics = _allocate(records, arrivals)
    exposure = {}
    for record in selected:
        decision = 2 if record.epoch <= 2 else 5
        multiplicity = record.epoch + 2 - decision + 1
        exposure[record.user_id] = exposure.get(record.user_id, 0.0) + (
            record.quality * multiplicity
        )
    assert max(exposure.values()) <= 3.0 + 1e-12
    assert diagnostics["total_lifetime_exposure"] == pytest.approx(sum(exposure.values()))

    late = replace(records[-1], nullifier="late")
    _, late_diagnostics = _allocate((late,), {"late": late.epoch + 3})
    assert late_diagnostics["effective_records"] == 0
    assert late_diagnostics["dropped_unavailable_records"] == 1


def test_group_debt_uses_past_service_to_break_equal_outcome_scores():
    records = (
        _record(0, 2, 0, 2.0, "u0-g0"),
        _record(1, 2, 0, 2.0, "u1-g0"),
        _record(2, 4, 0, 2.0, "u2-g0"),
        _record(2, 5, 1, 2.0, "u2-g1"),
    )
    arrivals = {record.nullifier: record.epoch for record in records}
    selected, diagnostics = _allocate(records, arrivals, fairness_strength=2.0)
    assert "u2-g1" in {record.nullifier for record in selected}
    assert diagnostics["effective_contributors_by_group"] == {"0": 2, "1": 1}


def test_missing_group_is_materialized_and_fixed_lag_drain_is_processed():
    records = (
        _record(0, 5, 0, 2.0, "tail"),
    )
    arrivals = {"tail": 6}
    selected, diagnostics = _allocate(
        records,
        arrivals,
        public_reference=np.zeros((8, 2)),
        processing_end=8,
        group_count=3,
    )
    assert [record.nullifier for record in selected] == ["tail"]
    assert diagnostics["processing_end"] == 8
    assert diagnostics["effective_contributors_by_group"] == {"0": 1, "1": 0, "2": 0}
    assert diagnostics["effective_contributor_gap"] == 1.0


def test_rejects_noncausal_or_malformed_inputs():
    record = _record(0, 0, 0, 1.0, "a")
    with pytest.raises(ValueError, match="public reference"):
        _allocate((record,), {"a": 0}, public_reference=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="decision interval"):
        _allocate((record,), {"a": 0}, decision_interval=0)
    with pytest.raises(ValueError, match="processing_end"):
        _allocate((record,), {"a": 0}, processing_end=5)
    with pytest.raises(ValueError, match="group_count"):
        _allocate((record,), {"a": 0}, group_count=0)

from types import SimpleNamespace

import numpy as np
import pytest

from airproof.records import Observation
from airproof.v7_privacy_controls import (
    nonnegative_stabilized_query,
    reviewer_release_evidence,
)


def _record(user, group, value, *, epoch=0):
    return Observation(
        user, epoch, group, group, value, 1.0, 1.0, 512,
        f"{user}:{epoch}:{group}", epoch, epoch,
    )


def test_nonnegative_query_clips_after_local_aggregation():
    rows = {
        0: [_record(0, 0, -10), _record(0, 0, 30)],
        1: [_record(1, 1, 100)],
    }
    query, counts = nonnegative_stabilized_query(rows, groups=2, upper=50, k_min=2)
    np.testing.assert_allclose(query, [5, 25])
    np.testing.assert_array_equal(counts, [1, 1])


def test_replacement_group_change_respects_two_coordinate_bound():
    left = {0: [_record(0, 0, 50)]}
    right = {0: [_record(0, 1, 50)]}
    q_left, _ = nonnegative_stabilized_query(left, groups=2, upper=50, k_min=20)
    q_right, _ = nonnegative_stabilized_query(right, groups=2, upper=50, k_min=20)
    assert np.abs(q_left - q_right).sum() == pytest.approx(5.0)


def test_reviewer_release_has_requested_scale_and_matched_support():
    steps, groups = 48, 4
    public = np.full((steps, groups), 12.0)
    world = SimpleNamespace(
        seed=73,
        truth=public.copy(),
        cell_groups=np.arange(groups),
        observations=(),
    )
    cfg = {
        "world": {"groups": groups, "burn_in_steps": 8},
        "privacy": {
            "k_min": 20,
            "epsilon_mean": 8 / 28,
            "epsilon_count": 0.0,
            "epsilon_user_max": 8.0,
            "private_eligibility": False,
            "release_deadline_steps": 24,
            "clip": 50.0,
            "residual_clip": 10.0,
            "nonnegative_raw_lower": 0.0,
            "nonnegative_raw_upper": 50.0,
        },
    }
    summary, arrays = reviewer_release_evidence(world, public, {}, cfg)
    control = summary["raw_nonnegative"]
    assert control["replacement_vector_l1_sensitivity"] == pytest.approx(5.0)
    assert control["laplace_scale"] == pytest.approx(17.5)
    assert control["laplace_variance"] == pytest.approx(612.5)
    np.testing.assert_array_equal(
        arrays["raw_nonnegative_mask"], arrays["residual_mask"]
    )
    assert "raw_symmetric_historical_control" in summary
    assert "raw" not in summary

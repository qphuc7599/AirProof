import numpy as np
import pytest

from airproof.v7_event_intervals import (
    fit_world_robust_event_intervals,
    interval_scores,
)


def test_world_robust_radius_uses_worst_calibration_unit_and_no_test_targets():
    truth = np.arange(10, 50, dtype=float).reshape(5, 8)
    mask = truth >= 30
    units = [
        (truth - 1, truth, mask),
        (truth - 3, truth, mask),
    ]
    model = fit_world_robust_event_intervals(
        units,
        clock="live",
        calibration_split_id="cal",
        evaluation_split_id="test",
        minimum_event_support=20,
    )
    assert model.radii == pytest.approx((3, 3))
    lower, upper = model.predict(truth - 2, evaluation_split_id="test")
    metrics = interval_scores(truth, lower, upper, event_mask=mask)
    assert metrics["coverage_event"] == [1.0, 1.0]
    with pytest.raises(ValueError, match="bound evaluation"):
        model.predict(truth, evaluation_split_id="wrong")


def test_calibrator_rejects_empty_event_support():
    values = np.ones((5, 8))
    with pytest.raises(ValueError, match="support"):
        fit_world_robust_event_intervals(
            [(values, values, np.zeros_like(values, dtype=bool))] * 2,
            clock="reconstructed",
            calibration_split_id="cal",
            evaluation_split_id="test",
            minimum_event_support=20,
        )

import numpy as np
import pytest

from airproof.v6_uncertainty_event_v3 import (
    fit_event_aware_intervals, public_tail_labels,
)


def model():
    pred = np.linspace(1, 100, 4000)
    scale = np.linspace(1, 3, 4000)
    truth = pred + np.where(pred > 95, 20.0, np.where(pred > 80, 8.0, 1.0))
    calibration_pred = np.linspace(1.1, 99.9, 4000)
    calibration_scale = np.linspace(1, 3, 4000)
    calibration_truth = calibration_pred + np.where(
        calibration_pred > 95, 18.0, np.where(calibration_pred > 80, 7.0, 1.2))
    return fit_event_aware_intervals(
        pred, truth, scale,
        calibration_pred, calibration_truth, calibration_scale,
        clock="live", fit_split_id="fit", calibration_split_id="cal",
        evaluation_split_id="eval", fit_end=1, calibration_end=2,
        evaluation_start=3, minimum_fit_support=100,
    )


def test_labels_accept_public_features_only_and_mark_tail():
    labels = public_tail_labels([1, 99], [1, 3],
                                prediction_cuts=(50, 80, 95), scale_cut=2)
    assert labels.tolist() == ["prediction-low|scale-stable",
                               "prediction-tail|scale-uncertain"]


def test_predict_does_not_accept_or_use_evaluation_truth():
    fitted = model()
    pred = np.array([[10.0, 99.0]])
    scales = np.array([[1.1, 2.9]])
    first = fitted.predict(pred, scales, epochs=3, evaluation_split_id="eval")
    second = fitted.predict(pred, scales, epochs=3, evaluation_split_id="eval")
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    low_width = first[1][0, 0, 0] - first[0][0, 0, 0]
    tail_width = first[1][0, 1, 0] - first[0][0, 1, 0]
    assert tail_width > low_width


def test_fit_freezes_event_threshold_and_nominal_levels():
    fitted = model()
    threshold = fitted.event_threshold
    mask = fitted.event_mask([threshold - 1, threshold, threshold + 1])
    assert mask.tolist() == [False, True, True]
    assert fitted.levels == (0.9, 0.95)


def test_roles_must_be_distinct_and_ordered():
    x = np.linspace(1, 10, 100)
    with pytest.raises(ValueError, match="distinct ordered"):
        fit_event_aware_intervals(
            x, x, np.ones_like(x), x, x, np.ones_like(x), clock="live",
            fit_split_id="same", calibration_split_id="same",
            evaluation_split_id="eval", fit_end=1, calibration_end=2,
            evaluation_start=3,
        )

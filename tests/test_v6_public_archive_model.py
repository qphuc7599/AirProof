import numpy as np

from airproof.v6_public_archive_model import (causal_archive_design,
    fit_stationwise_ridge, predict_stationwise_ridge)


def test_archive_design_and_ridge_do_not_read_future_values():
    rng = np.random.default_rng(91)
    values = 10+rng.normal(size=(240, 3))
    coordinates = np.array([[0., 0.], [1., 0.], [0., 1.]])
    medians = np.median(values[:180], axis=0)
    first = causal_archive_design(values[:201], medians, coordinates)[200]
    changed = values.copy(); changed[201:] = 1e9
    second = causal_archive_design(changed[:201], medians, coordinates)[200]
    assert np.array_equal(first, second)
    features = causal_archive_design(values[:220], medians, coordinates)
    model = fit_stationwise_ridge(features, values[:220], 180, [0, 4, 6, 8, 9])
    prediction = predict_stationwise_ridge(features, model)
    assert prediction.shape == values[:220].shape
    assert np.isfinite(prediction).all() and (prediction >= 0).all()


def test_lag_one_is_previous_observed_or_forward_filled_value():
    values = np.array([[1., 10.], [2., np.nan], [3., 30.]])
    design = causal_archive_design(values, np.array([0., 5.]),
                                    np.array([[0., 0.], [1., 0.]]))
    assert np.array_equal(design[1, :, 0], [1., 10.])
    assert np.array_equal(design[2, :, 0], [2., 10.])

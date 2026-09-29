import numpy as np
import pytest

from airproof.v6_public_spatial import spatial_public_features, spatial_public_idw


def test_zero_delay_excludes_current_target_value_and_all_future_values():
    values = np.arange(36, dtype=float).reshape(12, 3)
    coordinates = np.array([[0., 0.], [0., 1.], [0., 2.]])
    medians = np.array([3., 4., 5.])
    first = spatial_public_idw(values, coordinates, medians, neighbors=2,
                               power=2., delay=0)
    changed = values.copy(); changed[5, 1] += 10000; changed[7:] += 20000
    second = spatial_public_idw(changed, coordinates, medians, neighbors=2,
                                power=2., delay=0)
    assert first[5, 1] == pytest.approx(second[5, 1])
    np.testing.assert_allclose(first[:5], second[:5])


def test_delay_clock_and_missing_peer_fallback_are_explicit():
    values = np.array([[1., 2.], [3., 4.], [5., 6.]])
    coordinates = np.array([[0., 0.], [0., 1.]])
    prediction = spatial_public_idw(values, coordinates, [8., 9.], delay=1)
    np.testing.assert_allclose(prediction[0], [8., 9.])
    np.testing.assert_allclose(prediction[1], [2., 1.])
    missing = values.copy(); missing[0, 1] = np.nan
    prediction = spatial_public_idw(missing, coordinates, [8., 9.], delay=1)
    assert prediction[1, 0] == 8.


def test_spatial_public_rejects_single_station():
    with pytest.raises(ValueError):
        spatial_public_idw(np.ones((4, 1)), np.zeros((1, 2)), [1.])


def test_stacked_spatial_features_retain_target_exclusion():
    values = np.arange(30, dtype=float).reshape(10, 3)
    coordinates = np.array([[0., 0.], [0., 1.], [0., 2.]])
    specs = ({"neighbors": 2, "power": 1.}, {"neighbors": 2, "power": 2.})
    first = spatial_public_features(values, coordinates, [1., 2., 3.], specs)
    changed = values.copy(); changed[4, 0] += 9000
    second = spatial_public_features(changed, coordinates, [1., 2., 3.], specs)
    np.testing.assert_allclose(first[4, 0], second[4, 0])

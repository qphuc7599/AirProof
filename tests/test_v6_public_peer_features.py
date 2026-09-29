import numpy as np
import pytest

from airproof.v6_public_peer_features import contemporaneous_peer_features


def test_peer_feature_excludes_own_current_value_but_retains_public_peer():
    values = np.arange(30, dtype=float).reshape(10, 3)
    medians = np.array([4., 5., 6.])
    first = contemporaneous_peer_features(values, medians)
    own_changed = values.copy(); own_changed[4, 1] += 10000
    second = contemporaneous_peer_features(own_changed, medians)
    np.testing.assert_allclose(first[4, 1], second[4, 1])
    assert first[4, 0, 1] != second[4, 0, 1]


def test_peer_feature_prefix_is_future_invariant_and_missingness_is_flagged():
    values = np.arange(24, dtype=float).reshape(8, 3)
    medians = np.array([4., 5., 6.])
    values[3, 2] = np.nan
    first = contemporaneous_peer_features(values, medians)
    changed = values.copy(); changed[6:] += 9000
    second = contemporaneous_peer_features(changed, medians)
    np.testing.assert_allclose(first[:6], second[:6])
    assert first[3, 0, 3 + 2] == 0.


def test_peer_features_require_multiple_stations():
    with pytest.raises(ValueError):
        contemporaneous_peer_features(np.ones((3, 1)), [1.])

import numpy as np
import pytest
from scripts.benchmark_v5_epa import (TRAIN_END, VALIDATION, WINDOWS, common_stations,
                                     design, fit_public, predict_public, observed_rmse, score)


def test_chronology_common_mask():
    assert TRAIN_END+24 == VALIDATION[0]
    assert VALIDATION[1]+24 == WINDOWS[0][0]
    assert WINDOWS[0][1] < WINDOWS[1][0]
    inventory = {"epa_candidate_windows": [
        {"indices": WINDOWS[0], "stations_at_least_80pct": ["b", "a"]},
        {"indices": WINDOWS[1], "stations_at_least_80pct": ["a", "c"]}]}
    assert common_stations(inventory) == ["a"]


def test_causal_features_and_frozen_fit_ignore_future():
    rng = np.random.default_rng(11)
    values = 10+rng.normal(size=(120, 3))
    values[20:25, 0] = np.nan
    median, coef = fit_public(values, train_end=70)
    mutated = values.copy(); mutated[70:] = 1000
    other_median, other_coef = fit_public(mutated, train_end=70)
    np.testing.assert_array_equal(median, other_median)
    np.testing.assert_array_equal(coef, other_coef)
    np.testing.assert_array_equal(predict_public(values, median, coef)[:71],
                                  predict_public(mutated, median, coef)[:71])
    features = design(values, median)
    assert features[26, 0, 2] == values[2, 0]
    assert features[25, 0, 1] == values[19, 0]


def test_observed_mask_and_score_lock():
    assert observed_rmse(np.array([1., 100.]), np.array([2., np.nan])) == 1.
    with pytest.raises(ValueError, match="locked selected"):
        score(None, None, None, [1])

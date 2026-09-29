import numpy as np

from airproof.v6_public_rich_features import causal_rich_features
from airproof.v6_public_rolling_forest import rolling_stationwise_forest


def test_rich_features_exclude_current_target_and_all_future_values():
    rng = np.random.default_rng(5); values = rng.uniform(1, 20, (40, 3))
    medians = np.array([8., 9., 10.])
    first = causal_rich_features(values, medians)
    changed = values.copy(); changed[20, 1] += 10000; changed[25:] += 20000
    second = causal_rich_features(changed, medians)
    np.testing.assert_allclose(first[20, 1], second[20, 1])
    np.testing.assert_allclose(first[:20], second[:20])


def test_rolling_forest_prediction_prefix_is_future_invariant():
    rng = np.random.default_rng(6)
    features = rng.normal(size=(80, 2, 5)); targets = rng.normal(10, 2, (80, 2))
    base = np.full_like(targets, 10.)
    kwargs = dict(start_epoch=30, lookback=20, update_every=10,
                  minimum_history=10, n_estimators=8, min_samples_leaf=2)
    first, logs = rolling_stationwise_forest(features, targets, base, **kwargs)
    changed = targets.copy(); changed[55:] += 1000
    second, _ = rolling_stationwise_forest(features, changed, base, **kwargs)
    np.testing.assert_allclose(first[:60], second[:60])
    assert all(item["fit_end"] <= item["prediction_start"] for item in logs)

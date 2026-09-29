import numpy as np

from airproof.v6_public_rolling_hgb import rolling_stationwise_hgb


def test_rolling_hgb_prefix_cannot_read_changed_future_targets():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(80, 2, 3))
    targets = rng.normal(10, 1, size=(80, 2))
    base = np.full_like(targets, 10.)
    kwargs = dict(start_epoch=30, lookback=20, update_every=10,
                  minimum_history=10, max_iter=8, max_leaf_nodes=4)
    first, log = rolling_stationwise_hgb(features, targets, base, **kwargs)
    changed = targets.copy(); changed[55:] += 1000
    second, _ = rolling_stationwise_hgb(features, changed, base, **kwargs)
    np.testing.assert_allclose(first[:60], second[:60])
    assert all(item["fit_end"] <= item["prediction_start"] for item in log)


def test_rolling_hgb_rejects_nonfinite_public_inputs():
    features = np.ones((12, 1, 2)); targets = np.ones((12, 1)); base = targets.copy()
    base[0, 0] = np.nan
    try:
        rolling_stationwise_hgb(features, targets, base, start_epoch=6,
                                lookback=4, minimum_history=4)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite public baseline accepted")

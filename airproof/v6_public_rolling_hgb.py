"""Causal block-updated nonlinear public predictor for station archives."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def rolling_stationwise_hgb(features, targets, base_prediction, *, start_epoch,
                            lookback, update_every=168, minimum_history=168,
                            max_iter=120, max_leaf_nodes=15,
                            learning_rate=.05, l2_regularization=1.,
                            random_state=20260908):
    """Fit each block using only observations strictly before its first epoch."""
    x = np.asarray(features, float)
    y = np.asarray(targets, float)
    prediction = np.asarray(base_prediction, float).copy()
    if (x.ndim != 3 or x.shape[:2] != y.shape or prediction.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(prediction).all()
            or not 0 <= start_epoch < len(y) or lookback < minimum_history
            or update_every < 1 or minimum_history < 2 or max_iter < 1
            or max_leaf_nodes < 2 or learning_rate <= 0 or l2_regularization < 0):
        raise ValueError("valid causal rolling-HGB contract required")
    fits = []
    for block_start in range(start_epoch, len(y), update_every):
        block_end = min(len(y), block_start + update_every)
        fit_start = max(minimum_history, block_start - lookback)
        fitted = 0
        for station in range(y.shape[1]):
            valid = np.isfinite(y[fit_start:block_start, station])
            if int(valid.sum()) < minimum_history // 2:
                continue
            model = HistGradientBoostingRegressor(
                max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                learning_rate=learning_rate, l2_regularization=l2_regularization,
                early_stopping=False, random_state=random_state + station)
            model.fit(x[fit_start:block_start, station][valid],
                      y[fit_start:block_start, station][valid])
            prediction[block_start:block_end, station] = np.maximum(
                model.predict(x[block_start:block_end, station]), 0.)
            fitted += 1
        fits.append({"fit_start": int(fit_start), "fit_end": int(block_start),
                     "prediction_start": int(block_start),
                     "prediction_end": int(block_end), "fitted_stations": fitted})
    return prediction, fits

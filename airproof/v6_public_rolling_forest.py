"""Causal block-updated ExtraTrees public predictor."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


def rolling_stationwise_forest(features, targets, base_prediction, *, start_epoch,
                               lookback=2160, update_every=168,
                               minimum_history=168, n_estimators=120,
                               min_samples_leaf=2, max_depth=None,
                               max_features=1.0, random_state=20260908):
    x = np.asarray(features, float); y = np.asarray(targets, float)
    prediction = np.asarray(base_prediction, float).copy()
    if (x.ndim != 3 or x.shape[:2] != y.shape or prediction.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(prediction).all()
            or not 0 <= start_epoch < len(y) or lookback < minimum_history
            or update_every < 1 or n_estimators < 1 or min_samples_leaf < 1
            or not 0 < max_features <= 1):
        raise ValueError("valid rolling-forest contract required")
    logs = []
    for block_start in range(start_epoch, len(y), update_every):
        block_end = min(len(y), block_start + update_every)
        fit_start = max(minimum_history, block_start - lookback); fitted = 0
        for station in range(y.shape[1]):
            valid = np.isfinite(y[fit_start:block_start, station])
            if int(valid.sum()) < minimum_history // 2:
                continue
            model = ExtraTreesRegressor(n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                max_features=max_features, bootstrap=False, n_jobs=1,
                random_state=random_state + 1009 * station + block_start)
            model.fit(x[fit_start:block_start, station][valid],
                      y[fit_start:block_start, station][valid])
            prediction[block_start:block_end, station] = np.maximum(
                model.predict(x[block_start:block_end, station]), 0.)
            fitted += 1
        logs.append({"fit_start": int(fit_start), "fit_end": int(block_start),
                     "prediction_start": int(block_start), "prediction_end": int(block_end),
                     "fitted_stations": fitted})
    return prediction, logs

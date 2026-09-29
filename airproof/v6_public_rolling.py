"""Causal rolling stationwise ridge baseline with blockwise coefficient updates."""
from __future__ import annotations

import numpy as np


def rolling_stationwise_ridge(features, targets, base_prediction, *, start_epoch,
                              lookback, update_every=24, penalty=1.,
                              minimum_history=168):
    x = np.asarray(features, float)
    y = np.asarray(targets, float)
    prediction = np.asarray(base_prediction, float).copy()
    if (x.shape[:2] != y.shape or prediction.shape != y.shape or x.ndim != 3
            or not np.isfinite(x).all() or not np.isfinite(prediction).all()
            or not 0 <= start_epoch < len(y) or lookback < minimum_history
            or update_every < 1 or penalty <= 0):
        raise ValueError("valid features, targets and rolling-fit contract required")
    fits = []
    for block_start in range(start_epoch, len(y), update_every):
        block_end = min(len(y), block_start+update_every)
        fit_start = max(minimum_history, block_start-lookback)
        station_coefficients = []
        for station in range(y.shape[1]):
            valid = np.isfinite(y[fit_start:block_start, station])
            design = x[fit_start:block_start, station][valid]
            target = y[fit_start:block_start, station][valid]
            if len(target) < minimum_history//2:
                station_coefficients.append(None)
                continue
            mean = design.mean(0); scale = np.maximum(design.std(0), 1e-6)
            standardized = (design-mean)/scale
            augmented = np.column_stack((np.ones(len(standardized)), standardized))
            regularizer = penalty*np.eye(augmented.shape[1]); regularizer[0, 0] = 0
            coefficient = np.linalg.solve(augmented.T@augmented+regularizer,
                                          augmented.T@target)
            station_coefficients.append((mean, scale, coefficient))
            block = (x[block_start:block_end, station]-mean)/scale
            block = np.column_stack((np.ones(len(block)), block))
            prediction[block_start:block_end, station] = np.maximum(block@coefficient, 0.)
        fits.append({"fit_start": fit_start, "fit_end": block_start,
                     "prediction_start": block_start, "prediction_end": block_end,
                     "fitted_stations": sum(item is not None for item in station_coefficients)})
    return prediction, fits

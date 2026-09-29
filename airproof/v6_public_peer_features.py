"""Causal per-peer public covariates with explicit target exclusion."""
from __future__ import annotations

import numpy as np


def contemporaneous_peer_features(values, medians):
    """Return carried peer values and current-availability flags for each target.

    A target station's own current value and availability flag are replaced by
    public training constants in its feature vector.  Other stations at the
    current epoch are public nowcast inputs.
    """
    y = np.asarray(values, float)
    fallback = np.asarray(medians, float)
    if (y.ndim != 2 or fallback.shape != (y.shape[1],)
            or not np.isfinite(fallback).all() or (fallback < 0).any()):
        raise ValueError("time-by-station values and finite medians required")
    steps, stations = y.shape
    if stations < 2:
        raise ValueError("peer features require at least two stations")
    features = np.empty((steps, stations, 2 * stations), float)
    previous = fallback.copy()
    for epoch in range(steps):
        observed = np.isfinite(y[epoch])
        current = np.where(observed, y[epoch], previous)
        for target in range(stations):
            peer_values = current.copy(); peer_values[target] = fallback[target]
            peer_observed = observed.astype(float); peer_observed[target] = 0.
            features[epoch, target] = np.concatenate((peer_values, peer_observed))
        previous = current
    return features

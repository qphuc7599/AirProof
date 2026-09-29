"""Richer causal public covariates for abrupt pollution dynamics."""
from __future__ import annotations

import numpy as np

from .v6_public_peer_features import contemporaneous_peer_features


WINDOWS = (3, 6, 12, 24)


def causal_rich_features(values, medians):
    """Create own-history summaries and target-excluding current-peer features."""
    y = np.asarray(values, float)
    fallback = np.asarray(medians, float)
    if y.ndim != 2 or fallback.shape != (y.shape[1],):
        raise ValueError("time-by-station values and medians required")
    steps, stations = y.shape
    carried = np.empty_like(y); previous = fallback.copy()
    for epoch in range(steps):
        carried[epoch] = previous
        previous = np.where(np.isfinite(y[epoch]), y[epoch], previous)
    summaries = []
    for window in WINDOWS:
        mean = np.empty_like(y); std = np.empty_like(y)
        low = np.empty_like(y); high = np.empty_like(y)
        for epoch in range(steps):
            start = max(0, epoch - window)
            history = carried[start:epoch + 1]
            mean[epoch] = history.mean(0); std[epoch] = history.std(0)
            low[epoch] = history.min(0); high[epoch] = history.max(0)
        summaries.extend((mean, std, low, high))
    lag_one = np.broadcast_to(fallback, y.shape).copy()
    lag_three = lag_one.copy()
    if steps > 1:
        lag_one[1:] = carried[:-1]
    if steps > 3:
        lag_three[3:] = carried[:-3]
    differences = [carried - lag_one, carried - lag_three]
    own = np.stack((*summaries, *differences), axis=-1)
    peers = contemporaneous_peer_features(y, fallback)
    peer_values = peers[:, :, :stations]
    peer_flags = peers[:, :, stations:]
    peer_summary = np.stack((peer_values.mean(2), peer_values.std(2),
                             peer_values.min(2), peer_values.max(2),
                             peer_flags.sum(2)), axis=-1)
    return np.concatenate((own, peers, peer_summary), axis=2)

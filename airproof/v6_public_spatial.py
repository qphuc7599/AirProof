"""Leave-one-station-out public spatial reference with an explicit delay clock."""
from __future__ import annotations

import numpy as np


def spatial_public_idw(values, coordinates, medians, *, neighbors=4,
                       power=2., delay=0):
    """Predict each station from other public stations only.

    At target epoch ``t`` the newest permissible peer row is ``t-delay``.
    The target station is excluded even when delay is positive, which makes the
    control a spatial public reference rather than target persistence.
    """
    y = np.asarray(values, float)
    xy = np.asarray(coordinates, float)
    fallback = np.asarray(medians, float)
    if (y.ndim != 2 or xy.shape != (y.shape[1], 2)
            or fallback.shape != (y.shape[1],)
            or not np.isfinite(xy).all() or not np.isfinite(fallback).all()
            or (fallback < 0).any() or not isinstance(neighbors, int)
            or neighbors < 1 or power <= 0 or not isinstance(delay, int) or delay < 0):
        raise ValueError("valid spatial-public inputs required")
    stations = y.shape[1]
    if stations < 2:
        raise ValueError("leave-one-out reference requires at least two stations")
    distances = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(2))
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1)[:, :min(neighbors, stations - 1)]
    prediction = np.empty_like(y)
    for epoch in range(len(y)):
        source_epoch = epoch - delay
        for target in range(stations):
            if source_epoch < 0:
                prediction[epoch, target] = fallback[target]
                continue
            peer_ids = order[target]
            observed = np.isfinite(y[source_epoch, peer_ids])
            if not observed.any():
                prediction[epoch, target] = fallback[target]
                continue
            ids = peer_ids[observed]
            weights = 1. / np.maximum(distances[target, ids], 1e-9) ** power
            prediction[epoch, target] = np.average(y[source_epoch, ids], weights=weights)
    return np.maximum(prediction, 0.)


def spatial_public_features(values, coordinates, medians, specifications):
    """Stack registered target-excluding spatial references as model covariates."""
    items = tuple(specifications)
    if not items:
        raise ValueError("at least one spatial specification required")
    arrays = []
    for item in items:
        if set(item) != {"neighbors", "power"}:
            raise ValueError("spatial specifications require neighbors and power")
        arrays.append(spatial_public_idw(values, coordinates, medians,
            neighbors=int(item["neighbors"]), power=float(item["power"]), delay=0))
    return np.stack(arrays, axis=-1)

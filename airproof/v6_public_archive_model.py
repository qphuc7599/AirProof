"""Causal feature and frozen ridge helpers for official hourly archives."""
from __future__ import annotations

import numpy as np


LAGS = (1, 2, 3, 6, 24, 168)


def causal_archive_design(values, medians, coordinates):
    values = np.asarray(values, float)
    medians = np.asarray(medians, float)
    coordinates = np.asarray(coordinates, float)
    if values.ndim != 2 or medians.shape != (values.shape[1],):
        raise ValueError("time-by-station values and station medians required")
    if coordinates.shape != (values.shape[1], 2) or not np.isfinite(coordinates).all():
        raise ValueError("finite station coordinates required")
    if not np.isfinite(medians).all():
        raise ValueError("finite training medians required")
    inclusive = np.empty_like(values)
    previous = medians.copy()
    for epoch in range(len(values)):
        previous = np.where(np.isfinite(values[epoch]), values[epoch], previous)
        inclusive[epoch] = previous
    lagged = []
    for lag in LAGS:
        feature = np.broadcast_to(medians, values.shape).copy()
        if len(values) > lag:
            feature[lag:] = inclusive[:-lag]
        lagged.append(feature)
    lag_one = lagged[0]
    station_count = values.shape[1]
    if station_count == 1:
        peer = lag_one.copy()
        neighbor = lag_one.copy()
    else:
        peer = (lag_one.sum(1, keepdims=True)-lag_one)/(station_count-1)
        distances = ((coordinates[:, None, :]-coordinates[None, :, :])**2).sum(2)
        np.fill_diagonal(distances, np.inf)
        nearest = np.argsort(distances, axis=1)[:, :min(4, station_count-1)]
        neighbor = lag_one[:, nearest].mean(2)
    epoch = np.arange(len(values), dtype=float)
    calendar = [np.broadcast_to(signal[:, None], values.shape) for signal in
                (np.sin(2*np.pi*epoch/24), np.cos(2*np.pi*epoch/24),
                 np.sin(2*np.pi*epoch/168), np.cos(2*np.pi*epoch/168))]
    return np.stack((*lagged, peer, neighbor, *calendar), axis=-1)


def fit_stationwise_ridge(features, targets, train_end, feature_indices, *, penalty=1.):
    x = np.asarray(features, float)
    y = np.asarray(targets, float)
    indices = np.asarray(feature_indices, int)
    if x.shape[:2] != y.shape or not 0 < train_end <= len(y):
        raise ValueError("feature/target shape or training endpoint invalid")
    if indices.ndim != 1 or len(indices) == 0 or (indices < 0).any() or (indices >= x.shape[2]).any():
        raise ValueError("invalid feature subset")
    coefficients = []
    means = []
    scales = []
    for station in range(y.shape[1]):
        valid = np.isfinite(y[:train_end, station]) & (np.arange(train_end) >= max(LAGS))
        design = x[:train_end, station][valid][:, indices]
        target = y[:train_end, station][valid]
        mean = design.mean(0)
        scale = np.maximum(design.std(0), 1e-6)
        standardized = (design-mean)/scale
        augmented = np.column_stack((np.ones(len(standardized)), standardized))
        regularizer = penalty*np.eye(augmented.shape[1]); regularizer[0, 0] = 0
        coefficients.append(np.linalg.solve(augmented.T@augmented+regularizer,
                                            augmented.T@target))
        means.append(mean); scales.append(scale)
    return {"feature_indices": indices, "coefficients": np.array(coefficients),
            "means": np.array(means), "scales": np.array(scales),
            "penalty": float(penalty), "train_end": int(train_end)}


def predict_stationwise_ridge(features, model):
    x = np.asarray(features, float)[:, :, model["feature_indices"]]
    standardized = (x-model["means"][None, :, :])/model["scales"][None, :, :]
    augmented = np.concatenate((np.ones((*standardized.shape[:2], 1)), standardized), axis=2)
    return np.maximum(np.einsum("tsf,sf->ts", augmented, model["coefficients"]), 0.)

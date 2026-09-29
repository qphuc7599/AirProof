"""Frozen residual-quantile prediction intervals; no iid coverage theorem claimed."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntervalCalibrator:
    levels: tuple[float, ...]
    pooled_radii: np.ndarray
    stratum_radii: dict[str, np.ndarray]
    calibration_size: int
    minimum_stratum_size: int = 100
    method: str = "frozen absolute-residual quantiles; empirical temporal coverage"


def _radii(errors: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    # Finite-sample rank adjustment is a calibration convention. Dependence means
    # this alone does not yield an exchangeable conformal guarantee.
    ordered = np.sort(abs(errors))
    return np.array([ordered[min(int(np.ceil((len(ordered)+1)*level))-1, len(ordered)-1)] for level in levels])


def fit_interval_calibrator(predictions: np.ndarray, truth: np.ndarray,
                            public_strata: np.ndarray | None = None, *,
                            levels: tuple[float, ...] = (.9, .95),
                            minimum_stratum_size: int = 100) -> IntervalCalibrator:
    pred, target = np.asarray(predictions, float), np.asarray(truth, float)
    if pred.shape != target.shape or pred.size < 20 or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("at least twenty matched finite calibration values required")
    if not levels or any(not 0 < value < 1 for value in levels) or minimum_stratum_size < 20:
        raise ValueError("invalid coverage levels or minimum stratum size")
    errors = (pred-target).ravel()
    strata = {}
    if public_strata is not None:
        labels = np.asarray(public_strata).astype(str)
        if labels.shape != pred.shape:
            raise ValueError("public stratum shape mismatch")
        labels = labels.ravel()
        for label in np.unique(labels):
            selected = errors[labels == label]
            if len(selected) >= minimum_stratum_size:
                strata[str(label)] = _radii(selected, levels)
    return IntervalCalibrator(levels, _radii(errors, levels), strata, len(errors), minimum_stratum_size)


def predict_intervals(calibrator: IntervalCalibrator, predictions: np.ndarray,
                      public_strata: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predictions, float)
    if not np.isfinite(pred).all() or np.any(pred < 0):
        raise ValueError("finite nonnegative predictions required")
    radii = np.broadcast_to(calibrator.pooled_radii, (*pred.shape, len(calibrator.levels))).copy()
    if public_strata is not None:
        labels = np.asarray(public_strata).astype(str)
        if labels.shape != pred.shape:
            raise ValueError("public stratum shape mismatch")
        for label, values in calibrator.stratum_radii.items():
            radii[labels == label] = values
    return np.maximum(pred[..., None]-radii, 0.), pred[..., None]+radii


def interval_metrics(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     levels: tuple[float, ...] = (.9, .95)) -> dict:
    target = np.asarray(truth, float)[..., None]
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    if low.shape != high.shape or low.shape != (*target.shape[:-1], len(levels)):
        raise ValueError("interval shape mismatch")
    if not all(np.isfinite(v).all() for v in (target, low, high)) or np.any(low > high):
        raise ValueError("invalid intervals")
    result = {}
    for index, level in enumerate(levels):
        y, lo, hi = target[..., 0], low[..., index], high[..., index]
        score = hi-lo+2/(1-level)*np.maximum(lo-y, 0)+2/(1-level)*np.maximum(y-hi, 0)
        result[str(level)] = {"coverage": float(np.mean((lo <= y)&(y <= hi))),
                              "mean_width": float(np.mean(hi-lo)),
                              "mean_interval_score": float(np.mean(score)), "n": y.size}
    return result

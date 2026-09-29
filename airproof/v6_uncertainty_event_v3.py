"""Post-v2 event-aware PUBLIC intervals with immutable three-way data roles.

The model uses prediction-time public features only.  Fit and calibration truth
are permitted in their registered partitions; evaluation residuals never update
the interval model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEVELS = (0.9, 0.95)


def _finite_quantiles(values: np.ndarray, levels=LEVELS) -> tuple[float, ...]:
    ordered = np.sort(np.asarray(values, float).ravel())
    if ordered.size < 20 or not np.isfinite(ordered).all():
        raise ValueError("at least 20 finite calibration scores are required")
    return tuple(float(ordered[min(int(np.ceil((ordered.size + 1) * q)) - 1,
                                   ordered.size - 1)]) for q in levels)


def _check_arrays(predictions, truth, public_scales):
    pred, target, scale = (np.asarray(x, float) for x in
                           (predictions, truth, public_scales))
    if (pred.shape != target.shape or pred.shape != scale.shape or pred.size < 20
            or not all(np.isfinite(x).all() for x in (pred, target, scale))
            or (pred < 0).any() or (scale <= 0).any()):
        raise ValueError("aligned finite nonnegative PUBLIC predictions/truth and positive scales required")
    return pred, target, scale


def public_tail_labels(predictions, public_scales, *, prediction_cuts, scale_cut):
    """Prediction-time labels; no target or residual is accepted by this API."""
    pred, scale = np.broadcast_arrays(np.asarray(predictions, float),
                                      np.asarray(public_scales, float))
    if (not np.isfinite(pred).all() or not np.isfinite(scale).all()
            or (pred < 0).any() or (scale <= 0).any()
            or len(prediction_cuts) != 3
            or not np.all(np.diff(np.asarray(prediction_cuts, float)) > 0)
            or not np.isfinite(scale_cut) or scale_cut <= 0):
        raise ValueError("valid frozen public-feature cuts required")
    level = np.where(pred <= prediction_cuts[0], "low",
            np.where(pred <= prediction_cuts[1], "mid",
            np.where(pred <= prediction_cuts[2], "high", "tail")))
    context = np.where(scale <= scale_cut, "stable", "uncertain")
    return np.char.add(np.char.add(np.char.add("prediction-", level), "|scale-"), context)


@dataclass(frozen=True)
class EventAwareIntervals:
    clock: str
    fit_split_id: str
    calibration_split_id: str
    evaluation_split_id: str
    fit_end: float
    calibration_end: float
    evaluation_start: float
    prediction_cuts: tuple[float, float, float]
    scale_cut: float
    event_threshold: float
    pooled_error_scale: float
    stratum_error_scales: tuple[tuple[str, float], ...]
    normalized_radii: tuple[float, float]
    minimum_fit_support: int
    levels: tuple[float, float] = LEVELS

    def labels(self, predictions, public_scales):
        return public_tail_labels(predictions, public_scales,
                                  prediction_cuts=self.prediction_cuts,
                                  scale_cut=self.scale_cut)

    def predict(self, predictions, public_scales, *, epochs,
                evaluation_split_id):
        pred = np.asarray(predictions, float)
        scale = np.broadcast_to(np.asarray(public_scales, float), pred.shape)
        times = np.broadcast_to(np.asarray(epochs, float), pred.shape)
        if (evaluation_split_id != self.evaluation_split_id
                or not np.isfinite(times).all()
                or (times < self.evaluation_start).any()):
            raise ValueError("bound evaluation split after calibration is required")
        labels = self.labels(pred, scale)
        difficulty = np.full(pred.shape, self.pooled_error_scale, float)
        for label, value in self.stratum_error_scales:
            difficulty[labels == label] = value
        radii = difficulty[..., None] * np.asarray(self.normalized_radii)
        return np.maximum(pred[..., None] - radii, 0), pred[..., None] + radii

    def event_mask(self, truth):
        """Evaluation-only reporting mask against the fit-frozen threshold."""
        target = np.asarray(truth, float)
        if not np.isfinite(target).all():
            raise ValueError("finite evaluation truth required for scoring")
        return target >= self.event_threshold


def fit_event_aware_intervals(
    fit_predictions,
    fit_truth,
    fit_public_scales,
    calibration_predictions,
    calibration_truth,
    calibration_public_scales,
    *,
    clock,
    fit_split_id,
    calibration_split_id,
    evaluation_split_id,
    fit_end,
    calibration_end,
    evaluation_start,
    minimum_fit_support=100,
):
    """Fit public tail difficulty, then calibrate fixed normalized radii.

    Prediction cuts (50/80/95%) and the event threshold (95%) are frozen on the
    fit partition.  Fit residuals estimate stratum difficulty.  The disjoint
    calibration partition supplies the finite-sample 90/95% normalized-error
    quantiles.  Evaluation data are not accepted by this function.
    """
    fp, ft, fs = _check_arrays(fit_predictions, fit_truth, fit_public_scales)
    cp, ct, cs = _check_arrays(calibration_predictions, calibration_truth,
                               calibration_public_scales)
    if (clock not in {"live", "reconstructed"}
            or len({fit_split_id, calibration_split_id, evaluation_split_id}) != 3
            or not fit_end < calibration_end < evaluation_start
            or not isinstance(minimum_fit_support, int)
            or minimum_fit_support < 100):
        raise ValueError("distinct ordered fit/calibration/evaluation roles required")
    prediction_cuts = tuple(float(x) for x in np.quantile(fp, (0.5, 0.8, 0.95)))
    if not np.all(np.diff(prediction_cuts) > 0):
        raise ValueError("fit PUBLIC predictions do not support distinct tail strata")
    scale_cut = float(np.quantile(fs, 0.5))
    event_threshold = float(np.quantile(ft, 0.95))
    fit_errors = np.abs(fp - ft)
    pooled = max(float(np.median(fit_errors)), np.finfo(float).eps)
    fit_labels = public_tail_labels(fp, fs, prediction_cuts=prediction_cuts,
                                    scale_cut=scale_cut)
    learned = []
    for label in np.unique(fit_labels):
        values = fit_errors[fit_labels == label]
        if values.size >= minimum_fit_support:
            # Half-pool shrinkage prevents a small tail stratum from producing an
            # unstable zero or extreme normalization factor.
            local = float(np.median(values))
            learned.append((str(label), max(0.5 * local + 0.5 * pooled,
                                            np.finfo(float).eps)))
    calibration_labels = public_tail_labels(cp, cs,
            prediction_cuts=prediction_cuts, scale_cut=scale_cut)
    difficulty = np.full(cp.shape, pooled, float)
    for label, value in learned:
        difficulty[calibration_labels == label] = value
    radii = _finite_quantiles(np.abs(cp - ct) / difficulty)
    return EventAwareIntervals(
        clock, fit_split_id, calibration_split_id, evaluation_split_id,
        float(fit_end), float(calibration_end), float(evaluation_start),
        prediction_cuts, scale_cut, event_threshold, pooled, tuple(learned),
        (float(radii[0]), float(radii[1])), minimum_fit_support,
    )

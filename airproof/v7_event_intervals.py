"""World-robust event intervals calibrated without evaluation residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEVELS = (0.90, 0.95)


def finite_sample_quantile(values, level: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float).ravel())
    if ordered.size < 20 or not np.isfinite(ordered).all() or not 0 < level < 1:
        raise ValueError("finite calibration sample and probability required")
    index = min(int(np.ceil((ordered.size + 1) * level)) - 1, ordered.size - 1)
    return float(ordered[index])


@dataclass(frozen=True)
class WorldRobustEventIntervals:
    clock: str
    calibration_split_id: str
    evaluation_split_id: str
    radii: tuple[float, float]
    calibration_units: int
    minimum_event_support: int
    calibration_event_support: int
    levels: tuple[float, float] = LEVELS

    def predict(self, predictions, *, evaluation_split_id: str):
        prediction = np.asarray(predictions, dtype=float)
        if (
            evaluation_split_id != self.evaluation_split_id
            or not np.isfinite(prediction).all()
            or (prediction < 0).any()
        ):
            raise ValueError("bound evaluation split and finite nonnegative predictions required")
        radius = np.asarray(self.radii, dtype=float)
        return np.maximum(prediction[..., None] - radius, 0.0), prediction[..., None] + radius


def fit_world_robust_event_intervals(
    calibration_units,
    *,
    clock: str,
    calibration_split_id: str,
    evaluation_split_id: str,
    minimum_event_support: int = 100,
) -> WorldRobustEventIntervals:
    """Take the largest finite-sample event radius across calibration worlds/cells.

    Each unit is ``(prediction, truth, event_mask)``.  The mask must have been
    fixed from that world's pre-scoring prefix.  The resulting radius is applied
    to every evaluation point, so evaluation truth is needed only for scoring.
    """
    if (
        clock not in {"live", "reconstructed"}
        or not calibration_split_id
        or not evaluation_split_id
        or calibration_split_id == evaluation_split_id
        or not isinstance(minimum_event_support, int)
        or minimum_event_support < 20
    ):
        raise ValueError("valid disjoint calibration/evaluation roles required")
    unit_radii = []
    total = 0
    for prediction, truth, event_mask in calibration_units:
        pred = np.asarray(prediction, dtype=float)
        target = np.asarray(truth, dtype=float)
        mask = np.asarray(event_mask, dtype=bool)
        if (
            pred.shape != target.shape
            or mask.shape != target.shape
            or not np.isfinite(pred).all()
            or not np.isfinite(target).all()
            or (pred < 0).any()
            or int(mask.sum()) < minimum_event_support
        ):
            raise ValueError("aligned calibration unit with sufficient event support required")
        errors = np.abs(pred[mask] - target[mask])
        unit_radii.append(tuple(finite_sample_quantile(errors, level) for level in LEVELS))
        total += errors.size
    if len(unit_radii) < 2:
        raise ValueError("at least two calibration units required")
    radii = tuple(float(value) for value in np.max(np.asarray(unit_radii), axis=0))
    if radii[0] > radii[1]:
        raise AssertionError("nested event interval radii required")
    return WorldRobustEventIntervals(
        clock=clock,
        calibration_split_id=calibration_split_id,
        evaluation_split_id=evaluation_split_id,
        radii=radii,
        calibration_units=len(unit_radii),
        minimum_event_support=minimum_event_support,
        calibration_event_support=total,
    )


def interval_scores(truth, lower, upper, *, event_mask):
    target = np.asarray(truth, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    mask = np.asarray(event_mask, dtype=bool)
    if (
        low.shape != (*target.shape, 2)
        or high.shape != low.shape
        or mask.shape != target.shape
        or not all(np.isfinite(value).all() for value in (target, low, high))
        or (low > high).any()
        or not mask.any()
    ):
        raise ValueError("valid intervals and nonempty event mask required")
    levels = np.asarray(LEVELS)
    covered = (low <= target[..., None]) & (target[..., None] <= high)
    width = high - low
    score = width + 2.0 / (1.0 - levels) * (
        np.maximum(low - target[..., None], 0.0)
        + np.maximum(target[..., None] - high, 0.0)
    )
    return {
        "support_all": int(target.size),
        "support_event": int(mask.sum()),
        "coverage_all": covered.mean(axis=tuple(range(target.ndim))).tolist(),
        "coverage_event": covered[mask].mean(axis=0).tolist(),
        "width_all": width.mean(axis=tuple(range(target.ndim))).tolist(),
        "width_event": width[mask].mean(axis=0).tolist(),
        "interval_score_all": score.mean(axis=tuple(range(target.ndim))).tolist(),
        "interval_score_event": score[mask].mean(axis=0).tolist(),
    }

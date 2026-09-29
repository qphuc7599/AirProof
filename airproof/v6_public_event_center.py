"""Causal event-aware public centers fitted on a closed public prefix."""
from dataclasses import dataclass

import numpy as np

from airproof.v6_public_dynamics import PublicIncrementModel


@dataclass(frozen=True)
class HighStateHoldModel:
    """Raise an increment forecast toward the last public level in high states.

    The threshold is fixed from the model-development prefix.  Prediction reads
    public values only through t-1.  The one-sided rule does not manufacture a
    peak before one has appeared in the public channel.
    """

    base: PublicIncrementModel
    threshold: float
    blend: float
    threshold_quantile: float

    def predict_next(self, public_history):
        history = np.asarray(public_history, float)
        base_prediction = self.base.predict_next(history)
        last = history[-1]
        active = last >= self.threshold
        lift = np.maximum(last - base_prediction, 0.0)
        return base_prediction + self.blend * active * lift


def make_high_state_hold(base, training_prefix, *, threshold_quantile, blend):
    prefix = np.asarray(training_prefix, float)
    if prefix.ndim != 2 or not np.isfinite(prefix).all():
        raise ValueError("finite two-dimensional training prefix required")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold quantile must be strictly between zero and one")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0,1]")
    return HighStateHoldModel(
        base=base,
        threshold=float(np.quantile(prefix, threshold_quantile)),
        blend=float(blend),
        threshold_quantile=float(threshold_quantile),
    )

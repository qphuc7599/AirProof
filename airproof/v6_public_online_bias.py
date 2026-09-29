"""Causal public-only residual-bias adaptation for a frozen base predictor."""
from __future__ import annotations

import numpy as np


def causal_ewma_bias(base_prediction, public_observations, *, start_epoch,
                     initial_bias, half_life):
    base = np.asarray(base_prediction, float)
    observed = np.asarray(public_observations, float)
    bias = np.asarray(initial_bias, float).copy()
    if (base.shape != observed.shape or base.ndim != 2
            or bias.shape != (base.shape[1],) or not np.isfinite(base).all()
            or not np.isfinite(bias).all() or not 0 <= start_epoch < len(base)
            or not np.isfinite(half_life) or half_life <= 0):
        raise ValueError("valid base, public observations, bias and half-life required")
    rate = 1-np.power(.5, 1/half_life)
    prediction = base.copy()
    bias_history = np.zeros_like(base)
    for epoch in range(start_epoch, len(base)):
        prediction[epoch] = np.maximum(base[epoch]+bias, 0.)
        bias_history[epoch] = bias
        available = np.isfinite(observed[epoch])
        residual = observed[epoch]-base[epoch]
        bias[available] = (1-rate)*bias[available]+rate*residual[available]
    return prediction, bias_history

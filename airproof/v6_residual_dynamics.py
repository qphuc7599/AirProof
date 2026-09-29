"""Exact coordinate conversion, not an estimated physical dynamics model."""
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class FrozenResidualAR:
    mean: np.ndarray
    coefficient: np.ndarray
    pair_counts: np.ndarray
    training_start: int
    training_end: int

    def system(self, steps):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("positive prediction horizon required")
        transition = sparse.diags(self.coefficient, format="csr")
        forcing = np.broadcast_to((1-self.coefficient)*self.mean,
                                  (steps, len(self.mean))).copy()
        return [transition]*steps, forcing


def fit_frozen_residual_ar(public, truth, *, training_start, training_end,
                           ridge=.1, minimum_pairs=100):
    """Fit a stationary centered AR(1) only on an explicit closed partition."""
    center = np.asarray(public, float)
    target = np.asarray(truth, float)
    if (center.shape != target.shape or center.ndim != 2
            or not 0 <= training_start < training_end <= len(center)
            or ridge <= 0 or minimum_pairs < 2 or not np.isfinite(center).all()):
        raise ValueError("valid public/truth fields and closed training interval required")
    residual = target-center
    section = residual[training_start:training_end]
    pooled = section[np.isfinite(section)]
    if len(pooled) < minimum_pairs:
        raise ValueError("insufficient pooled residual calibration")
    pooled_mean = float(np.mean(pooled))
    coefficients = []
    means = []
    counts = []
    for cell in range(center.shape[1]):
        values = section[:, cell]
        valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
        count = int(valid.sum())
        if count < minimum_pairs:
            mean = pooled_mean
            coefficient = 0.
        else:
            paired_previous = values[:-1][valid]
            paired_current = values[1:][valid]
            mean = float(np.mean(np.concatenate((paired_previous, paired_current))))
            x = paired_previous-mean
            coefficient = float(np.dot(x, paired_current-mean)/(np.dot(x, x)+ridge))
            coefficient = float(np.clip(coefficient, 0., .99))
        means.append(mean); coefficients.append(coefficient); counts.append(count)
    return FrozenResidualAR(np.array(means), np.array(coefficients), np.array(counts),
                            int(training_start), int(training_end))


def public_residual_forcing(public: np.ndarray, transitions: Sequence,
                            physical_forcing: np.ndarray, *, previous_public: np.ndarray):
    """Convert x_t=F_t x_(t-1)+u_t+w_t to z_t=x_t-m_t coordinates.

    Returns F_t m_(t-1)+u_t-m_t. All inputs must be public and available at
    the corresponding epoch. The caller owns availability/provenance checks.
    This identity does not estimate u_t and does not imply it is zero.
    """
    center = np.asarray(public, dtype=float)
    forcing = np.asarray(physical_forcing, dtype=float)
    previous = np.asarray(previous_public, dtype=float)
    if (center.ndim != 2 or min(center.shape) < 1 or forcing.shape != center.shape
            or previous.shape != (center.shape[1],)
            or not all(np.isfinite(a).all() for a in (center, forcing, previous))):
        raise ValueError('finite compatible public fields and explicit forcing required')
    if len(transitions) != len(center):
        raise ValueError('one transition per public epoch required')
    result = np.empty_like(center)
    for epoch, transition in enumerate(transitions):
        operator = sparse.csr_matrix(transition)
        if operator.shape != (center.shape[1], center.shape[1]) or not np.isfinite(operator.data).all():
            raise ValueError('finite square transition required')
        result[epoch] = operator @ previous + forcing[epoch] - center[epoch]
        previous = center[epoch]
    return result

"""Public-only adaptive envelope for the frozen external predictor interface.

The estimator is run once without an output box.  Releases are then projected
into an envelope computed only from two causal public forecasts.  This makes the
bounded arm and its robust/unbounded control equal-information and isolates the
effect of the release envelope from the robust estimating objective.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .predictor_wrapper import synthetic_citizen_replay
from .records import Observation
from .v6_estimator import EstimatorConfig, estimate_public_field


@dataclass(frozen=True)
class PublicEnvelopeConfig:
    """Frozen public center and uncertainty-to-radius map."""

    primary_weight: float
    radius_floor: float
    uncertainty_gain: float
    global_radius: float

    def __post_init__(self) -> None:
        values = (self.primary_weight, self.radius_floor,
                  self.uncertainty_gain, self.global_radius)
        if (not np.isfinite(values).all() or not 0 <= self.primary_weight <= 1
                or self.radius_floor <= 0 or self.uncertainty_gain < 0
                or self.global_radius < self.radius_floor):
            raise ValueError("invalid public envelope configuration")


def public_context(primary: np.ndarray, causal_auxiliary: np.ndarray,
                   config: PublicEnvelopeConfig,
                   *, common_shift: np.ndarray | float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """Return a public-only center and radius without looking at scoring truth.

    ``causal_auxiliary`` is a separately cached one-step public prediction.  In
    the PurpleAir archive it is persistence at t, formed from public observation
    t-1.  A common reference-stress shift leaves forecast disagreement unchanged.
    """
    first = np.asarray(primary, float)
    second = np.asarray(causal_auxiliary, float)
    if (first.shape != second.shape or first.ndim != 2 or
            not np.isfinite(first).all() or not np.isfinite(second).all() or
            (first < 0).any() or (second < 0).any()):
        raise ValueError("aligned finite nonnegative public forecasts required")
    shift = np.broadcast_to(np.asarray(common_shift, float), first.shape)
    if not np.isfinite(shift).all():
        raise ValueError("finite public reference shift required")
    shifted_first = np.maximum(first + shift, 0.)
    shifted_second = np.maximum(second + shift, 0.)
    center = (config.primary_weight * shifted_first
              + (1. - config.primary_weight) * shifted_second)
    uncertainty = np.abs(second - first)
    radius = np.minimum(config.global_radius,
                        config.radius_floor + config.uncertainty_gain * uncertainty)
    return center, radius


def project_to_public_envelope(unbounded: np.ndarray, center: np.ndarray,
                               radius: np.ndarray,
                               *, global_radius: float) -> np.ndarray:
    """Project estimates into a fixed global public-reference envelope."""
    estimate = np.asarray(unbounded, float)
    public = np.asarray(center, float)
    bound = np.asarray(radius, float)
    if (estimate.shape != public.shape or bound.shape != public.shape or estimate.ndim != 2):
        raise ValueError("aligned time-by-cell estimates, center and radius required")
    if (not np.isfinite(estimate).all() or not np.isfinite(public).all()
            or not np.isfinite(bound).all() or (public < 0).any()
            or (bound <= 0).any() or not np.isfinite(global_radius)
            or global_radius <= 0 or np.max(bound) > global_radius + 1e-12):
        raise ValueError("invalid public envelope")
    return np.maximum(public + np.clip(estimate - public, -bound, bound), 0.)


def envelope_certificate(center: np.ndarray, radius: np.ndarray,
                         released: np.ndarray, *, global_radius: float) -> dict:
    """Machine-check the per-release cap and history-uniform global bound."""
    public = np.asarray(center, float)
    bound = np.asarray(radius, float)
    output = np.asarray(released, float)
    maximum = float(np.max(np.abs(output - public)))
    return {
        "maximum_realized_correction": maximum,
        "maximum_declared_radius": float(np.max(bound)),
        "global_radius": float(global_radius),
        "same_public_history_uniform_linf_bound": float(2 * global_radius),
        "cap_pass": bool(maximum <= global_radius + 1e-10
                         and np.max(bound) <= global_radius + 1e-12),
        "center_and_radius_inputs": "current/past cached public forecasts only",
        "citizen_or_scoring_truth_used_for_center_or_radius": False,
    }


def matched_corruption_channel(truth: np.ndarray, *, seed: int, kind: str,
                               onset_fraction: float = .5,
                               reports_per_station: int = 3,
                               sigma: float = 3., availability: float = .6,
                               attack_fraction: float = .2,
                               attack_amplitude: float = 40.) -> list[Observation]:
    """Overlay declared corruption on a matched synthetic-citizen replay.

    The real archive supplies the pollution trajectory.  Reports, delays and
    attacker identities are synthetic and matched across clean/attack cells;
    this function does not assert that an observed PurpleAir fault occurred.
    """
    field = np.asarray(truth, float)
    if kind not in ("clean", "drift", "negative_bias"):
        raise ValueError("unknown corruption protocol")
    if not 0 <= onset_fraction < 1 or attack_amplitude <= 0:
        raise ValueError("invalid corruption onset or amplitude")
    clean = synthetic_citizen_replay(
        field, seed=seed, kind="clean", reports_per_station=reports_per_station,
        sigma=sigma, availability=availability, attack_fraction=attack_fraction,
        attack_amplitude=attack_amplitude,
    )
    agents = field.shape[1] * reports_per_station
    attacker_rng = np.random.default_rng(np.random.SeedSequence([seed, 8003]))
    attackers = set(attacker_rng.choice(
        agents, size=round(agents * attack_fraction), replace=False).tolist())
    onset = int(np.floor(len(field) * onset_fraction))
    attacked: list[Observation] = []
    for item in clean:
        malicious = kind != "clean" and item.user_id in attackers and item.epoch >= onset
        value = item.value
        if malicious and kind == "drift":
            value += attack_amplitude * min(1., (item.epoch - onset + 1) / 24.)
        elif malicious and kind == "negative_bias":
            value = max(0., value - attack_amplitude)
        attacked.append(replace(item, value=float(value), corrupted=bool(malicious)))
    return attacked


def run_equal_information_controls(center: np.ndarray, radius: np.ndarray,
                                   coordinates: np.ndarray,
                                   observations: list[Observation], *,
                                   estimator_config: EstimatorConfig,
                                   innovation_scale: float,
                                   global_radius: float) -> dict[str, tuple[np.ndarray, dict]]:
    """Run matched controls and derive bounded releases from one robust solve."""
    robust_config = replace(estimator_config, loss="huber", output_cap=False,
                            input_clip=False)
    quadratic_config = replace(estimator_config, loss="quadratic", output_cap=False,
                               input_clip=False)
    robust = estimate_public_field(center, coordinates, observations, robust_config,
                                   innovation_scales=innovation_scale)
    quadratic = estimate_public_field(center, coordinates, observations, quadratic_config,
                                      innovation_scales=innovation_scale)
    adaptive = project_to_public_envelope(
        robust.live, center, radius, global_radius=global_radius)
    fixed_radius = np.full_like(center, min(8., global_radius))
    fixed = project_to_public_envelope(
        robust.live, center, fixed_radius, global_radius=global_radius)
    return {
        "adaptive_bounded": (adaptive, envelope_certificate(
            center, radius, adaptive, global_radius=global_radius)),
        "robust_unbounded": (robust.live, robust.diagnostics),
        "fixed_cap8": (fixed, envelope_certificate(
            center, fixed_radius, fixed, global_radius=global_radius)),
        "quadratic_unbounded": (quadratic.live, quadratic.diagnostics),
        "public_only": (center.copy(), {
            "citizen_independent": True,
            "method": "frozen public-only center",
        }),
    }

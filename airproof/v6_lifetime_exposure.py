"""Causal per-user exposure accounting across overlapping estimator windows."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np

from .records import Observation


def lifetime_recursive_sensitivity_certificate(
    *,
    per_user_exposure_budget: float,
    huber_delta: float,
    scale_floor: float,
    strong_convexity_mu: float,
    boundary_coupling_norm: float,
) -> dict[str, float | str]:
    """Bound one user's effect across a causal sequence of lag-window solves.

    The certificate fixes public inputs, operators, forcing, scales, other-user
    admissions, and the feasible set. Each solve is ``mu``-strongly convex, and
    its preceding-boundary gradient perturbation is bounded by
    ``boundary_coupling_norm`` times the prior state perturbation. Replacing one
    budgeted history with another gives the factor two in the direct impulse.
    """
    values = {
        "per_user_exposure_budget": per_user_exposure_budget,
        "huber_delta": huber_delta,
        "scale_floor": scale_floor,
        "strong_convexity_mu": strong_convexity_mu,
        "boundary_coupling_norm": boundary_coupling_norm,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("finite sensitivity parameters required")
    if (
        per_user_exposure_budget <= 0
        or huber_delta <= 0
        or scale_floor <= 0
        or strong_convexity_mu <= 0
        or boundary_coupling_norm < 0
    ):
        raise ValueError("positive parameters and nonnegative coupling required")
    contraction = boundary_coupling_norm / strong_convexity_mu
    if contraction >= 1:
        raise ValueError("contractive normalized boundary coupling required")
    direct_impulse = (
        2.0
        * per_user_exposure_budget
        * huber_delta
        / (strong_convexity_mu * scale_floor)
    )
    return {
        **{key: float(value) for key, value in values.items()},
        "normalized_boundary_contraction": float(contraction),
        "uniform_active_window_l2_bound": float(direct_impulse),
        "sum_active_window_l2_bound": float(direct_impulse / (1.0 - contraction)),
        "scope": (
            "whole solve sequence under fixed public context, other-user admissions, "
            "and contractive causal boundary propagation; replacement stability, "
            "not truth error or attack attenuation"
        ),
    }


def causal_lifetime_exposure_weights(
    observations: Iterable[Observation],
    arrival_map: Mapping[str, int | None],
    *,
    lag: int,
    per_user_exposure_budget: float,
) -> tuple[tuple[Observation, ...], dict[str, float | int]]:
    """Allocate a causal budget for total weight across all rolling-window uses.

    A record acquired at ``e`` and admitted at ``a`` can enter solves
    ``a,...,e+lag``. Its declared quality is divided only when the remaining
    lifetime exposure cannot fund every such use. Records with no usable solve or
    no remaining budget are omitted. The decision uses only public timestamps,
    identity, quality, and earlier admissions.
    """
    if not isinstance(lag, int) or lag < 0:
        raise ValueError("nonnegative integer lag required")
    if not np.isfinite(per_user_exposure_budget) or per_user_exposure_budget <= 0:
        raise ValueError("positive finite lifetime exposure budget required")
    items = tuple(observations)
    for item in items:
        if not isinstance(item.user_id, (int, np.integer)):
            raise TypeError("integer user partition required")
        if not np.isfinite(item.quality) or not 0 < item.quality <= 1:
            raise ValueError("quality must be finite and in (0,1]")
    ordered = sorted(items, key=lambda item: (
        np.inf if arrival_map.get(item.nullifier) is None
        else int(arrival_map[item.nullifier]),
        int(item.epoch), int(item.cell), int(item.user_id), item.nullifier,
    ))
    remaining: defaultdict[int, float] = defaultdict(
        lambda: float(per_user_exposure_budget))
    allocated: defaultdict[int, float] = defaultdict(float)
    kept: list[Observation] = []
    dropped_unavailable = dropped_exhausted = scaled = 0
    for item in ordered:
        arrival = arrival_map.get(item.nullifier)
        if arrival is None or arrival > item.epoch + lag:
            dropped_unavailable += 1
            continue
        multiplicity = item.epoch + lag - int(arrival) + 1
        requested = float(item.quality) * multiplicity
        granted = min(requested, remaining[int(item.user_id)])
        if granted <= 0:
            dropped_exhausted += 1
            continue
        effective_quality = granted / multiplicity
        if effective_quality < item.quality - 1e-15:
            scaled += 1
        kept.append(replace(item, quality=effective_quality))
        remaining[int(item.user_id)] -= granted
        allocated[int(item.user_id)] += granted
    maximum = max(allocated.values(), default=0.0)
    if maximum > per_user_exposure_budget + 1e-10:
        raise AssertionError("lifetime exposure budget exceeded")
    kept.sort(key=lambda item: (
        int(item.epoch), int(item.cell), int(item.user_id), item.nullifier))
    return tuple(kept), {
        "input_records": len(items),
        "effective_records": len(kept),
        "scaled_records": scaled,
        "dropped_exhausted_records": dropped_exhausted,
        "dropped_unavailable_records": dropped_unavailable,
        "users_with_exposure": len(allocated),
        "maximum_user_lifetime_exposure": float(maximum),
        "total_lifetime_exposure": float(sum(allocated.values())),
        "per_user_exposure_budget": float(per_user_exposure_budget),
        "lag": lag,
    }

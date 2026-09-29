"""Causal horizon-paced allocation of a fixed per-user lifetime exposure."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np

from .records import Observation


def causal_paced_lifetime_exposure_weights(
    observations: Iterable[Observation],
    arrival_map: Mapping[str, int | None],
    *,
    lag: int,
    per_user_exposure_budget: float,
    horizon_start: int,
    horizon_end: int,
) -> tuple[tuple[Observation, ...], dict[str, float | int | str]]:
    """Release each user's fixed lifetime budget linearly over public time.

    At arrival epoch ``a``, cumulative spend may not exceed
    ``B * (a-horizon_start+1)/(horizon_end-horizon_start)``.  Unused exposure
    carries forward.  This changes only the causal grant schedule: total
    exposure remains at most ``B``, so the whole-sequence sensitivity
    certificate continues to use the same lifetime budget.
    """
    if not isinstance(lag, int) or lag < 0:
        raise ValueError("nonnegative integer lag required")
    if not np.isfinite(per_user_exposure_budget) or per_user_exposure_budget <= 0:
        raise ValueError("positive finite lifetime exposure budget required")
    if not isinstance(horizon_start, int) or not isinstance(horizon_end, int):
        raise TypeError("integer horizon endpoints required")
    if horizon_end <= horizon_start:
        raise ValueError("horizon_end must exceed horizon_start")

    items = tuple(observations)
    for item in items:
        if not isinstance(item.user_id, (int, np.integer)):
            raise TypeError("integer user partition required")
        if not np.isfinite(item.quality) or not 0 < item.quality <= 1:
            raise ValueError("quality must be finite and in (0,1]")
    ordered = sorted(
        items,
        key=lambda item: (
            np.inf if arrival_map.get(item.nullifier) is None else int(arrival_map[item.nullifier]),
            int(item.epoch),
            int(item.cell),
            int(item.user_id),
            item.nullifier,
        ),
    )
    span = horizon_end - horizon_start
    allocated: defaultdict[int, float] = defaultdict(float)
    kept: list[Observation] = []
    dropped_unavailable = dropped_unreleased = scaled = 0
    maximum_release_slack = 0.0
    for item in ordered:
        arrival = arrival_map.get(item.nullifier)
        if arrival is None or arrival > item.epoch + lag:
            dropped_unavailable += 1
            continue
        arrival = int(arrival)
        released_fraction = min(1.0, max(0.0, (arrival - horizon_start + 1) / span))
        released = float(per_user_exposure_budget) * released_fraction
        available = max(0.0, released - allocated[int(item.user_id)])
        multiplicity = item.epoch + lag - arrival + 1
        requested = float(item.quality) * multiplicity
        granted = min(requested, available)
        if granted <= 0:
            dropped_unreleased += 1
            continue
        effective_quality = granted / multiplicity
        if effective_quality < item.quality - 1e-15:
            scaled += 1
        kept.append(replace(item, quality=effective_quality))
        allocated[int(item.user_id)] += granted
        if allocated[int(item.user_id)] > released + 1e-10:
            raise AssertionError("paced cumulative exposure exceeded public release curve")
        maximum_release_slack = max(
            maximum_release_slack, allocated[int(item.user_id)] - released
        )

    maximum = max(allocated.values(), default=0.0)
    if maximum > per_user_exposure_budget + 1e-10:
        raise AssertionError("lifetime exposure budget exceeded")
    kept.sort(key=lambda item: (int(item.epoch), int(item.cell), int(item.user_id), item.nullifier))
    return tuple(kept), {
        "allocation_policy": "linear_public_horizon_accrual_v1",
        "input_records": len(items),
        "effective_records": len(kept),
        "scaled_records": scaled,
        "dropped_unreleased_records": dropped_unreleased,
        "dropped_unavailable_records": dropped_unavailable,
        "users_with_exposure": len(allocated),
        "maximum_user_lifetime_exposure": float(maximum),
        "total_lifetime_exposure": float(sum(allocated.values())),
        "per_user_exposure_budget": float(per_user_exposure_budget),
        "maximum_release_curve_violation": float(maximum_release_slack),
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "lag": lag,
    }

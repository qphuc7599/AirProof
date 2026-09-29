"""Outcome-independent chronology and missingness rules for fresh archives."""
from __future__ import annotations

import numpy as np


def latest_disjoint_windows(observed, *, candidate_start, width=336, gap=168,
                            minimum_support=.8, minimum_stations=4,
                            station_pool=None):
    """Select the latest pair using only an observed/missing mask.

    The first window ends at least ``gap`` epochs before the second begins.
    Candidate second windows and then first windows are searched latest-first.
    """
    mask = np.asarray(observed, bool)
    if mask.ndim != 2 or len(mask) < 2*width+gap or not 0 <= candidate_start < len(mask):
        raise ValueError("invalid archive mask or candidate range")
    if width < 1 or gap < 0 or not 0 < minimum_support <= 1 or minimum_stations < 1:
        raise ValueError("invalid window rule")
    pool = np.arange(mask.shape[1]) if station_pool is None else np.asarray(station_pool, int)
    if pool.ndim != 1 or len(np.unique(pool)) != len(pool) or (pool < 0).any() or (pool >= mask.shape[1]).any():
        raise ValueError("invalid frozen station pool")
    cumulative = np.vstack((np.zeros((1, mask.shape[1]), int), np.cumsum(mask, axis=0)))

    def supported(start, end):
        counts = cumulative[end]-cumulative[start]
        return set(pool[(counts[pool]/width) >= minimum_support].tolist())

    for second_end in range(len(mask), candidate_start+width-1, -1):
        second_start = second_end-width
        if second_start < candidate_start:
            break
        second_support = supported(second_start, second_end)
        latest_first_end = second_start-gap
        for first_end in range(latest_first_end, candidate_start+width-1, -1):
            first_start = first_end-width
            if first_start < candidate_start:
                break
            shared = sorted(second_support & supported(first_start, first_end))
            if len(shared) >= minimum_stations:
                return {"windows": [[first_start, first_end], [second_start, second_end]],
                        "station_indices": shared,
                        "gap": second_start-first_end}
    raise ValueError("no eligible disjoint window pair")

"""Matched synthetic attack overlays for fixed real-pollution histories."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from airproof.predictor_wrapper import synthetic_citizen_replay


def matched_archive_channel(truth, reference, *, seed, kind,
                            reports_per_station=3, attack_fraction=.2,
                            attack_amplitude=12., huber_delta=1.345,
                            sigma=3., hotspot_cells=None):
    """Keep channel randomness fixed while changing only malicious payloads."""
    field = np.asarray(truth, float)
    public = np.asarray(reference, float)
    if field.shape != public.shape or field.ndim != 2:
        raise ValueError("matched truth and public fields required")
    if kind not in ("clean", "drift", "hotspot_suppression", "coordinated_inlier"):
        raise ValueError("unknown archive attack")
    base = synthetic_citizen_replay(field, seed=seed, kind="clean",
                                    reports_per_station=reports_per_station, sigma=sigma)
    agents = field.shape[1]*reports_per_station
    attacker_rng = np.random.default_rng(np.random.SeedSequence([seed, 6006]))
    attackers = set(attacker_rng.choice(agents, size=round(agents*attack_fraction), replace=False).tolist())
    hotspots = set(range(field.shape[1])) if hotspot_cells is None else set(map(int, hotspot_cells))
    if any(cell < 0 or cell >= field.shape[1] for cell in hotspots):
        raise ValueError("hotspot cell outside field")
    attacked = []
    midpoint = len(field)//2
    for item in base:
        malicious = item.user_id in attackers and item.epoch >= midpoint
        value = item.value
        if malicious and kind == "drift":
            value += attack_amplitude*min(1., (item.epoch-midpoint+1)/24.)
        elif malicious and kind == "hotspot_suppression" and item.cell in hotspots:
            value = max(0., value-attack_amplitude)
        elif malicious and kind == "coordinated_inlier":
            # A shared public-relative value lies just inside the declared Huber cutoff.
            value = max(0., public[item.epoch, item.cell]+.95*huber_delta*sigma)
        attacked.append(replace(item, value=float(value), corrupted=bool(malicious and kind != "clean")))
    return attacked

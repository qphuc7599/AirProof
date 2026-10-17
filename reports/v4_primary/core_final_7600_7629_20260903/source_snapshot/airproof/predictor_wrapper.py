"""Actual fixed-lag bounded-correction adapter for a frozen public predictor.

This module evaluates the estimator interface, not transport, DP or a field
deployment. Its citizen observations can be synthetic replays of public fields.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .meteorology import city_graph
from .records import Observation
from .twin import FixedLagTwin


def wrap_public_predictions(
    public_predictions: np.ndarray, coordinates_north_east: np.ndarray,
    observations: list[Observation], *, fixed_lag: int = 6,
    delta: float = 4., correction_clip: float = 8., lambda_temporal: float = .5,
    lambda_spatial: float = .2,
) -> tuple[np.ndarray, dict]:
    predictions = np.asarray(public_predictions, dtype=float)
    xy = np.asarray(coordinates_north_east, dtype=float)
    if predictions.ndim != 2 or not np.isfinite(predictions).all() or np.any(predictions < 0):
        raise ValueError("finite nonnegative public time-by-station predictions required")
    steps, stations = predictions.shape
    if steps < 1 or xy.shape != (stations, 2) or fixed_lag < 0:
        raise ValueError("invalid wrapper dimensions/lag")
    if min(delta, correction_clip, lambda_temporal) <= 0 or lambda_spatial < 0:
        raise ValueError("invalid wrapper regularization")
    # Reuse the production fixed-lag solver without inventing grid links between
    # irregular stations: padded vertices have NO spatial links or observations.
    side = int(np.ceil(np.sqrt(stations)))
    padding = side * side - stations
    graph = city_graph(xy, neighbors=4)
    laplacian = sparse.diags(np.asarray(graph.sum(axis=1)).ravel()) - graph
    laplacian = sparse.block_diag((laplacian, sparse.csr_matrix((padding, padding))), format="csr")
    padded = np.pad(predictions, ((0, 0), (0, padding)))
    twin = FixedLagTwin(side=side, steps=steps, fixed_lag=fixed_lag, huber=True, delta=delta,
                        lambda_prior=lambda_temporal, lambda_spatial=lambda_spatial,
                        max_irls=12, tolerance=1e-5, initial_state=padded[0].copy(),
                        predictive_residual=True, correction_delta=delta,
                        correction_clip=correction_clip, spatial_laplacian=laplacian)
    arrivals: list[list[Observation]] = [[] for _ in range(steps)]
    seen = set()
    for item in observations:
        if not 0 <= item.cell < stations or not 0 <= item.epoch < steps:
            raise ValueError("observation outside the public field")
        if item.nullifier in seen:
            continue
        seen.add(item.nullifier)
        arrival = item.relay_arrival
        if arrival is not None and arrival < item.epoch:
            raise ValueError("an observation cannot arrive before acquisition")
        if arrival is not None and arrival < steps:
            arrivals[arrival].append(item)
    online = np.empty_like(predictions)
    for epoch in range(steps):
        start = max(0, epoch - fixed_lag)
        twin.ingest_at(epoch, arrivals[epoch], external_predictor=padded[start : epoch + 1])
        # Freeze the live estimate. Later fixed-lag revisions must not improve
        # an earlier live-time RMSE with information that had not arrived yet.
        online[epoch] = twin.states[epoch, :stations]
    return online, {
        "solver_failure_rate": float(np.mean([not item.converged for item in twin.diagnostics])),
        "maximum_correction": float(np.max(np.abs(online - predictions))),
        "retrospective_records": len(twin.retrospective_records),
        "unique_input_records": len(seen), "padded_isolated_vertices": padding,
        "weather_operator": "identity: author pollution archive does not include observed meteorology",
        "estimate_clock": "live arrival epoch; no retrospective estimate substituted",
        "public_predictor_updated_from_citizen": False,
    }


def synthetic_citizen_replay(
    truth: np.ndarray, *, seed: int, kind: str = "clean", reports_per_station: int = 3,
    sigma: float = 3., availability: float = .6, attack_fraction: float = .2,
    attack_amplitude: float = 12.,
) -> list[Observation]:
    """Synthetic independent citizen channel over a real public concentration field.

    Independent RNG streams ensure matched noise, participation, delay and attacker
    identities across clean/drift/negative-bias scenarios. These are not real private
    measurements, nor independent real-world datasets.
    """
    field = np.asarray(truth, dtype=float)
    if field.ndim != 2 or not np.isfinite(field).all() or np.any(field < 0):
        raise ValueError("finite nonnegative truth required")
    if kind not in ("clean", "adversarial_drift", "negative_bias"):
        raise ValueError("unknown replay attack")
    if reports_per_station < 1 or sigma <= 0 or not 0 < availability <= 1 or not 0 <= attack_fraction <= 1:
        raise ValueError("invalid replay channel")
    rngs = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    noise_rng, presence_rng, delay_rng, attack_rng = rngs
    steps, stations = field.shape
    agents = stations * reports_per_station
    attackers = set(attack_rng.choice(agents, size=round(agents * attack_fraction), replace=False))
    noise = noise_rng.normal(0, sigma, (steps, agents))
    present = presence_rng.random((steps, agents)) < availability
    delays = delay_rng.choice([0, 1, 2, 8], size=(steps, agents), p=[.7, .2, .08, .02])
    observations = []
    for epoch, user in zip(*np.nonzero(present), strict=True):
        station = user // reports_per_station
        value = field[epoch, station] + noise[epoch, user]
        corrupted = kind != "clean" and user in attackers and epoch >= steps // 2
        if corrupted and kind == "adversarial_drift":
            value += attack_amplitude * min(1., (epoch - steps // 2 + 1) / 24.)
        elif corrupted:
            value -= attack_amplitude
        arrival = int(epoch + delays[epoch, user])
        observations.append(Observation(int(user), int(epoch), int(station), int(station % 4),
            float(max(0, value)), sigma, 1., 512, f"replay-{seed}-{user}-{epoch}",
            arrival, arrival, bool(corrupted)))
    return observations

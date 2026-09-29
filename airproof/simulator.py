from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .field import SyntheticFieldMechanism, policy_groups, synthetic_field_with_mechanism
from .meteorology import PublicMeteorology, synthetic_public_meteorology
from .records import Observation, contribution_nullifier
from .rng import rng_streams, seed_map


@dataclass(frozen=True)
class SyntheticWorld:
    seed: int
    truth: np.ndarray
    cell_groups: np.ndarray
    observations: tuple[Observation, ...]
    reference_observations: tuple[Observation, ...]
    metadata: dict[str, Any]
    public_meteorology: PublicMeteorology | None = None
    physical_mechanism: SyntheticFieldMechanism | None = None


def _random_walks(side: int, steps: int, agents: int, rng: np.random.Generator) -> np.ndarray:
    cells = np.empty((steps, agents), dtype=int)
    rows = rng.integers(0, side, size=agents)
    cols = rng.integers(0, side, size=agents)
    cells[0] = rows * side + cols
    moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0]])
    for epoch in range(1, steps):
        delta = moves[rng.integers(0, len(moves), size=agents)]
        rows = np.clip(rows + delta[:, 0], 0, side - 1)
        cols = np.clip(cols + delta[:, 1], 0, side - 1)
        cells[epoch] = rows * side + cols
    return cells


def _connectivity(
    agent_groups: np.ndarray,
    steps: int,
    availability: float,
    correlation: float,
    rng: np.random.Generator,
    outage_median_steps: float = 1.0,
    outage_p95_steps: float = 1.0,
) -> np.ndarray:
    """Generate temporally persistent, partially group-correlated connectivity.

    The previous implementation drew a fresh Bernoulli state every epoch, so the
    registered outage-duration parameters had no effect.  This alternating-renewal
    model preserves the requested long-run availability while calibrating offline
    runs to the configured median and p95.
    """
    agents = len(agent_groups)
    if availability >= 1.0 or outage_median_steps <= 0:
        return np.ones((steps, agents), dtype=bool)
    if availability <= 0.0:
        return np.zeros((steps, agents), dtype=bool)

    median = max(float(outage_median_steps), 1.0)
    p95 = max(float(outage_p95_steps), median)
    log_sigma = max(0.0, np.log(p95 / median) / 1.6448536269514722)
    mean_off = median * np.exp(0.5 * log_sigma * log_sigma)
    mean_on = max(1.0, mean_off * availability / max(1.0 - availability, 1e-9))

    def renewal_series() -> np.ndarray:
        series = np.empty(steps, dtype=bool)
        online = bool(rng.random() < availability)
        cursor = 0
        while cursor < steps:
            if online:
                duration = max(1, int(np.ceil(rng.exponential(mean_on))))
            else:
                duration = max(1, round(rng.lognormal(np.log(median), log_sigma)))
            end = min(steps, cursor + duration)
            series[cursor:end] = online
            cursor = end
            online = not online
        return series

    groups = np.unique(agent_groups)
    group_series = {int(group): renewal_series() for group in groups}
    result = np.empty((steps, agents), dtype=bool)
    for agent, group in enumerate(agent_groups):
        # Choose the correlation source per agent rather than per epoch. Epoch-wise
        # switching fragments an intended 24-hour outage into many one-hour runs.
        result[:, agent] = (
            group_series[int(group)] if rng.random() < correlation else renewal_series()
        )
    return result


def _next_true(connectivity: np.ndarray, epoch: int, agent: int, ttl: int) -> int | None:
    end = min(connectivity.shape[0], epoch + ttl + 1)
    matches = np.flatnonzero(connectivity[epoch:end, agent])
    return None if len(matches) == 0 else int(epoch + matches[0])


def generate_world(config: dict[str, Any], seed: int) -> SyntheticWorld:
    streams = rng_streams(seed)
    world_cfg = config["world"]
    network_cfg = config["network"]
    attack_cfg = config["attack"]
    side = int(world_cfg["grid_side"])
    steps = int(world_cfg["steps"])
    agents = int(world_cfg["agents"])
    groups = int(world_cfg.get("groups", 4))
    cell_groups = policy_groups(side, groups)
    weather = (synthetic_public_meteorology(steps, streams["public_meteorology"])
               if world_cfg.get("public_meteorology", False) else None)
    truth, physical_mechanism = synthetic_field_with_mechanism(
        side,
        steps,
        streams["field"],
        process_noise=float(world_cfg.get("process_noise", 0.35)),
        baseline=float(world_cfg.get("baseline", 12.0)),
        mechanism_horizon=steps + int(config.get("twin", {}).get("fixed_lag", 0)),
    )
    reference_count = min(int(world_cfg.get("reference_station_count", 0)), side * side)
    reference_sigma = float(world_cfg.get("reference_measurement_sigma", 0.5))
    if reference_count < 0 or reference_sigma <= 0:
        raise ValueError("reference station count must be non-negative and sigma positive")
    reference_observations: list[Observation] = []
    if reference_count:
        axis_count = int(np.ceil(np.sqrt(reference_count)))
        axis = np.unique(np.rint(np.linspace(0, side - 1, axis_count)).astype(int))
        reference_cells = [int(row * side + col) for row in axis for col in axis][
            :reference_count
        ]
        for epoch in range(steps):
            for station, cell in enumerate(reference_cells):
                value = float(
                    truth[epoch, cell]
                    + streams["reference"].normal(0.0, reference_sigma)
                )
                nullifier = hashlib.sha256(
                    f"reference:{seed}:{station}:{epoch}".encode()
                ).hexdigest()
                reference_observations.append(
                    Observation(
                        user_id=-(station + 1),
                        epoch=epoch,
                        cell=cell,
                        group=int(cell_groups[cell]),
                        value=value,
                        sigma=reference_sigma,
                        quality=1.0,
                        size_bytes=0,
                        nullifier=nullifier,
                        direct_arrival=epoch,
                        relay_arrival=epoch,
                        source_class="regulatory",
                    )
                )
    paths = _random_walks(side, steps, agents, streams["mobility"])
    agent_groups = cell_groups[paths[0]]
    availability = float(network_cfg.get("availability", 0.75))
    correlation = float(network_cfg.get("group_correlation", 0.5))
    connected = _connectivity(
        agent_groups,
        steps,
        availability,
        correlation,
        streams["outage"],
        outage_median_steps=float(network_cfg.get("outage_median_hours", 1.0)),
        outage_p95_steps=float(network_cfg.get("outage_p95_hours", 1.0)),
    )
    ttl = int(network_cfg.get("ttl_steps", 24))
    contact_probability = float(network_cfg.get("contact_probability", 0.15))
    contact_draws = streams["contact"].random((steps, agents))

    corrupt_fraction = float(attack_cfg.get("fraction", 0.0))
    corrupt_count = round(agents * corrupt_fraction)
    corrupted_agents = set(
        map(int, streams["attacker"].choice(agents, size=corrupt_count, replace=False))
    ) if corrupt_count else set()
    attack_kind = str(attack_cfg.get("kind", "clean"))
    attack_amplitude = float(attack_cfg.get("amplitude", 12.0))
    start_epoch = int(attack_cfg.get("start_epoch", steps // 3))
    base_participation = float(world_cfg.get("participation_rate", 0.65))
    skew = float(world_cfg.get("participation_skew", 1.0))
    measurement_sigma = float(world_cfg.get("measurement_sigma", 2.0))
    clip = float(config["privacy"].get("clip", 50.0))
    record_size = int(network_cfg.get("record_size_bytes", 512))
    secret = hashlib.sha256(f"world-secret:{seed}".encode()).digest()
    observations: list[Observation] = []

    for epoch in range(steps):
        any_connected = bool(np.any(connected[epoch : min(steps, epoch + ttl + 1)]))
        for agent in range(agents):
            cell = int(paths[epoch, agent])
            group = int(cell_groups[cell])
            propensity = base_participation / skew if group == 0 else base_participation
            if streams["participation"].random() >= min(propensity, 1.0):
                continue
            value = float(truth[epoch, cell] + streams["measurement"].normal(0.0, measurement_sigma))
            corrupted = agent in corrupted_agents and epoch >= start_epoch and attack_kind != "clean"
            if corrupted:
                if attack_kind == "gradual_drift":
                    value += attack_amplitude * (epoch - start_epoch + 1) / max(1, steps - start_epoch)
                elif attack_kind == "adversarial_drift":
                    # A direction-fixed positive drift can accidentally correct an
                    # underpredicting model.  This validation-only family instead
                    # uses latent truth to choose the sign that opposes the true
                    # excursion from a predeclared baseline.  It is therefore an
                    # explicitly white-box, bounded stress attack rather than a
                    # model of attacker knowledge in deployment.
                    direction = -1.0 if truth[epoch, cell] >= float(
                        world_cfg.get("baseline", 12.0)
                    ) else 1.0
                    value += (
                        direction
                        * attack_amplitude
                        * (epoch - start_epoch + 1)
                        / max(1, steps - start_epoch)
                    )
                elif attack_kind == "hotspot_suppression":
                    value -= attack_amplitude
                elif attack_kind == "inlier":
                    value += min(attack_amplitude, 1.25 * measurement_sigma)
                else:
                    value += attack_amplitude
            value = float(np.clip(value, -clip, clip))
            direct_arrival = _next_true(connected, epoch, agent, ttl)
            relay_arrival = direct_arrival
            if any_connected:
                end = min(steps, epoch + ttl + 1)
                for future in range(epoch, end):
                    if connected[future, agent]:
                        relay_arrival = future
                        break
                    if contact_draws[future, agent] < contact_probability and np.any(connected[future]):
                        relay_arrival = future
                        break
            observations.append(
                Observation(
                    user_id=agent,
                    epoch=epoch,
                    cell=cell,
                    group=group,
                    value=value,
                    sigma=measurement_sigma,
                    quality=0.85,
                    size_bytes=record_size,
                    nullifier=contribution_nullifier(secret, agent, group, epoch),
                    direct_arrival=direct_arrival,
                    relay_arrival=relay_arrival,
                    corrupted=corrupted,
                )
            )
    off_runs: list[int] = []
    for agent in range(agents):
        run = 0
        for online in connected[:, agent]:
            if not online:
                run += 1
            elif run:
                off_runs.append(run)
                run = 0
        if run:
            off_runs.append(run)
    metadata = {
        "seed_map": seed_map(seed),
        "realized_availability": float(np.mean(connected)),
        "realized_off_median_steps": float(np.median(off_runs)) if off_runs else 0.0,
        "realized_off_p95_steps": float(np.percentile(off_runs, 95)) if off_runs else 0.0,
        "candidate_count": len(observations),
        "corrupted_agents": len(corrupted_agents),
        "reference_station_count": reference_count,
        "reference_observation_count": len(reference_observations),
        "meteorology_source": None if weather is None else weather.source,
        "field_transition": "diffusion-with-moving-plume",
    }
    return SyntheticWorld(
        seed,
        truth,
        cell_groups,
        tuple(observations),
        tuple(reference_observations),
        metadata,
        weather,
        physical_mechanism,
    )

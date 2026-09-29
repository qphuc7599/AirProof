from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SyntheticFieldMechanism:
    """Publicly declared deterministic part of the synthetic state equation."""
    transition: sparse.csr_matrix
    exogenous_forcing: np.ndarray
    source: str = "synthetic-public-plume-forcing-v2"


def grid_laplacian(side: int) -> sparse.csr_matrix:
    if side < 2:
        raise ValueError("side must be at least 2")
    # Correct boundary degrees for a combinatorial grid Laplacian.
    adjacency = sparse.lil_matrix((side * side, side * side))
    for row in range(side):
        for col in range(side):
            index = row * side + col
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + drow, col + dcol
                if 0 <= rr < side and 0 <= cc < side:
                    adjacency[index, rr * side + cc] = 1.0
    adjacency = adjacency.tocsr()
    return sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency


def policy_groups(side: int, group_count: int) -> np.ndarray:
    if group_count < 1 or group_count > side:
        raise ValueError("group_count must lie between 1 and grid_side")
    groups = np.empty(side * side, dtype=int)
    for row in range(side):
        group = min(group_count - 1, (row * group_count) // side)
        groups[row * side : (row + 1) * side] = group
    return groups


def synthetic_field(
    side: int,
    steps: int,
    rng: np.random.Generator,
    *,
    process_noise: float = 0.35,
    baseline: float = 12.0,
) -> np.ndarray:
    """Generate a causal diffusion field with a moving, seed-randomized plume."""
    values, _ = synthetic_field_with_mechanism(
        side, steps, rng, process_noise=process_noise, baseline=baseline)
    return values


def synthetic_field_with_mechanism(
    side: int,
    steps: int,
    rng: np.random.Generator,
    *,
    process_noise: float = 0.35,
    baseline: float = 12.0,
    mechanism_horizon: int | None = None,
) -> tuple[np.ndarray, SyntheticFieldMechanism]:
    """Generate the existing field and retain its pre-state public mechanism.

    The forcing contains only the baseline relaxation and deterministic plume drawn
    before the state trajectory. Process innovations and realized truth are excluded.
    Epoch zero is initialization, so the state equation applies from epoch one.
    """
    mechanism_steps = steps if mechanism_horizon is None else int(mechanism_horizon)
    if (side < 2 or steps < 1 or mechanism_steps < steps
            or not np.isfinite((process_noise, baseline)).all()
            or process_noise < 0 or baseline < 0):
        raise ValueError("valid synthetic field dimensions and parameters required")
    cells = side * side
    laplacian = grid_laplacian(side)
    diffusion = sparse.eye(cells) - 0.08 * laplacian
    transition = (0.90 * diffusion).tocsr()
    coords = np.indices((side, side)).reshape(2, -1).T
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = rng.uniform(14.0, 22.0)
    width = max(1.0, side / 5.0)
    forcing = np.empty((mechanism_steps, cells), dtype=float)
    plumes = np.empty((steps, cells), dtype=float)
    for epoch in range(mechanism_steps):
        center = np.array(
            [
                (side - 1) * (0.5 + 0.32 * np.sin(phase + epoch / max(steps, 1) * 2 * np.pi)),
                (side - 1) * (0.5 + 0.32 * np.cos(phase + epoch / max(steps, 1) * 2 * np.pi)),
            ]
        )
        plume = amplitude * np.exp(-np.sum((coords - center) ** 2, axis=1) / (2 * width**2))
        if epoch < steps:
            plumes[epoch] = plume
        forcing[epoch] = 0.10 * baseline + 0.22 * plume
    # Materialize the entire declared forcing before the first realized state.
    values = np.empty((steps, cells), dtype=float)
    values[0] = baseline + rng.normal(0.0, process_noise, cells)
    for epoch in range(steps):
        if epoch == 0:
            values[epoch] += plumes[epoch]
        else:
            innovations = rng.normal(0.0, process_noise, cells)
            values[epoch] = transition @ values[epoch - 1] + forcing[epoch] + innovations
        values[epoch] = np.maximum(values[epoch], 0.0)
    forcing.setflags(write=False)
    return values, SyntheticFieldMechanism(transition, forcing)


def deterministic_diffusion_fixture(side: int, steps: int) -> np.ndarray:
    cells = side * side
    laplacian = grid_laplacian(side)
    transition = sparse.eye(cells) - 0.05 * laplacian
    values = np.empty((steps, cells))
    values[0] = 10.0
    values[0, (side // 2) * side + side // 2] = 30.0
    for epoch in range(1, steps):
        values[epoch] = transition @ values[epoch - 1]
    return values

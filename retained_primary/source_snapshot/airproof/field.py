from __future__ import annotations

import numpy as np
from scipy import sparse


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
    cells = side * side
    laplacian = grid_laplacian(side)
    transition = sparse.eye(cells) - 0.08 * laplacian
    coords = np.indices((side, side)).reshape(2, -1).T
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = rng.uniform(14.0, 22.0)
    width = max(1.0, side / 5.0)
    values = np.empty((steps, cells), dtype=float)
    values[0] = baseline + rng.normal(0.0, process_noise, cells)
    for epoch in range(steps):
        center = np.array(
            [
                (side - 1) * (0.5 + 0.32 * np.sin(phase + epoch / max(steps, 1) * 2 * np.pi)),
                (side - 1) * (0.5 + 0.32 * np.cos(phase + epoch / max(steps, 1) * 2 * np.pi)),
            ]
        )
        plume = amplitude * np.exp(-np.sum((coords - center) ** 2, axis=1) / (2 * width**2))
        if epoch == 0:
            values[epoch] += plume
        else:
            innovations = rng.normal(0.0, process_noise, cells)
            values[epoch] = 0.90 * (transition @ values[epoch - 1]) + 0.10 * baseline + 0.22 * plume + innovations
        values[epoch] = np.maximum(values[epoch], 0.0)
    return values


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

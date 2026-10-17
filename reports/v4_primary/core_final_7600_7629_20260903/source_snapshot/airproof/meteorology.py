"""Public weather covariates and stable, directed spatial transition operators.

Coordinates are (north, east); wind components are eastward u and northward v,
not the meteorological direction *from* which wind blows. Operators map the
previous field to the current field and have nonnegative, unit-sum rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class PublicMeteorology:
    wind_u: np.ndarray
    wind_v: np.ndarray
    temperature: np.ndarray
    humidity: np.ndarray
    source: str

    def validate(self, steps: int) -> None:
        for values in (self.wind_u, self.wind_v, self.temperature, self.humidity):
            if np.asarray(values).shape != (steps,) or not np.all(np.isfinite(values)):
                raise ValueError("weather must contain one finite observation per epoch")
        if np.any(self.humidity < 0) or np.any(self.humidity > 100):
            raise ValueError("relative humidity must lie in [0, 100]")
        if not self.source:
            raise ValueError("weather provenance is required")


def synthetic_public_meteorology(steps: int, rng: np.random.Generator) -> PublicMeteorology:
    """Synthetic public covariates: never label these as observed/ERA5 weather."""
    t = np.arange(steps, dtype=float)
    phase = rng.uniform(0, 2 * np.pi)
    direction = phase + .6 * np.sin(t / 36) + .2 * np.sin(t / 7)
    speed = np.maximum(.1, 2.5 + np.sin(t / 18) + rng.normal(0, .2, steps))
    return PublicMeteorology(
        speed * np.cos(direction), speed * np.sin(direction),
        22 + 6 * np.sin(2 * np.pi * t / 24 + phase),
        np.clip(65 - 15 * np.sin(2 * np.pi * t / 24 + phase), 0, 100),
        "synthetic-public-weather-v1",
    )


@lru_cache(maxsize=16)
def grid_coordinates(side: int) -> np.ndarray:
    if side < 2:
        raise ValueError("side must be >=2")
    return np.indices((side, side)).reshape(2, -1).T.astype(float) / (side - 1)


def city_graph(coordinates: np.ndarray, *, neighbors: int = 4,
               length_scale: float | None = None) -> sparse.csr_matrix:
    """Symmetric k-nearest-neighbor graph; use training-side calibration only.

    Coordinates should be locally projected or consistently scaled, not raw
    latitude/longitude degrees treated as Euclidean metres.
    """
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.all(np.isfinite(xy)):
        raise ValueError("city graph requires finite two-dimensional coordinates")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    distance = cdist(xy, xy)
    np.fill_diagonal(distance, np.inf)
    if np.any(distance == 0):
        raise ValueError("city graph coordinates must be distinct")
    k = min(neighbors, len(xy) - 1)
    chosen = np.argsort(distance, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(len(xy)), k)
    cols = chosen.ravel()
    scale = float(np.median(distance[rows, cols])) if length_scale is None else length_scale
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("graph length scale must be positive")
    adjacency = sparse.csr_matrix(
        (np.exp(-.5 * (distance[rows, cols] / scale) ** 2), (rows, cols)),
        shape=(len(xy), len(xy)),
    )
    return adjacency.maximum(adjacency.T).tocsr()


def wind_transition(adjacency: sparse.csr_matrix, coordinates: np.ndarray, *,
                    wind_u: float, wind_v: float, temperature: float = 22,
                    humidity: float = 65, transport: float = .12,
                    directionality: float = 1.0) -> sparse.csr_matrix:
    """Fetch upwind neighbors using exp(alpha*cos(angle)); preserve constants.

    This is an explainable regularizing transition, not a calibrated chemical
    transport model. Temperature/humidity modulate the bounded mixing fraction.
    """
    xy = np.asarray(coordinates, dtype=float)
    graph = sparse.csr_matrix(adjacency, dtype=float)
    params = [wind_u, wind_v, temperature, humidity, transport, directionality]
    if (xy.ndim != 2 or xy.shape[1] != 2 or graph.shape != (len(xy), len(xy))
            or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(params))
            or not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0)):
        raise ValueError("invalid weather transition inputs")
    if not 0 <= transport <= 1 or not 0 <= humidity <= 100 or not 0 <= directionality <= 10:
        raise ValueError("invalid weather transition parameter range")
    graph = graph.copy()
    graph.setdiag(0)
    graph.eliminate_zeros()
    row, col = graph.nonzero()
    displacement = xy[row] - xy[col]
    distance = np.linalg.norm(displacement, axis=1)
    if np.any(distance == 0):
        raise ValueError("transition graph has coincident connected nodes")
    speed = float(np.hypot(wind_u, wind_v))
    alignment = (displacement @ np.array([wind_v, wind_u])) / np.maximum(distance * speed, 1e-12)
    directed = sparse.csr_matrix(
        (graph.data * np.exp(directionality * np.tanh(speed / 3) * alignment), (row, col)),
        shape=graph.shape,
    )
    degree = np.asarray(directed.sum(axis=1)).ravel()
    weather_factor = np.clip(1 + .005 * (temperature - 22) - .002 * (humidity - 65), .5, 1.5)
    mixing = float(np.clip(transport * (.5 + .5 * np.tanh(speed / 3)) * weather_factor, 0, 1))
    active = (degree > 0).astype(float)
    return (sparse.diags(1 - mixing * active)
            + mixing * sparse.diags(1 / np.maximum(degree, 1e-12)) @ directed).tocsr()


def transition_series(adjacency: sparse.csr_matrix, coordinates: np.ndarray,
                      weather: PublicMeteorology, *, transport: float = .12,
                      directionality: float = 1.0) -> tuple[sparse.csr_matrix, ...]:
    weather.validate(len(weather.wind_u))
    return tuple(wind_transition(
        adjacency, coordinates, wind_u=float(weather.wind_u[t]), wind_v=float(weather.wind_v[t]),
        temperature=float(weather.temperature[t]), humidity=float(weather.humidity[t]),
        transport=transport, directionality=directionality,
    ) for t in range(len(weather.wind_u)))

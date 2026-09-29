"""Matched fixed-lag estimators for the v5 diagnostic and locked campaigns.

The public field is immutable. Residual regularization acts on corrections, not
on the public signal itself. No observation is moved to its arrival timestamp.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from .meteorology import city_graph
from .records import Observation


@dataclass(frozen=True)
class EstimatorConfig:
    objective: str = "residual"
    delta: float = 4.0
    cap: float = 8.0
    lag: int = 6
    regularization_multiplier: float = 1.0
    lambda_temporal: float = .5
    lambda_spatial: float = .2
    lambda_zero: float = .5
    input_clip: bool = True
    output_cap: bool = True
    loss: str = "quadratic"
    huber_delta: float = 1.345
    tolerance: float = 1e-7
    max_irls: int = 20

    def __post_init__(self):
        if self.objective not in ("residual", "full_field") or self.loss not in ("quadratic", "huber"):
            raise ValueError("unknown objective/loss")
        numeric = (self.delta, self.cap, self.regularization_multiplier,
                   self.lambda_temporal, self.lambda_zero, self.huber_delta, self.tolerance)
        if not np.isfinite(numeric).all() or min(numeric) <= 0 or self.lag < 0:
            raise ValueError("positive finite estimator parameters and nonnegative lag required")
        if not np.isfinite(self.lambda_spatial) or self.lambda_spatial < 0 or self.max_irls < 1:
            raise ValueError("invalid spatial penalty/iteration budget")

    @property
    def candidate_id(self) -> str:
        return f"{self.objective}_reg{self.regularization_multiplier:g}_cap{self.cap:g}"


@dataclass
class EstimatorResult:
    live: np.ndarray
    reconstructed: np.ndarray
    diagnostics: dict


def enumerate_candidates() -> list[EstimatorConfig]:
    """The registered twelve candidates; controls are not selection candidates."""
    return [EstimatorConfig(objective=objective, regularization_multiplier=reg, cap=cap)
            for objective in ("full_field", "residual") for reg in (.1, .3, 1.) for cap in (8., 16.)]


def estimate_public_field(public_predictions: np.ndarray, coordinates: np.ndarray,
                          observations: Iterable[Observation],
                          config: EstimatorConfig | None = None, *,
                          laplacian: sparse.spmatrix | None = None,
                          spatial_laplacian: sparse.spmatrix | None = None,
                          transitions: Sequence[sparse.spmatrix] | None = None,
                          arrival_map: Mapping[str, int | None] | None = None) -> EstimatorResult:
    config = EstimatorConfig() if config is None else config
    public = np.asarray(public_predictions, dtype=float)
    if public.ndim != 2 or min(public.shape) < 1 or not np.isfinite(public).all() or np.any(public < 0):
        raise ValueError("finite nonnegative time-by-cell public field required")
    steps, cells = public.shape
    if laplacian is not None and spatial_laplacian is not None:
        raise ValueError("provide only one laplacian argument")
    laplacian = spatial_laplacian if spatial_laplacian is not None else laplacian
    if laplacian is None:
        if cells == 1:
            lap = sparse.csr_matrix((1, 1))
        else:
            adjacency = city_graph(np.asarray(coordinates, float), neighbors=4)
            lap = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    else:
        lap = sparse.csr_matrix(laplacian)
    if lap.shape != (cells, cells) or not np.isfinite(lap.data).all():
        raise ValueError("invalid laplacian")
    operators = None if transitions is None else [sparse.csr_matrix(op) for op in transitions]
    if operators is not None and (len(operators) != steps or any(
            op.shape != (cells, cells) or not np.isfinite(op.data).all() for op in operators)):
        raise ValueError("one finite transition per epoch required")
    arrivals: list[list[Observation]] = [[] for _ in range(steps)]
    seen: set[str] = set()
    for item in observations:
        if not (0 <= item.cell < cells and 0 <= item.epoch < steps):
            raise ValueError("observation outside public field")
        if not np.isfinite((item.value, item.sigma, item.quality)).all() or item.sigma <= 0:
            raise ValueError("finite observations and positive sigma required")
        arrival = item.relay_arrival if arrival_map is None else arrival_map.get(item.nullifier)
        if arrival is not None and (not isinstance(arrival, (int, np.integer)) or arrival < item.epoch):
            raise ValueError("arrival precedes acquisition")
        if item.nullifier in seen:
            continue
        seen.add(item.nullifier)
        if arrival is not None and arrival < steps:
            arrivals[arrival].append(item)
    known: list[list[tuple[int, float, float, float]]] = [[] for _ in range(steps)]
    states = public.copy()
    live = np.empty_like(public)
    clip_count = cap_count = processed = retrospective = failures = 0
    maximum_input_residual = 0.
    terms = []
    bases: dict[int, sparse.csr_matrix] = {}
    scale = config.regularization_multiplier
    lt, ls = scale * config.lambda_temporal, scale * config.lambda_spatial
    lz = scale * config.lambda_zero if config.objective == "residual" else 0.
    identity = sparse.eye(cells, format="csr")
    for epoch in range(steps):
        start = max(0, epoch - config.lag)
        for item in arrivals[epoch]:
            if item.epoch < start:
                retrospective += 1
                continue
            residual = item.value - public[item.epoch, item.cell]
            radius = config.delta * item.sigma
            clip_input = config.input_clip and item.source_class == "citizen"
            bounded = float(np.clip(residual, -radius, radius)) if clip_input else residual
            clip_count += int(clip_input and abs(residual) > radius)
            processed += 1
            maximum_input_residual = max(maximum_input_residual, abs(bounded))
            value = bounded if config.objective == "residual" else public[item.epoch, item.cell] + bounded
            known[item.epoch].append((item.cell, value, item.sigma, float(np.clip(item.quality, 1e-3, 1))))
        window = epoch - start + 1
        cache_key = window if operators is None else epoch + steps
        if cache_key not in bases:
            # D includes the first difference to a fixed pre-window boundary.
            difference = sparse.diags([np.ones(window), -np.ones(max(0, window-1))], [0, -1],
                                      shape=(window, window), format="csr")
            if operators is None:
                temporal = sparse.kron(difference.T @ difference, identity)
            else:
                blocks = [[None for _ in range(window)] for _ in range(window)]
                for local in range(window):
                    blocks[local][local] = identity
                    if local > 0:
                        blocks[local][local-1] = -operators[start+local]
                operator_difference = sparse.bmat(blocks, format="csr")
                temporal = operator_difference.T @ operator_difference
            bases[cache_key] = (lt * temporal
                               + sparse.kron(sparse.eye(window), ls * lap + lz * identity)).tocsr()
        boundary = (np.zeros(cells) if config.objective == "residual" else public[0]) if start == 0 else states[start-1].copy()
        if start > 0 and config.objective == "residual":
            boundary -= public[start-1]
        if operators is not None:
            boundary = operators[start] @ boundary
        initial = states[start:epoch+1].copy()
        if config.objective == "residual":
            initial -= public[start:epoch+1]
        state = initial.ravel()
        obs = [(local*cells+cell, value, sigma, quality)
               for local, rows in enumerate(known[start:epoch+1]) for cell, value, sigma, quality in rows]
        indices = np.array([row[0] for row in obs], dtype=int)
        values = np.array([row[1] for row in obs])
        sigmas = np.array([row[2] for row in obs])
        quality = np.array([row[3] for row in obs])
        converged = False
        for iteration in range(config.max_irls if config.loss == "huber" else 1):
            weights = quality / sigmas**2
            if config.loss == "huber":
                weights *= np.minimum(1., config.huber_delta / np.maximum(abs((values-state[indices])/sigmas), 1e-12))
            diagonal = np.bincount(indices, weights=weights, minlength=window*cells).astype(float)
            # NumPy returns integer bincounts for an empty weighted input.
            rhs = np.bincount(indices, weights=weights*values, minlength=window*cells).astype(float)
            rhs[:cells] += lt * boundary
            updated, info = cg(bases[cache_key] + sparse.diags(diagonal), rhs, x0=state,
                               rtol=config.tolerance, atol=1e-10, maxiter=max(200, window*cells*3))
            change = np.linalg.norm(updated-state)/(np.linalg.norm(state)+1e-12)
            state = updated
            if info == 0 and (config.loss == "quadratic" or change < config.tolerance):
                converged = True
                break
        failures += int(not converged)
        solved = state.reshape(window, cells)
        raw = public[start:epoch+1]+solved if config.objective == "residual" else solved
        correction = raw-public[start:epoch+1]
        cap_count += int(np.count_nonzero(abs(correction[-1]) > config.cap)) if config.output_cap else 0
        if config.output_cap:
            correction = np.clip(correction, -config.cap, config.cap)
        states[start:epoch+1] = np.maximum(public[start:epoch+1]+correction, 0.)
        live[epoch] = states[epoch]
        data_term = float(np.sum(quality*((values-state[indices])/sigmas)**2))
        if config.loss == "huber":
            standardized = abs((values-state[indices])/sigmas)
            data_term = float(np.sum(quality*np.where(standardized <= config.huber_delta,
                standardized**2, 2*config.huber_delta*standardized-config.huber_delta**2)))
        terms.append({"epoch": epoch, "data_loss_twice": data_term,
                      "regularization_quadratic_twice": float(state @ (bases[cache_key] @ state)
                                                               - 2*lt*boundary @ state[:cells]
                                                               + lt*boundary @ boundary),
                      "max_live_correction": float(np.max(abs(live[epoch]-public[epoch]))),
                      "irls_iterations": iteration+1})
        if operators is not None:
            del bases[cache_key]
    return EstimatorResult(live, states.copy(), {
        "config": asdict(config), "unique_input_records": len(seen),
        "assimilated_records": processed, "retrospective_records": retrospective,
        "input_clip_count": clip_count, "input_clip_fraction": clip_count/max(processed, 1),
        "output_cap_live_count": cap_count, "output_cap_live_fraction": cap_count/(steps*cells),
        "maximum_bounded_input_residual": maximum_input_residual,
        "maximum_correction": float(np.max(abs(live-public))),
        "solver_failure_rate": failures/steps, "epoch_terms": terms,
        "estimate_clock": "frozen live arrival epoch and separate final reconstruction",
        "public_predictor_updated_from_citizen": False,
    })

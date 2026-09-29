"""Bound-constrained Huber residual twin; public context is supplied, never fitted."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from .meteorology import city_graph
from .records import Observation
from .v5_estimator import EstimatorResult
from .v6_solver_refinement import refine_to_kkt


@dataclass(frozen=True)
class EstimatorConfig:
    cap: float = 8.0
    lag: int = 6
    huber_delta: float = 1.345
    lambda_temporal: float = .5
    lambda_spatial: float = .2
    lambda_zero: float = .5
    time_step: float = 1.0
    precision_normalizer: float = 1.0
    tolerance: float = 1e-7
    max_iterations: int = 1000
    numerical_refinements: int = 0
    loss: str = "huber"
    consensus_spread_threshold: float = 1.5
    consensus_singleton_quadratic: bool = False
    output_cap: bool = True
    input_clip: bool = False
    clip_delta: float = 4.0
    history_normalized_huber: bool = False
    per_user_weight_budget: float = 1.0

    def __post_init__(self):
        positive = (self.cap, self.huber_delta, self.lambda_zero, self.time_step,
                    self.precision_normalizer, self.tolerance, self.clip_delta,
                    self.per_user_weight_budget)
        nonnegative = (self.lambda_temporal, self.lambda_spatial)
        if (not np.isfinite(positive + nonnegative).all() or min(positive) <= 0
                or min(nonnegative) < 0 or not isinstance(self.lag, int) or self.lag < 0
                or not isinstance(self.max_iterations, int) or self.max_iterations < 1
                or not isinstance(self.numerical_refinements,int) or self.numerical_refinements<0
                or not isinstance(self.consensus_singleton_quadratic, bool)
                or not isinstance(self.history_normalized_huber, bool)
                or not np.isfinite(self.consensus_spread_threshold)
                or self.consensus_spread_threshold <= 0
                or self.loss not in ("huber", "quadratic", "consensus_huber")):
            raise ValueError("invalid constrained estimator configuration")
        if self.history_normalized_huber and self.loss != "huber":
            raise ValueError("history normalization is defined only for ordinary Huber loss")


def huber_value_gradient(residual: np.ndarray, delta: float):
    """Standard Huber loss and bounded score (not input clipping)."""
    absolute = np.abs(residual)
    return (np.where(absolute <= delta, .5 * residual**2,
                     delta * (absolute - .5 * delta)), np.clip(residual, -delta, delta))


def history_normalized_weights(user_ids: np.ndarray, raw_weights: np.ndarray,
                               per_user_budget: float) -> np.ndarray:
    """Bound each user's total weight in one fixed optimization window."""
    users=np.asarray(user_ids)
    weights=np.asarray(raw_weights,float)
    if users.ndim!=1 or weights.shape!=users.shape or not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError("aligned user ids and finite nonnegative weights required")
    if not np.isfinite(per_user_budget) or per_user_budget<=0:
        raise ValueError("per-user weight budget must be positive")
    result=weights.copy()
    for user in np.unique(users):
        mask=users==user;total=float(weights[mask].sum())
        if total>per_user_budget:result[mask]*=per_user_budget/total
    return result


def history_normalized_sensitivity_certificate(*, strong_convexity_mu: float,
        huber_delta: float, scale_floor: float, per_user_weight_budget: float) -> dict[str,float]:
    """Fixed-window replacement bound; it makes no recursive horizon claim."""
    values=(strong_convexity_mu,huber_delta,scale_floor,per_user_weight_budget)
    if not np.isfinite(values).all() or min(values)<=0:
        raise ValueError("positive finite certificate inputs required")
    gradient_bound=per_user_weight_budget*huber_delta/scale_floor
    return {"strong_convexity_mu":float(strong_convexity_mu),
        "huber_delta":float(huber_delta),"scale_floor":float(scale_floor),
        "per_user_weight_budget":float(per_user_weight_budget),
        "one_history_gradient_bound":float(gradient_bound),
        "replacement_gradient_bound":float(2*gradient_bound),
        "fixed_window_minimizer_l2_bound":float(2*gradient_bound/strong_convexity_mu)}


def estimate_public_field(public_predictions: np.ndarray, coordinates: np.ndarray,
                          observations: Iterable[Observation],
                          config: EstimatorConfig | None = None, *,
                          innovation_scales: np.ndarray | float,
                          laplacian: sparse.spmatrix | None = None,
                          spatial_laplacian: sparse.spmatrix | None = None,
                          transitions: Sequence[sparse.spmatrix] | None = None,
                          residual_forcing: np.ndarray | None = None,
                          arrival_map: Mapping[str, int | None] | None = None) -> EstimatorResult:
    config = config or EstimatorConfig()
    public = np.array(public_predictions, dtype=float, copy=True)
    if public.ndim != 2 or min(public.shape) < 1 or not np.isfinite(public).all() or (public < 0).any():
        raise ValueError("finite nonnegative time-by-cell public field required")
    steps, cells = public.shape
    scales = np.broadcast_to(np.asarray(innovation_scales, float), public.shape).copy()
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("positive finite frozen public innovation scales required")
    forcing = (np.zeros_like(public) if residual_forcing is None
               else np.array(residual_forcing, float, copy=True))
    if forcing.shape != public.shape or not np.isfinite(forcing).all():
        raise ValueError("finite time-by-cell residual forcing required")
    identity = sparse.eye(cells, format="csr")
    operators = ([identity] * steps if transitions is None
                 else [sparse.csr_matrix(op) for op in transitions])
    if len(operators) != steps or any(op.shape != (cells, cells) or
                                    not np.isfinite(op.data).all() for op in operators):
        raise ValueError("one finite transition per epoch required")
    if laplacian is not None and spatial_laplacian is not None:
        raise ValueError("provide only one laplacian")
    lap = spatial_laplacian if spatial_laplacian is not None else laplacian
    if lap is None:
        adjacency = (sparse.csr_matrix((1, 1)) if cells == 1
                     else city_graph(np.asarray(coordinates, float), neighbors=4))
        lap = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    lap = sparse.csr_matrix(lap, dtype=float)
    if lap.shape != (cells, cells) or not np.isfinite(lap.data).all():
        raise ValueError("invalid laplacian")
    # Require a symmetric graph Laplacian; this establishes PSD without dense eigensolves.
    off = lap - sparse.diags(lap.diagonal())
    if ((lap - lap.T).nnz and np.max(np.abs((lap - lap.T).data)) > 1e-10
            or (off.data > 1e-10).any()
            or not np.allclose(np.asarray(lap.sum(axis=1)), 0, atol=1e-10)):
        raise ValueError("symmetric graph laplacian required")
    arrivals = [[] for _ in range(steps)]
    for order, item in enumerate(observations):
        if not (0 <= item.cell < cells and 0 <= item.epoch < steps):
            raise ValueError("observation outside public field")
        if not np.isfinite((item.value, item.quality)).all() or not 0 < item.quality <= 1:
            raise ValueError("finite value and quality in (0,1] required")
        if (config.history_normalized_huber and
                not isinstance(item.user_id, (int, np.integer))):
            raise ValueError("history normalization requires an integer user partition")
        arrival = item.relay_arrival if arrival_map is None else arrival_map.get(item.nullifier)
        if arrival is not None and (not isinstance(arrival, (int, np.integer)) or arrival < item.epoch):
            raise ValueError("arrival precedes acquisition")
        if arrival is not None and arrival < steps:
            arrivals[arrival].append((order, item))
    known = [[] for _ in range(steps)]
    states, live = np.zeros_like(public), np.empty_like(public)
    seen = set()
    processed = late = duplicates = failures = 0
    terms = []
    lt = config.lambda_temporal / config.time_step
    spatial = config.time_step * (config.lambda_spatial * lap + config.lambda_zero * identity)
    for epoch in range(steps):
        start = max(0, epoch - config.lag)
        for _, item in arrivals[epoch]:
            if item.nullifier in seen:
                duplicates += 1
                continue
            seen.add(item.nullifier)
            if item.epoch < start:
                late += 1
                continue
            known[item.epoch].append(item)
            processed += 1
        window = epoch - start + 1
        blocks = [[None] * window for _ in range(window)]
        for local in range(window):
            blocks[local][local] = identity
            if local:
                blocks[local][local-1] = -operators[start+local]
        difference = sparse.bmat(blocks, format="csr")
        target = forcing[start:epoch+1].copy()
        if start:
            target[0] += operators[start] @ states[start-1]
        target = target.ravel()
        quadratic = (lt * difference.T @ difference
                     + sparse.kron(sparse.eye(window), spatial, format="csr"))
        rhs = lt * (difference.T @ target)
        if config.loss == "consensus_huber":
            rows = []
            for local, batch in enumerate(known[start:epoch+1]):
                grouped = {}
                for observation in batch:
                    grouped.setdefault(observation.cell, []).append(observation)
                for cell, group in sorted(grouped.items()):
                    group_values = np.array([o.value-public[o.epoch, cell] for o in group])
                    group_quality = np.array([o.quality for o in group])
                    group_scale = scales[group[0].epoch, cell]
                    median = float(np.median(group_values))
                    normalized_spread = float(np.max(np.abs(group_values-median))/group_scale)
                    consensus = ((len(group) >= 2
                                  and normalized_spread <= config.consensus_spread_threshold)
                                 or (len(group) == 1
                                     and config.consensus_singleton_quadratic))
                    if consensus:
                        center = float(np.average(group_values, weights=group_quality))
                        weight = float(group_quality.sum())
                    else:
                        center = median
                        weight = float(max(group_quality.max(),
                                           2*group_quality.sum()/np.pi))
                    rows.append((local*cells+cell, center, group_scale, weight, consensus,
                                 len(group), normalized_spread, None))
        else:
            rows = [(local*cells+o.cell, o.value-public[o.epoch, o.cell],
                     scales[o.epoch, o.cell], o.quality, False, 1, 0., o.user_id)
                    for local, batch in enumerate(known[start:epoch+1]) for o in batch]
        indices = np.array([r[0] for r in rows], int)
        values = np.array([r[1] for r in rows])
        sigma = np.array([r[2] for r in rows])
        weights = np.array([r[3] for r in rows]) / config.precision_normalizer
        consensus_rows = np.array([r[4] for r in rows], bool)
        if config.history_normalized_huber:
            weights=history_normalized_weights(np.array([r[7] for r in rows],int),weights,
                                               config.per_user_weight_budget)
        if config.input_clip:
            values = np.clip(values, -config.clip_delta*sigma, config.clip_delta*sigma)

        def objective(z, indices=indices, values=values, sigma=sigma,
                      quadratic=quadratic, rhs=rhs, weights=weights):
            residual = (z[indices]-values)/sigma
            if config.loss == "quadratic":
                loss, score = .5*residual**2, residual
            else:
                loss, score = huber_value_gradient(residual, config.huber_delta)
                if config.loss == "consensus_huber" and consensus_rows.any():
                    loss = np.where(consensus_rows, .5*residual**2, loss)
                    score = np.where(consensus_rows, residual, score)
            value = .5*z @ (quadratic @ z) - rhs @ z + weights @ loss
            gradient = quadratic @ z - rhs
            np.add.at(gradient, indices, weights*score/sigma)
            return float(value), gradient

        lower = (np.maximum(-config.cap, -public[start:epoch+1]).ravel()
                 if config.output_cap else -public[start:epoch+1].ravel())
        upper = np.full(window*cells, config.cap if config.output_cap else np.inf)
        result = minimize(objective, states[start:epoch+1].ravel(), jac=True,
                          method="L-BFGS-B", bounds=list(zip(lower, upper)),
                          options={"gtol": config.tolerance, "ftol": 1e-15,
                                   "maxiter": config.max_iterations, "maxls": 40})
        refinement_history=[]
        if config.numerical_refinements:
            result.x,refinement_history=refine_to_kkt(objective,result.x,lower,upper,
                tolerance=config.tolerance,max_iterations=config.max_iterations,restarts=config.numerical_refinements)
        gradient = objective(result.x)[1]
        kkt = float(np.max(np.abs(result.x - np.clip(result.x-gradient, lower, upper))))
        failures += int(kkt > 10*config.tolerance)
        states[start:epoch+1] = result.x.reshape(window, cells)
        live[epoch] = public[epoch] + states[epoch]
        maximum_user_weight=0.
        if config.history_normalized_huber and len(rows):
            row_users=np.array([r[7] for r in rows],int)
            maximum_user_weight=max(float(weights[row_users==user].sum()) for user in np.unique(row_users))
        term={"epoch": epoch, "start": start, "projected_gradient_inf": kkt,
              "iterations": int(result.nit), "optimizer_success": bool(result.success),
              "refinement_history": refinement_history}
        if config.history_normalized_huber:
            term["maximum_user_window_weight"]=maximum_user_weight
        terms.append(term)
    diagnostics={
        "config": asdict(config), "assimilated_records": processed,
        "retrospective_records": late, "duplicate_records": duplicates,
        "solver_failure_rate": failures/steps, "epoch_terms": terms,
        "strong_convexity_lower_bound": config.lambda_zero*config.time_step,
        "maximum_correction": float(np.max(np.abs(live-public))),
        "public_predictor_updated_from_citizen": False,
        "scale_source": "caller supplied frozen public context; never private sigma",
        "dynamics": "explicit residual forcing" if residual_forcing is not None else "zero-forcing residual model",
        "observation_rule": ("same-cell/acquisition dispersion-gated mean or median"
                             if config.loss == "consensus_huber" else "individual records"),
        "estimate_clock": "immutable live; lag-inclusive reconstruction; closed prefix",
    }
    if config.history_normalized_huber:
        diagnostics["history_normalized_sensitivity_certificate"] = \
            history_normalized_sensitivity_certificate(
                strong_convexity_mu=config.lambda_zero*config.time_step,
                huber_delta=config.huber_delta,scale_floor=float(scales.min()),
                per_user_weight_budget=config.per_user_weight_budget)
        diagnostics["history_normalization_scope"] = \
            "fixed window and fixed preceding boundary, with collision-free admitted nullifiers; no recursive whole-horizon sensitivity claim"
    return EstimatorResult(live, public+states, diagnostics)


def estimate_matched_controls(public_predictions, coordinates, observations,
                              config: EstimatorConfig | None = None, **kwargs):
    """Five same-information mechanistic controls; no model selection performed.

    Unbounded means no correction envelope; physical nonnegativity remains shared.
    """
    base = config or EstimatorConfig()
    items = tuple(observations)
    public = np.array(public_predictions, float, copy=True)
    arms = {
        "quadratic": replace(base, loss="quadratic", output_cap=False, input_clip=False),
        "clipping_only": replace(base, loss="quadratic", output_cap=False, input_clip=True),
        "cap_only": replace(base, loss="quadratic", output_cap=True, input_clip=False),
        "bounded_huber": replace(base, loss="huber", output_cap=True, input_clip=False),
    }
    results = {name: estimate_public_field(public, coordinates, items, arm, **kwargs)
               for name, arm in arms.items()}
    results["public_only"] = EstimatorResult(public.copy(), public.copy(), {
        "citizen_independent": True, "method": "identical supplied public field"})
    return results


def bounded_development_candidates():
    """Three predeclared regularization choices; cap8 is never relaxed."""
    return {f"huber_reg{scale:g}": EstimatorConfig(
        lambda_temporal=.5*scale, lambda_spatial=.2*scale, lambda_zero=.5*scale)
        for scale in (.1, .3, 1.)}

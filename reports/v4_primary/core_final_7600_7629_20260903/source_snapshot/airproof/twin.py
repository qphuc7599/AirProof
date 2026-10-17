from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from .field import grid_laplacian
from .records import Observation


@dataclass(frozen=True)
class SolverDiagnostics:
    converged: bool
    irls_iterations: int
    cg_failures: int
    relative_change: float


@dataclass(frozen=True)
class AdaptiveHuberState:
    delta: float
    estimated_contamination: float
    residual_location: float
    residual_count: int


def adaptive_huber_state(
    closed_standardized_residuals: Iterable[float],
    *,
    clean_delta: float = 6.0,
    robust_delta: float = 1.345,
    tail_z: float = 2.5,
    clean_tail_fraction: float = 0.02,
    contaminated_tail_fraction: float = 0.15,
) -> AdaptiveHuberState:
    """Choose Huber's transition from residuals closed before the current update.

    The mapping is deliberately monotone: a Gaussian-like tail rate selects the
    clean-efficient transition, while a larger lagged tail rate contracts toward
    the robust transition. Current observations must never be passed here because
    they would then influence their own weight.
    """
    if not 0 < robust_delta <= clean_delta:
        raise ValueError("require 0 < robust_delta <= clean_delta")
    if tail_z <= 0 or not 0 <= clean_tail_fraction < contaminated_tail_fraction <= 1:
        raise ValueError("invalid adaptive Huber calibration")
    values = np.asarray(list(closed_standardized_residuals), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return AdaptiveHuberState(float(clean_delta), 0.0, 0.0, 0)
    tail_fraction = float(np.mean(np.abs(values) > tail_z))
    residual_location = float(np.median(values))
    mixture = float(
        np.clip(
            (tail_fraction - clean_tail_fraction)
            / (contaminated_tail_fraction - clean_tail_fraction),
            0.0,
            1.0,
        )
    )
    delta = float(clean_delta - mixture * (clean_delta - robust_delta))
    return AdaptiveHuberState(delta, tail_fraction, residual_location, int(values.size))


def innovation_contamination_fraction(
    observations: Iterable[Observation],
    predictive_state: np.ndarray,
    *,
    tail_z: float = 3.5,
    minimum_scale: float = 0.25,
) -> float:
    """Robustly estimate the anomalous fraction in a current innovation batch."""
    if tail_z <= 0 or minimum_scale <= 0:
        raise ValueError("tail_z and minimum_scale must be positive")
    residuals = np.asarray(
        [
            (float(item.value) - predictive_state[item.cell])
            / max(float(item.sigma), 1e-6)
            for item in observations
        ],
        dtype=float,
    )
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 4:
        return 0.0
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    scale = max(1.4826 * mad, minimum_scale)
    return float(np.mean(np.abs(residuals - center) > tail_z * scale))


def standardized_innovation_tail_fraction(
    observations: Iterable[Observation],
    predictive_state: np.ndarray,
    *,
    tail_z: float = 3.5,
) -> float:
    """Measure predictive-standardized tail mass without recentering the batch.

    The predictor is fixed before the current citizen batch, so zero is the declared
    innovation center. This detector is intentionally distinct from the robust batch-MAD
    diagnostic above: coordinated shifts should not erase their own signal by moving the
    within-batch center or scale.
    """
    if tail_z <= 0:
        raise ValueError("tail_z must be positive")
    residuals = np.asarray(
        [
            (float(item.value) - predictive_state[item.cell])
            / max(float(item.sigma), 1e-6)
            for item in observations
        ],
        dtype=float,
    )
    residuals = residuals[np.isfinite(residuals)]
    return float(np.mean(np.abs(residuals) > tail_z)) if residuals.size else 0.0


def standardized_irls_weight(
    *, quality: float, sigma: float, residual_standard: float, huber: bool, delta: float
) -> float:
    """Raw-residual WLS precision with variance counted exactly once."""
    sigma = max(float(sigma), 1e-6)
    omega = min(1.0, delta / max(abs(residual_standard), 1e-12)) if huber else 1.0
    return float(np.clip(quality, 1e-3, 1.0)) * omega / (sigma * sigma)


def observation_variance(
    sensor_variance: float,
    calibration_variance: float,
    model_variance: float,
    *,
    dp_laplace_scale: float | None = None,
) -> float:
    """Path A excludes DP noise; Path B adds the known Laplace variance ``2b²``."""
    parts = (sensor_variance, calibration_variance, model_variance)
    if any(value < 0 for value in parts) or (
        dp_laplace_scale is not None and dp_laplace_scale < 0
    ):
        raise ValueError("variance components and DP scale must be non-negative")
    result = float(sum(parts))
    if dp_laplace_scale is not None:
        result += 2.0 * dp_laplace_scale * dp_laplace_scale
    return result


def robust_state_update(
    prior: np.ndarray,
    observations: Iterable[Observation],
    laplacian: sparse.csr_matrix,
    *,
    huber: bool,
    delta: float,
    lambda_prior: float,
    lambda_spatial: float,
    max_irls: int,
    tolerance: float,
) -> tuple[np.ndarray, SolverDiagnostics]:
    """Solve one consistent standardized-residual Huber objective.

    q_tilde contains source quality and delay only. The raw-residual normal
    weight is q_tilde * omega / sigma**2, so precision is counted exactly once.
    """
    obs = list(observations)
    cells = prior.size
    if not obs:
        system = lambda_prior * sparse.eye(cells) + lambda_spatial * laplacian
        rhs = lambda_prior * prior
        value, info = cg(system.tocsr(), rhs, x0=prior, rtol=tolerance, maxiter=max(50, cells * 2))
        return np.maximum(value, 0.0), SolverDiagnostics(info == 0, 0, int(info != 0), 0.0)

    state = prior.astype(float, copy=True)
    cg_failures = 0
    relative_change = float("inf")
    for iteration in range(1, max_irls + 1):
        diagonal = np.zeros(cells, dtype=float)
        rhs_obs = np.zeros(cells, dtype=float)
        for item in obs:
            sigma = max(float(item.sigma), 1e-6)
            residual_standard = (float(item.value) - state[item.cell]) / sigma
            raw_weight = standardized_irls_weight(
                quality=item.quality,
                sigma=sigma,
                residual_standard=residual_standard,
                huber=huber,
                delta=delta,
            )
            diagonal[item.cell] += raw_weight
            rhs_obs[item.cell] += raw_weight * float(item.value)
        system = sparse.diags(diagonal + lambda_prior) + lambda_spatial * laplacian
        rhs = rhs_obs + lambda_prior * prior
        updated, info = cg(system.tocsr(), rhs, x0=state, rtol=tolerance, maxiter=max(100, cells * 3))
        if info != 0:
            cg_failures += 1
        relative_change = float(np.linalg.norm(updated - state) / (np.linalg.norm(state) + 1e-12))
        state = updated
        if relative_change < tolerance:
            return np.maximum(state, 0.0), SolverDiagnostics(info == 0, iteration, cg_failures, relative_change)
    return np.maximum(state, 0.0), SolverDiagnostics(False, max_irls, cg_failures, relative_change)


def robust_trajectory_update(
    boundary: np.ndarray,
    initial_trajectory: np.ndarray,
    observations_by_step: list[list[Observation]],
    laplacian: sparse.csr_matrix,
    *,
    huber: bool,
    delta: float,
    lambda_temporal: float,
    lambda_spatial: float,
    max_irls: int,
    tolerance: float,
    transitions: list[sparse.csr_matrix] | tuple[sparse.csr_matrix, ...] | None = None,
) -> tuple[np.ndarray, SolverDiagnostics]:
    """Jointly optimize a fixed-lag trajectory with a fixed pre-window boundary.

    Unlike a sequence of independent filters, this block solve lets delayed evidence
    update its acquisition-time state and all still-mutable descendants while preserving
    states before the lag window. Temporal penalties are
    lambda_temporal/2 * ||x_s - A_s x_(s-1)||^2, with the first
    x_(s-1) fixed to boundary. ``transitions[0]`` acts on that boundary.
    """
    state = np.asarray(initial_trajectory, dtype=float).copy()
    if state.ndim != 2 or len(observations_by_step) != state.shape[0]:
        raise ValueError("trajectory and observation window lengths must match")
    window, cells = state.shape
    if boundary.shape != (cells,) or laplacian.shape != (cells, cells):
        raise ValueError("boundary/laplacian dimensions do not match trajectory")
    if lambda_temporal <= 0 or lambda_spatial < 0:
        raise ValueError("lambda_temporal must be positive and lambda_spatial non-negative")

    identity = sparse.eye(cells, format="csr")
    operators = [identity] * window if transitions is None else list(transitions)
    if len(operators) != window:
        raise ValueError("one transition per trajectory step is required")
    operators = [sparse.csr_matrix(operator) for operator in operators]
    if any(operator.shape != (cells, cells) or not np.all(np.isfinite(operator.data))
           for operator in operators):
        raise ValueError("invalid trajectory transition dimensions or values")
    spatial = lambda_spatial * laplacian
    cg_failures = 0
    relative_change = float("inf")
    for iteration in range(1, max_irls + 1):
        diagonals = [np.zeros(cells, dtype=float) for _ in range(window)]
        rhs_parts = [np.zeros(cells, dtype=float) for _ in range(window)]
        for local_step, observations in enumerate(observations_by_step):
            for item in observations:
                sigma = max(float(item.sigma), 1e-6)
                residual = (float(item.value) - state[local_step, item.cell]) / sigma
                weight = standardized_irls_weight(
                    quality=item.quality,
                    sigma=sigma,
                    residual_standard=residual,
                    huber=huber,
                    delta=delta,
                )
                diagonals[local_step][item.cell] += weight
                rhs_parts[local_step][item.cell] += weight * float(item.value)

        blocks: list[list[sparse.spmatrix | None]] = [
            [None for _ in range(window)] for _ in range(window)
        ]
        for local_step in range(window):
            blocks[local_step][local_step] = sparse.diags(diagonals[local_step]) + spatial
        blocks[0][0] = blocks[0][0] + lambda_temporal * identity
        rhs_parts[0] += lambda_temporal * (operators[0] @ boundary)
        for local_step in range(1, window):
            operator = operators[local_step]
            blocks[local_step - 1][local_step - 1] = (
                blocks[local_step - 1][local_step - 1]
                + lambda_temporal * (operator.T @ operator)
            )
            blocks[local_step][local_step] = (
                blocks[local_step][local_step] + lambda_temporal * identity
            )
            blocks[local_step - 1][local_step] = -lambda_temporal * operator.T
            blocks[local_step][local_step - 1] = -lambda_temporal * operator

        system = sparse.bmat(blocks, format="csr")
        rhs = np.concatenate(rhs_parts)
        updated_flat, info = cg(
            system,
            rhs,
            x0=state.reshape(-1),
            rtol=tolerance,
            maxiter=max(200, window * cells * 3),
        )
        if info != 0:
            cg_failures += 1
        updated = updated_flat.reshape(window, cells)
        relative_change = float(
            np.linalg.norm(updated - state) / (np.linalg.norm(state) + 1e-12)
        )
        state = updated
        if relative_change < tolerance:
            return (
                np.maximum(state, 0.0),
                SolverDiagnostics(info == 0, iteration, cg_failures, relative_change),
            )
    return (
        np.maximum(state, 0.0),
        SolverDiagnostics(False, max_irls, cg_failures, relative_change),
    )


def robust_score_correction_trajectory(
    backbone: np.ndarray,
    observations_by_step: list[list[Observation]],
    laplacian: sparse.csr_matrix,
    *,
    delta: float,
    lambda_correction: float,
    lambda_temporal: float,
    lambda_spatial: float,
    correction_clip: float,
    tolerance: float,
) -> tuple[np.ndarray, SolverDiagnostics, float]:
    """Compute a bounded one-step robust correction around a squared-loss backbone.

    The backbone retains clean efficiency.  The correction is driven only by the
    difference between the Huber score and the squared score,

        psi_delta(r) - r.

    Consequently the correction is exactly zero when every standardized innovation
    lies in Huber's quadratic region, and it converges to the squared path as
    ``delta`` grows.  Tail observations can only pull the correction opposite to
    their excess squared influence.  A ridge, temporal smoothing, graph smoothing,
    and an explicit elementwise bound keep the corrective layer well posed.
    """
    state = np.asarray(backbone, dtype=float)
    if state.ndim != 2 or len(observations_by_step) != state.shape[0]:
        raise ValueError("backbone and observation window lengths must match")
    window, cells = state.shape
    if laplacian.shape != (cells, cells):
        raise ValueError("laplacian dimensions do not match backbone")
    if delta <= 0 or lambda_correction <= 0 or lambda_temporal < 0:
        raise ValueError("delta/lambda_correction must be positive and lambda_temporal non-negative")
    if lambda_spatial < 0 or correction_clip <= 0 or tolerance <= 0:
        raise ValueError("invalid correction regularization, bound, or tolerance")

    diagonals = [np.zeros(cells, dtype=float) for _ in range(window)]
    rhs_parts = [np.zeros(cells, dtype=float) for _ in range(window)]
    tail_count = 0
    residual_count = 0
    for local_step, observations in enumerate(observations_by_step):
        for item in observations:
            sigma = max(float(item.sigma), 1e-6)
            quality = float(np.clip(item.quality, 1e-3, 1.0))
            residual = (float(item.value) - state[local_step, item.cell]) / sigma
            psi = float(np.clip(residual, -delta, delta))
            diagonals[local_step][item.cell] += quality / (sigma * sigma)
            rhs_parts[local_step][item.cell] += quality * (psi - residual) / sigma
            tail_count += int(abs(residual) > delta)
            residual_count += 1

    identity = sparse.eye(cells, format="csr")
    spatial = lambda_spatial * laplacian
    blocks: list[list[sparse.spmatrix | None]] = [
        [None for _ in range(window)] for _ in range(window)
    ]
    for local_step in range(window):
        blocks[local_step][local_step] = (
            sparse.diags(diagonals[local_step] + lambda_correction) + spatial
        )
    if lambda_temporal > 0:
        # The correction is zero immediately before the mutable lag window.
        blocks[0][0] = blocks[0][0] + lambda_temporal * identity
        for local_step in range(1, window):
            blocks[local_step - 1][local_step - 1] = (
                blocks[local_step - 1][local_step - 1] + lambda_temporal * identity
            )
            blocks[local_step][local_step] = (
                blocks[local_step][local_step] + lambda_temporal * identity
            )
            blocks[local_step - 1][local_step] = -lambda_temporal * identity
            blocks[local_step][local_step - 1] = -lambda_temporal * identity

    system = sparse.bmat(blocks, format="csr")
    rhs = np.concatenate(rhs_parts)
    correction_flat, info = cg(
        system,
        rhs,
        x0=np.zeros(window * cells, dtype=float),
        rtol=tolerance,
        maxiter=max(200, window * cells * 3),
    )
    correction = np.clip(
        correction_flat.reshape(window, cells), -correction_clip, correction_clip
    )
    relative_change = float(np.linalg.norm(correction) / (np.linalg.norm(state) + 1e-12))
    tail_fraction = tail_count / residual_count if residual_count else 0.0
    diagnostics = SolverDiagnostics(info == 0, 1, int(info != 0), relative_change)
    return correction, diagnostics, float(tail_fraction)


@dataclass
class FixedLagTwin:
    side: int
    steps: int
    fixed_lag: int
    huber: bool
    delta: float
    lambda_prior: float
    lambda_spatial: float
    max_irls: int
    tolerance: float
    initial_state: np.ndarray
    adaptive_huber: bool = False
    gated_correction: bool = False
    residual_correction: bool = False
    guarded_correction: bool = False
    predictive_residual: bool = False
    predictive_adaptive_delta: bool = False
    predictive_clean_delta: float = 4.0
    predictive_robust_delta: float = 2.9
    predictive_activation_gain: float = 1.0
    predictive_calibrated_gate: bool = False
    predictive_calibration_epochs: int = 48
    predictive_excess_gate_start: float = 0.01
    predictive_excess_gate_full: float = 0.04
    clean_delta: float = 6.0
    robust_delta: float = 1.345
    adaptive_tail_z: float = 2.5
    adaptive_clean_tail_fraction: float = 0.02
    adaptive_contaminated_tail_fraction: float = 0.15
    innovation_gate_start: float = 0.02
    innovation_gate_full: float = 0.12
    innovation_gate_tail_z: float = 3.5
    correction_delta: float = 3.0
    lambda_correction: float = 0.5
    correction_clip: float = 8.0
    correction_clean_gain: float = 0.25
    correction_attack_gain: float = 3.0
    correction_gate_start: float = 0.15
    correction_gate_full: float = 0.22
    correction_gate_ewma: float = 0.25
    transitions: tuple[sparse.csr_matrix, ...] | None = None
    spatial_laplacian: sparse.csr_matrix | None = None

    def __post_init__(self) -> None:
        cells = self.side * self.side
        if self.initial_state.shape != (cells,):
            raise ValueError(f"initial_state must have shape {(cells,)}")
        self.laplacian = (grid_laplacian(self.side) if self.spatial_laplacian is None
                          else sparse.csr_matrix(self.spatial_laplacian))
        if self.laplacian.shape != (cells, cells):
            raise ValueError("spatial laplacian does not match twin dimensions")
        if self.transitions is not None:
            if len(self.transitions) != self.steps or any(
                operator.shape != (cells, cells) or not np.all(np.isfinite(operator.data))
                for operator in self.transitions
            ):
                raise ValueError("transitions must cover the complete finite twin horizon")
            if self.residual_correction:
                raise ValueError("legacy score-correction mode does not support nonidentity dynamics")
        self.states = np.tile(self.initial_state, (self.steps, 1)).astype(float)
        self.clean_states = self.states.copy()
        self.robust_states = self.states.copy()
        self.predictive_states = self.states.copy()
        self.correction_states = np.zeros_like(self.states)
        self.release_baselines = np.tile(self.initial_state, (self.steps, 1)).astype(float)
        self.known: dict[int, list[Observation]] = {epoch: [] for epoch in range(self.steps)}
        self.robust_known: dict[int, list[Observation]] = {
            epoch: [] for epoch in range(self.steps)
        }
        self.raw_known: dict[int, list[Observation]] = {epoch: [] for epoch in range(self.steps)}
        self.retrospective_records: list[Observation] = []
        self.diagnostics: list[SolverDiagnostics] = []
        self.adaptive_history: list[AdaptiveHuberState] = []
        self.robust_gate_history: list[float] = []
        self._closed_standardized_residuals: list[float] = []
        self.innovation_contamination_history: list[float] = []
        self.correction_abs_mean_history: list[float] = []
        self.correction_abs_max_history: list[float] = []
        self.correction_tail_fraction_history: list[float] = []
        self.correction_activation_history: list[float] = []
        self.closed_tail_ewma_history: list[float] = []
        self.predictive_delta_history: list[float] = []
        self.predictive_tail_fraction_history: list[float] = []
        self.predictive_activation_by_epoch = np.zeros(self.steps, dtype=float)
        self._predictive_calibration_tail: list[float] = []
        self.predictive_tail_baseline_history: list[float] = []
        self._closed_tail_ewma = 0.0
        if self.residual_correction and not self.huber:
            raise ValueError("residual correction requires the robust layer")
        if self.guarded_correction and not self.residual_correction:
            raise ValueError("guarded correction requires residual correction")
        if self.predictive_residual and not self.huber:
            raise ValueError("predictive-residual assimilation requires the robust layer")
        if not 0 < self.predictive_robust_delta <= self.predictive_clean_delta:
            raise ValueError("invalid predictive adaptive-delta endpoints")
        if self.predictive_activation_gain <= 0:
            raise ValueError("predictive activation gain must be positive")
        if self.predictive_calibration_epochs < 1 or not (
            0
            <= self.predictive_excess_gate_start
            < self.predictive_excess_gate_full
            <= 1
        ):
            raise ValueError("invalid predictive calibrated-tail gate")
        if self.predictive_adaptive_delta and not (
            0 <= self.innovation_gate_start < self.innovation_gate_full <= 1
        ):
            raise ValueError("invalid predictive adaptive-delta gate")
        if not 0 < self.correction_gate_ewma <= 1:
            raise ValueError("correction_gate_ewma must lie in (0, 1]")
        if not 0 <= self.correction_gate_start < self.correction_gate_full <= 1:
            raise ValueError("invalid correction-gate thresholds")
        if not 0 <= self.correction_clean_gain <= self.correction_attack_gain:
            raise ValueError("invalid correction gains")

    def ingest_at(
        self,
        current_epoch: int,
        arrivals: Iterable[Observation],
        *,
        external_predictor: np.ndarray | None = None,
    ) -> None:
        if external_predictor is not None:
            expected_shape = (min(current_epoch, self.fixed_lag) + 1, self.side * self.side)
            if not self.predictive_residual or external_predictor.shape != expected_shape:
                raise ValueError("external predictor requires predictive mode and the active window")
            if not np.all(np.isfinite(external_predictor)) or np.any(external_predictor < 0):
                raise ValueError("external predictor must be finite and non-negative")
        self.release_baselines[current_epoch] = (
            self.initial_state
            if current_epoch == 0
            else self.states[current_epoch - 1].copy()
        )
        adaptive = adaptive_huber_state(
            self._closed_standardized_residuals,
            clean_delta=self.clean_delta,
            robust_delta=self.robust_delta,
            tail_z=self.adaptive_tail_z,
            clean_tail_fraction=self.adaptive_clean_tail_fraction,
            contaminated_tail_fraction=self.adaptive_contaminated_tail_fraction,
        )
        delta = adaptive.delta if self.huber and self.adaptive_huber else self.delta
        self.adaptive_history.append(adaptive)
        robust_gate = float(self.huber and not self.gated_correction)
        correction_activation = 0.0
        if self.guarded_correction:
            if adaptive.residual_count:
                self._closed_tail_ewma = (
                    self.correction_gate_ewma * adaptive.estimated_contamination
                    + (1.0 - self.correction_gate_ewma) * self._closed_tail_ewma
                )
            correction_activation = float(
                np.clip(
                    (self._closed_tail_ewma - self.correction_gate_start)
                    / (self.correction_gate_full - self.correction_gate_start),
                    0.0,
                    1.0,
                )
            )
            robust_gate = float(
                self.correction_clean_gain
                + correction_activation
                * (self.correction_attack_gain - self.correction_clean_gain)
            )
        self.closed_tail_ewma_history.append(self._closed_tail_ewma)
        self.correction_activation_history.append(correction_activation)
        incoming = list(arrivals)
        start = max(0, current_epoch - self.fixed_lag)
        transition_window = (None if self.transitions is None
                             else self.transitions[start : current_epoch + 1])
        boundary = self.initial_state if start == 0 else self.states[start - 1].copy()
        initial = self.states[start : current_epoch + 1].copy()
        predictor_trajectory: np.ndarray | None = None
        predictor_diagnostics: SolverDiagnostics | None = None
        robust_predictor_trajectory: np.ndarray | None = None
        robust_predictor_diagnostics: SolverDiagnostics | None = None
        robust_boundary: np.ndarray | None = None
        robust_initial: np.ndarray | None = None
        bounded_incoming = incoming
        trusted_incoming: list[Observation] = []
        if self.predictive_residual:
            if self.predictive_adaptive_delta:
                boundary = (
                    self.initial_state
                    if start == 0
                    else self.clean_states[start - 1].copy()
                )
                initial = self.clean_states[start : current_epoch + 1].copy()
                robust_boundary = (
                    self.initial_state
                    if start == 0
                    else self.robust_states[start - 1].copy()
                )
                robust_initial = self.robust_states[start : current_epoch + 1].copy()
            trusted_incoming = [
                item for item in incoming if item.source_class == "regulatory"
            ]
            bounded_incoming = [
                item for item in incoming if item.source_class != "regulatory"
            ]
            for item in trusted_incoming:
                if start <= item.epoch <= current_epoch:
                    self.raw_known[item.epoch].append(item)
                    self.known[item.epoch].append(item)
                    self.robust_known[item.epoch].append(item)
                elif item.epoch < start:
                    self.retrospective_records.append(item)
            if external_predictor is not None:
                # Both tracks share the externally supplied, pre-batch anchor. The
                # caller owns its causality/public-input contract; no citizen state
                # is fed back into it by this class.
                predictor_trajectory = external_predictor.copy()
                predictor_diagnostics = SolverDiagnostics(True, 0, 0, 0.0)
                if self.predictive_adaptive_delta:
                    robust_predictor_trajectory = external_predictor.copy()
                    robust_predictor_diagnostics = SolverDiagnostics(True, 0, 0, 0.0)
            else:
                predictor_trajectory, predictor_diagnostics = robust_trajectory_update(
                    boundary,
                    initial,
                    [self.known[epoch] for epoch in range(start, current_epoch + 1)],
                    self.laplacian,
                    transitions=transition_window,
                    huber=False,
                    delta=self.clean_delta,
                    lambda_temporal=self.lambda_prior,
                    lambda_spatial=self.lambda_spatial,
                    max_irls=max(2, self.max_irls),
                    tolerance=self.tolerance,
                )
                if self.predictive_adaptive_delta:
                    assert robust_boundary is not None and robust_initial is not None
                    robust_predictor_trajectory, robust_predictor_diagnostics = (
                        robust_trajectory_update(
                            robust_boundary,
                            robust_initial,
                            [self.robust_known[epoch] for epoch in range(start, current_epoch + 1)],
                            self.laplacian,
                            transitions=transition_window,
                            huber=False,
                            delta=self.clean_delta,
                            lambda_temporal=self.lambda_prior,
                            lambda_spatial=self.lambda_spatial,
                            max_irls=max(2, self.max_irls),
                            tolerance=self.tolerance,
                        )
                    )
            # A pre-batch internal state is NOT automatically a public baseline:
            # it may still depend on protected citizen history. Only an independently
            # public or privacy-accounted external predictor repairs that dependence.
            self.release_baselines[current_epoch] = predictor_trajectory[-1].copy()
        current_untrusted_raw = [
            item
            for item in bounded_incoming
            if item.epoch == current_epoch and item.source_class != "regulatory"
        ]
        innovation_fraction = innovation_contamination_fraction(
            current_untrusted_raw,
            predictor_trajectory[-1] if predictor_trajectory is not None else initial[-1],
            tail_z=self.innovation_gate_tail_z,
        )
        self.innovation_contamination_history.append(innovation_fraction)
        predictive_tail_fraction = standardized_innovation_tail_fraction(
            current_untrusted_raw,
            predictor_trajectory[-1] if predictor_trajectory is not None else initial[-1],
            tail_z=self.innovation_gate_tail_z,
        )
        self.predictive_tail_fraction_history.append(predictive_tail_fraction)
        if current_epoch < self.predictive_calibration_epochs:
            self._predictive_calibration_tail.append(predictive_tail_fraction)
        calibrated_baseline = float(
            np.median(self._predictive_calibration_tail)
        ) if self._predictive_calibration_tail else 0.0
        self.predictive_tail_baseline_history.append(calibrated_baseline)
        predictive_delta = self.correction_delta
        predictive_activation = 0.0
        if self.predictive_residual and self.predictive_adaptive_delta:
            if self.predictive_calibrated_gate:
                gate_value = predictive_tail_fraction - calibrated_baseline
                gate_start = self.predictive_excess_gate_start
                gate_full = self.predictive_excess_gate_full
            else:
                gate_value = predictive_tail_fraction
                gate_start = self.innovation_gate_start
                gate_full = self.innovation_gate_full
            predictive_activation = float(
                np.clip(
                    self.predictive_activation_gain
                    * (gate_value - gate_start)
                    / (gate_full - gate_start),
                    0.0,
                    1.0,
                )
            )
            predictive_delta = float(
                self.predictive_clean_delta
                - predictive_activation
                * (self.predictive_clean_delta - self.predictive_robust_delta)
            )
        self.predictive_activation_by_epoch[current_epoch] = predictive_activation
        self.predictive_delta_history.append(predictive_delta)
        for item in bounded_incoming:
            if item.epoch <= current_epoch and item.epoch >= max(0, current_epoch - self.fixed_lag):
                self.raw_known[item.epoch].append(item)
                if self.predictive_residual:
                    assert predictor_trajectory is not None
                    predicted = float(predictor_trajectory[item.epoch - start, item.cell])
                    sigma = max(float(item.sigma), 1e-6)
                    clean_radius = (
                        self.predictive_clean_delta
                        if self.predictive_adaptive_delta
                        else predictive_delta
                    ) * sigma
                    clean_residual = float(
                        np.clip(float(item.value) - predicted, -clean_radius, clean_radius)
                    )
                    bounded_value = predicted + clean_residual
                    self.known[item.epoch].append(replace(item, value=bounded_value))
                    if self.predictive_adaptive_delta:
                        assert robust_predictor_trajectory is not None
                        robust_predicted = float(
                            robust_predictor_trajectory[item.epoch - start, item.cell]
                        )
                        robust_radius = self.predictive_robust_delta * max(
                            float(item.sigma), 1e-6
                        )
                        robust_residual = float(
                            np.clip(
                                float(item.value) - robust_predicted,
                                -robust_radius,
                                robust_radius,
                            )
                        )
                        robust_value = robust_predicted + robust_residual
                        self.robust_known[item.epoch].append(
                            replace(item, value=robust_value)
                        )
                else:
                    self.known[item.epoch].append(item)
            elif item.epoch < max(0, current_epoch - self.fixed_lag):
                self.retrospective_records.append(item)
        observation_window = [self.known[epoch] for epoch in range(start, current_epoch + 1)]
        robust_observation_window = [
            self.robust_known[epoch] for epoch in range(start, current_epoch + 1)
        ]
        if self.huber and self.gated_correction:
            if not 0 <= self.innovation_gate_start < self.innovation_gate_full <= 1:
                raise ValueError("invalid innovation-gate thresholds")
            robust_gate = (
                (innovation_fraction - self.innovation_gate_start)
                / (self.innovation_gate_full - self.innovation_gate_start)
            )
        if not self.guarded_correction:
            robust_gate = float(np.clip(robust_gate, 0.0, 1.0))
        self.robust_gate_history.append(robust_gate)
        if self.huber and self.predictive_residual:
            assert predictor_trajectory is not None and predictor_diagnostics is not None
            clean_assimilated, assimilation_diagnostics = robust_trajectory_update(
                boundary,
                predictor_trajectory,
                observation_window,
                self.laplacian,
                transitions=transition_window,
                huber=False,
                delta=self.clean_delta,
                lambda_temporal=self.lambda_prior,
                lambda_spatial=self.lambda_spatial,
                max_irls=max(2, self.max_irls),
                tolerance=self.tolerance,
            )
            applied_correction = np.clip(
                clean_assimilated - predictor_trajectory,
                -self.correction_clip,
                self.correction_clip,
            )
            clean_trajectory = np.maximum(predictor_trajectory + applied_correction, 0.0)
            trajectory = clean_trajectory
            diagnostics = SolverDiagnostics(
                predictor_diagnostics.converged and assimilation_diagnostics.converged,
                max(
                    predictor_diagnostics.irls_iterations,
                    assimilation_diagnostics.irls_iterations,
                ),
                predictor_diagnostics.cg_failures + assimilation_diagnostics.cg_failures,
                max(
                    predictor_diagnostics.relative_change,
                    assimilation_diagnostics.relative_change,
                ),
            )
            self.predictive_states[start : current_epoch + 1] = predictor_trajectory
            if self.predictive_adaptive_delta:
                assert (
                    robust_boundary is not None
                    and robust_predictor_trajectory is not None
                    and robust_predictor_diagnostics is not None
                )
                robust_assimilated, robust_assimilation_diagnostics = (
                    robust_trajectory_update(
                        robust_boundary,
                        robust_predictor_trajectory,
                        robust_observation_window,
                        self.laplacian,
                        transitions=transition_window,
                        huber=False,
                        delta=self.clean_delta,
                        lambda_temporal=self.lambda_prior,
                        lambda_spatial=self.lambda_spatial,
                        max_irls=max(2, self.max_irls),
                        tolerance=self.tolerance,
                    )
                )
                robust_correction = np.clip(
                    robust_assimilated - robust_predictor_trajectory,
                    -self.correction_clip,
                    self.correction_clip,
                )
                robust_trajectory = np.maximum(
                    robust_predictor_trajectory + robust_correction, 0.0
                )
                self.clean_states[start : current_epoch + 1] = clean_trajectory
                self.robust_states[start : current_epoch + 1] = robust_trajectory
                weights = self.predictive_activation_by_epoch[
                    start : current_epoch + 1
                ].reshape(-1, 1)
                blended = (1.0 - weights) * clean_trajectory + weights * robust_trajectory
                applied_correction = np.clip(
                    blended - predictor_trajectory,
                    -self.correction_clip,
                    self.correction_clip,
                )
                trajectory = np.maximum(predictor_trajectory + applied_correction, 0.0)
                diagnostics = SolverDiagnostics(
                    diagnostics.converged
                    and robust_predictor_diagnostics.converged
                    and robust_assimilation_diagnostics.converged,
                    max(
                        diagnostics.irls_iterations,
                        robust_predictor_diagnostics.irls_iterations,
                        robust_assimilation_diagnostics.irls_iterations,
                    ),
                    diagnostics.cg_failures
                    + robust_predictor_diagnostics.cg_failures
                    + robust_assimilation_diagnostics.cg_failures,
                    max(
                        diagnostics.relative_change,
                        robust_predictor_diagnostics.relative_change,
                        robust_assimilation_diagnostics.relative_change,
                    ),
                )
            self.correction_states[start : current_epoch + 1] = applied_correction
            self.correction_abs_mean_history.append(float(np.mean(np.abs(applied_correction))))
            self.correction_abs_max_history.append(float(np.max(np.abs(applied_correction))))
            raw_innovations = [
                abs(
                    (float(item.value) - predictor_trajectory[item.epoch - start, item.cell])
                    / max(float(item.sigma), 1e-6)
                )
                for item in bounded_incoming
                if start <= item.epoch <= current_epoch
            ]
            self.correction_tail_fraction_history.append(
                float(np.mean(np.asarray(raw_innovations) > predictive_delta))
                if raw_innovations
                else 0.0
            )
        elif self.huber and self.residual_correction:
            clean_trajectory, clean_diagnostics = robust_trajectory_update(
                boundary,
                initial,
                observation_window,
                self.laplacian,
                transitions=transition_window,
                huber=False,
                delta=self.clean_delta,
                lambda_temporal=self.lambda_prior,
                lambda_spatial=self.lambda_spatial,
                max_irls=max(2, self.max_irls),
                tolerance=self.tolerance,
            )
            correction_delta = adaptive.delta if self.adaptive_huber else self.correction_delta
            correction, correction_diagnostics, tail_fraction = (
                robust_score_correction_trajectory(
                    clean_trajectory,
                    observation_window,
                    self.laplacian,
                    delta=correction_delta,
                    lambda_correction=self.lambda_correction,
                    lambda_temporal=self.lambda_prior,
                    lambda_spatial=self.lambda_spatial,
                    correction_clip=self.correction_clip,
                    tolerance=self.tolerance,
                )
            )
            applied_correction = np.clip(
                robust_gate * correction, -self.correction_clip, self.correction_clip
            )
            trajectory = np.maximum(clean_trajectory + applied_correction, 0.0)
            diagnostics = SolverDiagnostics(
                correction_diagnostics.converged and clean_diagnostics.converged,
                max(correction_diagnostics.irls_iterations, clean_diagnostics.irls_iterations),
                correction_diagnostics.cg_failures + clean_diagnostics.cg_failures,
                max(correction_diagnostics.relative_change, clean_diagnostics.relative_change),
            )
            self.predictive_states[start : current_epoch + 1] = clean_trajectory
            self.correction_states[start : current_epoch + 1] = applied_correction
            self.correction_abs_mean_history.append(float(np.mean(np.abs(applied_correction))))
            self.correction_abs_max_history.append(float(np.max(np.abs(applied_correction))))
            self.correction_tail_fraction_history.append(tail_fraction)
        else:
            trajectory, diagnostics = robust_trajectory_update(
                boundary,
                initial,
                observation_window,
                self.laplacian,
                transitions=transition_window,
                huber=self.huber,
                delta=delta,
                lambda_temporal=self.lambda_prior,
                lambda_spatial=self.lambda_spatial,
                max_irls=self.max_irls,
                tolerance=self.tolerance,
            )
            if self.huber and self.gated_correction:
                clean_trajectory, clean_diagnostics = robust_trajectory_update(
                    boundary,
                    initial,
                    observation_window,
                    self.laplacian,
                    transitions=transition_window,
                    huber=False,
                    delta=self.clean_delta,
                    lambda_temporal=self.lambda_prior,
                    lambda_spatial=self.lambda_spatial,
                    max_irls=max(2, self.max_irls),
                    tolerance=self.tolerance,
                )
                trajectory = clean_trajectory + robust_gate * (trajectory - clean_trajectory)
                diagnostics = SolverDiagnostics(
                    diagnostics.converged and clean_diagnostics.converged,
                    max(diagnostics.irls_iterations, clean_diagnostics.irls_iterations),
                    diagnostics.cg_failures + clean_diagnostics.cg_failures,
                    max(diagnostics.relative_change, clean_diagnostics.relative_change),
                )
            self.predictive_states[start : current_epoch + 1] = trajectory
            self.correction_states[start : current_epoch + 1] = 0.0
        self.states[start : current_epoch + 1] = trajectory
        self.diagnostics.append(diagnostics)
        # These residuals become eligible only for the next epoch's detector.  A
        # current record therefore cannot enlarge its own Huber transition.
        self._closed_standardized_residuals = [
            (float(item.value) - self.states[current_epoch, item.cell])
            / max(float(item.sigma), 1e-6)
            for item in current_untrusted_raw
        ]

    @property
    def failure_rate(self) -> float:
        if not self.diagnostics:
            return 0.0
        return float(np.mean([not item.converged for item in self.diagnostics]))

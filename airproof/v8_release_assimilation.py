"""Bounded post-processing of AirProof's fixed protected-release transcript.

The deployable API accepts public reference values, public uncertainty and the
already released numeric vector.  It has no contributor-count, raw-record,
noiseless-query or exact-state input.  A conjugate constant-bias state pools the
28 delayed releases; the published correction is uncertainty gated and clipped.
This is deterministic post-processing and therefore spends no privacy budget.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ReleaseAssimilationConfig:
    groups: int
    scheduled_acquisition_epochs: tuple[int, ...]
    deadline_epochs: int = 24
    epsilon_history: float = 8.0
    k_min: int = 20
    laplace_scale: float = 7.0
    prior_variance: float = 25.0
    public_uncertainty_gate: float = 4.0
    innovation_clip: float = 21.0
    correction_cap: float = 8.0

    def __post_init__(self) -> None:
        slots = tuple(int(value) for value in self.scheduled_acquisition_epochs)
        numeric = (
            self.epsilon_history,
            self.laplace_scale,
            self.prior_variance,
            self.public_uncertainty_gate,
            self.innovation_clip,
            self.correction_cap,
        )
        if (
            self.groups < 1
            or len(slots) != 28
            or len(set(slots)) != 28
            or slots != tuple(sorted(slots))
            or slots[0] < 0
            or self.deadline_epochs != 24
            or self.epsilon_history != 8.0
            or self.k_min != 20
            or not all(np.isfinite(value) and value > 0 for value in numeric)
        ):
            raise ValueError("registered epsilon-8/28-release/k20/deadline24 contract required")


class ProtectedReleaseAssimilator:
    """Causal robust normal-moment pool for a persistent public-reference bias."""

    def __init__(self, config: ReleaseAssimilationConfig):
        self.config = config
        self._scheduled = set(config.scheduled_acquisition_epochs)
        self._public: dict[int, np.ndarray] = {}
        self._seen_slots: set[tuple[int, int]] = set()
        self._precision = np.full(config.groups, 1.0 / config.prior_variance)
        self._weighted_sum = np.zeros(config.groups)
        self._posterior_mean = np.zeros(config.groups)
        self._last_epoch = -1
        self.assimilated = 0
        self.suppressed = 0
        self.gated_group_epochs = 0
        self.clipped_innovations = 0

    def update(
        self,
        epoch: int,
        public_reference: np.ndarray,
        public_uncertainty: np.ndarray,
        *,
        acquisition_epoch: int | None = None,
        protected_values: np.ndarray | None = None,
        released_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance one epoch and optionally assimilate one published vector.

        ``protected_values`` contains absolute released group estimates.  The
        innovation subtracts the stored public value at acquisition time.  A
        vector may enter only at acquisition+24 and only once per group.
        """
        cfg = self.config
        public = np.asarray(public_reference, dtype=float)
        uncertainty = np.asarray(public_uncertainty, dtype=float)
        if (
            epoch != self._last_epoch + 1
            or public.shape != (cfg.groups,)
            or uncertainty.shape != (cfg.groups,)
            or not np.isfinite(public).all()
            or not np.isfinite(uncertainty).all()
            or np.any(uncertainty < 0)
        ):
            raise ValueError("chronological finite public reference/uncertainty required")
        self._last_epoch = epoch
        self._public[epoch] = public.copy()

        supplied = (protected_values is not None, released_mask is not None)
        if any(supplied) or acquisition_epoch is not None:
            if not all(supplied) or acquisition_epoch is None:
                raise ValueError("publication requires acquisition, values and mask together")
            if (
                acquisition_epoch not in self._scheduled
                or epoch - acquisition_epoch != cfg.deadline_epochs
                or acquisition_epoch not in self._public
            ):
                raise ValueError("protected vector violates fixed schedule/publication clock")
            values = np.asarray(protected_values, dtype=float)
            mask = np.asarray(released_mask)
            if values.shape != (cfg.groups,) or mask.shape != values.shape or mask.dtype != bool:
                raise ValueError("matched protected vector and boolean mask required")
            if not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():
                raise ValueError("suppression must be represented only as missing values")
            observation_variance = 2.0 * cfg.laplace_scale**2
            acquisition_public = self._public[acquisition_epoch]
            for group in range(cfg.groups):
                slot = (acquisition_epoch, group)
                if slot in self._seen_slots:
                    raise ValueError("a protected query slot can influence the state only once")
                self._seen_slots.add(slot)
                if not mask[group]:
                    self.suppressed += 1
                    continue
                innovation = values[group] - acquisition_public[group]
                centered = innovation - self._posterior_mean[group]
                bounded = self._posterior_mean[group] + np.clip(
                    centered, -cfg.innovation_clip, cfg.innovation_clip
                )
                self.clipped_innovations += int(bounded != innovation)
                self._precision[group] += 1.0 / observation_variance
                self._weighted_sum[group] += bounded / observation_variance
                self._posterior_mean[group] = (
                    self._weighted_sum[group] / self._precision[group]
                )
                self.assimilated += 1

        gate = uncertainty >= cfg.public_uncertainty_gate
        self.gated_group_epochs += int(gate.sum())
        correction = np.where(
            gate,
            np.clip(self._posterior_mean, -cfg.correction_cap, cfg.correction_cap),
            0.0,
        )
        return np.maximum(public + correction, 0.0)

    def diagnostics(self) -> dict:
        cfg = self.config
        return {
            "state": "groupwise persistent public-reference bias",
            "assimilated_releases": self.assimilated,
            "suppressed_releases": self.suppressed,
            "unique_slots_seen": len(self._seen_slots),
            "gated_group_epochs": self.gated_group_epochs,
            "clipped_innovations": self.clipped_innovations,
            "posterior_mean": self._posterior_mean.tolist(),
            "posterior_variance": (1.0 / self._precision).tolist(),
            "privacy_spend_added_by_postprocessing": 0.0,
            "exact_private_counts_used": False,
            "raw_records_used": False,
            "noiseless_query_used": False,
            "release_count_contract": len(cfg.scheduled_acquisition_epochs),
            "deadline_epochs": cfg.deadline_epochs,
            "epsilon_history": cfg.epsilon_history,
            "k_min": cfg.k_min,
            "laplace_variance_added_once": 2.0 * cfg.laplace_scale**2,
            "correction_cap": cfg.correction_cap,
            "public_uncertainty_gate": cfg.public_uncertainty_gate,
        }


def run_protected_assimilation(
    public_reference: np.ndarray,
    public_uncertainty: np.ndarray,
    protected_values: np.ndarray,
    released_mask: np.ndarray,
    config: ReleaseAssimilationConfig,
) -> tuple[np.ndarray, dict]:
    """Run the release-only consumer over one fixed public transcript."""
    public = np.asarray(public_reference, dtype=float)
    uncertainty = np.asarray(public_uncertainty, dtype=float)
    values = np.asarray(protected_values, dtype=float)
    mask = np.asarray(released_mask)
    if (
        public.ndim != 2
        or public.shape[1] != config.groups
        or uncertainty.shape != public.shape
        or values.shape != public.shape
        or mask.shape != public.shape
        or mask.dtype != bool
        or not np.isfinite(public).all()
        or not np.isfinite(uncertainty).all()
        or not np.isfinite(values[mask]).all()
        or not np.isnan(values[~mask]).all()
    ):
        raise ValueError("finite matched public and sparse protected arrays required")
    scheduled = np.zeros(len(public), dtype=bool)
    scheduled[list(config.scheduled_acquisition_epochs)] = True
    if np.any(mask[~scheduled]):
        raise ValueError("protected values outside the fixed schedule")
    assimilator = ProtectedReleaseAssimilator(config)
    predictions = []
    for epoch in range(len(public)):
        acquisition = epoch - config.deadline_epochs
        published = acquisition in assimilator._scheduled
        predictions.append(
            assimilator.update(
                epoch,
                public[epoch],
                uncertainty[epoch],
                acquisition_epoch=acquisition if published else None,
                protected_values=values[acquisition] if published else None,
                released_mask=mask[acquisition] if published else None,
            )
        )
    return np.asarray(predictions), assimilator.diagnostics()


def persistent_reference_stress(
    public_reference: np.ndarray,
    nominal_uncertainty: np.ndarray,
    group_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a declared downstream public-reference calibration stress.

    The protected absolute release transcript is unchanged.  The public service
    exposes only the biased reference and its inflated uncertainty; the consumer
    never receives the latent bias vector.
    """
    public = np.asarray(public_reference, dtype=float)
    uncertainty = np.asarray(nominal_uncertainty, dtype=float)
    bias = np.asarray(group_bias, dtype=float)
    if (
        public.ndim != 2
        or uncertainty.shape != public.shape
        or bias.shape != (public.shape[1],)
        or not np.isfinite(public).all()
        or not np.isfinite(uncertainty).all()
        or not np.isfinite(bias).all()
        or np.any(uncertainty < 0)
    ):
        raise ValueError("matched finite public stress inputs required")
    stressed = public + bias
    inflated = np.sqrt(uncertainty**2 + bias[None, :] ** 2)
    return stressed, inflated


def stress_public_reference_observations(reference_observations, group_bias: np.ndarray):
    """Apply a fixed calibration shift to public regulatory observations.

    This transform is performed before public-model fitting and protected-query
    construction.  It reads only public regulatory records, never citizen data.
    """
    bias = np.asarray(group_bias, dtype=float)
    if bias.ndim != 1 or not np.isfinite(bias).all():
        raise ValueError("finite one-dimensional public group bias required")
    stressed = []
    for observation in reference_observations:
        group = int(observation.group)
        if observation.source_class != "regulatory" or not 0 <= group < len(bias):
            raise ValueError("valid public regulatory observation required")
        stressed.append(replace(observation, value=float(observation.value + bias[group])))
    return tuple(stressed)


def published_release_carry(
    public_reference: np.ndarray,
    protected_values: np.ndarray,
    released_mask: np.ndarray,
    config: ReleaseAssimilationConfig,
) -> np.ndarray:
    """Causal direct-release control on the same full prediction support."""
    public = np.asarray(public_reference, dtype=float)
    values = np.asarray(protected_values, dtype=float)
    mask = np.asarray(released_mask)
    if (
        public.ndim != 2
        or public.shape[1] != config.groups
        or values.shape != public.shape
        or mask.shape != public.shape
        or mask.dtype != bool
        or not np.isfinite(public).all()
        or not np.isfinite(values[mask]).all()
        or not np.isnan(values[~mask]).all()
    ):
        raise ValueError("matched public and sparse protected arrays required")
    scheduled = np.zeros(len(public), dtype=bool)
    scheduled[list(config.scheduled_acquisition_epochs)] = True
    if np.any(mask[~scheduled]):
        raise ValueError("protected values outside fixed schedule")
    last = np.zeros(config.groups)
    seen = np.zeros(config.groups, dtype=bool)
    prediction = np.empty_like(public)
    for epoch in range(len(public)):
        acquisition = epoch - config.deadline_epochs
        if acquisition in set(config.scheduled_acquisition_epochs):
            take = mask[acquisition]
            last[take] = values[acquisition, take]
            seen[take] = True
        prediction[epoch] = np.maximum(np.where(seen, last, public[epoch]), 0.0)
    return prediction

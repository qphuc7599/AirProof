"""Release-only delayed linear estimator with a correlated latent bias state.

The state is (truth minus public reference, noiseless query minus truth).
AR(1) moments are fitted on separate public/synthetic calibration histories.
This is a linear moment estimator under Laplace noise, not an exact Gaussian
posterior. All online inputs belong to the declared protected release view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ReleaseCalibration:
    mean: np.ndarray
    transition: np.ndarray
    process_covariance: np.ndarray
    initial_covariance: np.ndarray
    groups: int
    source: str = "public-synthetic-calibration-only"
    covariance_shrinkage: float = .1


@dataclass(frozen=True)
class PublicReleaseObservation:
    release_id: str
    acquisition_epoch: int
    publication_epoch: int
    group: int
    value: float | None
    laplace_scale: float


def _covariance(values: np.ndarray, shrinkage: float) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    covariance = (1-shrinkage)*covariance + shrinkage*np.diag(np.diag(covariance))
    ridge = max(float(np.trace(covariance)/len(covariance))*1e-8, 1e-8)
    return (covariance+covariance.T)/2 + ridge*np.eye(len(covariance))


def fit_release_calibration(public_truth: np.ndarray, public_baselines: np.ndarray,
                            noiseless_queries: np.ndarray, *, covariance_shrinkage: float = .1,
                            source: str = "public-synthetic-calibration-only") -> ReleaseCalibration:
    """Inputs [world,time,group] or [time,group]; never join world boundaries.

    Noiseless query must include actual clipping, stabilization and sampling.
    It excludes only the independently generated DP noise, added analytically.
    """
    arrays = [np.asarray(value, dtype=float) for value in (public_truth, public_baselines, noiseless_queries)]
    if arrays[0].ndim == 2:
        arrays = [value[None] for value in arrays]
    truth, public, query = arrays
    if truth.ndim != 3 or truth.shape != public.shape or truth.shape != query.shape:
        raise ValueError("matched world-time-group calibration arrays required")
    if truth.shape[1] < 4 or truth.shape[2] < 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("finite calibration with at least four epochs required")
    if covariance_shrinkage != .1 or source != "public-synthetic-calibration-only":
        raise ValueError("registered public/synthetic calibration and shrinkage .1 required")
    latent = np.concatenate((truth-public, query-truth), axis=-1)
    mean = latent.mean(axis=(0, 1))
    centered = latent-mean
    before, after = centered[:, :-1].reshape(-1, len(mean)), centered[:, 1:].reshape(-1, len(mean))
    ar = np.clip(np.sum(before*after, axis=0)/np.maximum(np.sum(before**2, axis=0), 1e-12), -.98, .98)
    transition = np.diag(ar)
    innovations = after-before@transition.T
    return ReleaseCalibration(mean, transition, _covariance(innovations, covariance_shrinkage),
                              _covariance(centered.reshape(-1, len(mean)), covariance_shrinkage),
                              truth.shape[2], source, covariance_shrinkage)


class ReleaseOnlyTwin:
    def __init__(self, calibration: ReleaseCalibration, *, lag: int = 48):
        if not isinstance(lag, (int, np.integer)) or lag < 24:
            raise ValueError("release lag must retain the declared 24-hour collection delay")
        self.calibration = calibration
        self.groups = calibration.groups
        self.dimension = 2*self.groups
        if calibration.mean.shape != (self.dimension,) or not np.isfinite(calibration.mean).all():
            raise ValueError("invalid calibration dimension")
        if calibration.transition.shape != (self.dimension, self.dimension) or not np.isfinite(calibration.transition).all():
            raise ValueError("invalid calibration transition")
        for matrix in (calibration.process_covariance, calibration.initial_covariance):
            if matrix.shape != (self.dimension, self.dimension) or not np.isfinite(matrix).all():
                raise ValueError("invalid calibration covariance")
            np.linalg.cholesky(matrix)
        self.lag = lag
        self.epochs: list[int] = []
        self.mean = np.empty(0)
        self.covariance = np.empty((0, 0))
        self.public: dict[int, np.ndarray] = {}
        self.live: dict[int, np.ndarray] = {}
        self.reconstructed: dict[int, np.ndarray] = {}
        self.seen: dict[str, PublicReleaseObservation] = {}
        self.assimilated = self.suppressed = self.retrospective = self.duplicates = 0

    def update(self, epoch: int, public_baseline: np.ndarray,
               releases: Iterable[PublicReleaseObservation] = ()) -> np.ndarray:
        baseline = np.asarray(public_baseline, float)
        if baseline.shape != (self.groups,) or not np.isfinite(baseline).all():
            raise ValueError("finite public group baseline required")
        if epoch != (max(self.public)+1 if self.public else 0):
            raise ValueError("update every epoch once in chronological order")
        incoming = list(releases)
        batch_seen = {}
        for release in incoming:
            if not release.release_id or not 0 <= release.group < self.groups:
                raise ValueError("invalid release identifier/group")
            if release.publication_epoch != epoch or not 0 <= release.acquisition_epoch <= epoch:
                raise ValueError("release clock mismatch")
            if not np.isfinite(release.laplace_scale) or release.laplace_scale < 0:
                raise ValueError("finite nonnegative mechanism scale required")
            if release.value is not None and not np.isfinite(release.value):
                raise ValueError("finite released value or explicit suppression required")
            if release.release_id in self.seen and self.seen[release.release_id] != release:
                raise ValueError("conflicting reused release identifier")
            if release.release_id in batch_seen and batch_seen[release.release_id] != release:
                raise ValueError("conflicting release identifier within batch")
            batch_seen[release.release_id] = release
        model = self.calibration
        if self.epochs:
            if len(self.epochs) >= self.lag+1:
                self.epochs.pop(0)
                self.mean = self.mean[self.dimension:]
                self.covariance = self.covariance[self.dimension:, self.dimension:]
            newest = model.mean+model.transition@(self.mean[-self.dimension:]-model.mean)
            cross = self.covariance[:, -self.dimension:]@model.transition.T
            variance = model.transition@self.covariance[-self.dimension:, -self.dimension:]@model.transition.T+model.process_covariance
            self.mean = np.concatenate((self.mean, newest))
            self.covariance = np.block([[self.covariance, cross], [cross.T, variance]])
        else:
            self.mean = model.mean.copy()
            self.covariance = model.initial_covariance.copy()
        self.epochs.append(epoch)
        self.public[epoch] = baseline.copy()
        for release in incoming:
            if release.release_id in self.seen:
                self.duplicates += 1
                continue
            self.seen[release.release_id] = release
            if release.value is None:
                self.suppressed += 1
                continue
            if release.acquisition_epoch not in self.epochs:
                self.retrospective += 1
                continue
            offset = self.epochs.index(release.acquisition_epoch)*self.dimension
            truth_index, bias_index = offset+release.group, offset+self.groups+release.group
            # H observes residual truth + latent clipping/stabilization bias.
            covariance_h = self.covariance[:, truth_index]+self.covariance[:, bias_index]
            innovation_variance = covariance_h[truth_index]+covariance_h[bias_index]+2*release.laplace_scale**2
            innovation = release.value-self.public[release.acquisition_epoch][release.group]-self.mean[truth_index]-self.mean[bias_index]
            self.mean += covariance_h*(innovation/max(innovation_variance, 1e-12))
            self.covariance -= np.outer(covariance_h, covariance_h)/max(innovation_variance, 1e-12)
            self.covariance = (self.covariance+self.covariance.T)/2
            self.assimilated += 1
        for local, acquisition in enumerate(self.epochs):
            self.reconstructed[acquisition] = np.maximum(self.public[acquisition]+self.mean[local*self.dimension:local*self.dimension+self.groups], 0.)
        result = self.reconstructed[epoch].copy()
        self.live[epoch] = result.copy()
        return result

    def diagnostics(self) -> dict:
        return {"lag_epochs": self.lag, "assimilated_releases": self.assimilated,
                "suppressed_releases": self.suppressed, "retrospective_releases": self.retrospective,
                "duplicate_releases": self.duplicates, "covariance_shrinkage": .1,
                "calibration_source": self.calibration.source,
                "state": "joint truth residual and clipped/stabilized query bias AR(1)",
                "dp_variance_added_once": True, "privacy_spend_added_by_postprocessing": 0.,
                "uncertainty_scope": "linear moment covariance, not exact Laplace posterior",
                "exact_private_counts_used": False, "raw_twin_state_used": False}

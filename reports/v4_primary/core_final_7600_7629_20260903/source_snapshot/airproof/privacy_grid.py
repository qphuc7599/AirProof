"""Sufficient-statistic replay of the production fixed-schedule DP mechanism.

The raw twin and fair allocator are NOT rerun for each privacy setting: the
release channel is independent of both by contract. Production equivalence is
tested against release_epoch, including its RNG order and optional count gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from .continual_release import FixedReleasePlan, independent_contributions
from .rng import stream_seed


@dataclass(frozen=True)
class ReleaseStatistics:
    public_baselines: np.ndarray
    counts: np.ndarray
    raw_sums: np.ndarray
    residual_sums: np.ndarray
    raw_clip: float
    residual_clip: float
    contributor_epochs: int


def prepare_statistics(records, *, public_baselines: np.ndarray, groups: int,
                       raw_clip: float = 50., residual_clip: float = 10.,
                       relay: bool = True, deadline_steps: int = 24) -> ReleaseStatistics:
    baseline = np.asarray(public_baselines, float)
    if baseline.ndim != 2 or baseline.shape[1] != groups or not np.isfinite(baseline).all():
        raise ValueError("finite time-by-group public baseline required")
    if min(raw_clip, residual_clip) <= 0:
        raise ValueError("positive contribution clips required")
    steps = len(baseline)
    contributions = independent_contributions(records, steps=steps, groups=groups,
                                              relay=relay, deadline_steps=deadline_steps)
    counts = np.zeros((steps, groups), dtype=int)
    raw_sums, residual_sums = np.zeros_like(baseline), np.zeros_like(baseline)
    for epoch, users in contributions.items():
        group = np.array([items[0].group for items in users.values()], dtype=int)
        # Aggregate each user's own readings before clipping, as in production.
        values = np.array([np.mean([item.value for item in items]) for items in users.values()])
        counts[epoch] = np.bincount(group, minlength=groups)
        raw_sums[epoch] = np.bincount(group, weights=np.clip(values, -raw_clip, raw_clip), minlength=groups)
        residual_sums[epoch] = np.bincount(group,
            weights=np.clip(values - baseline[epoch, group], -residual_clip, residual_clip), minlength=groups)
    return ReleaseStatistics(baseline.copy(), counts, raw_sums, residual_sums,
                              raw_clip, residual_clip, int(counts.sum()))


def replay_release(statistics: ReleaseStatistics, plan: FixedReleasePlan, *, residual: bool,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if statistics.public_baselines.shape != (plan.steps, plan.groups):
        raise ValueError("release statistics/plan mismatch")
    clip = statistics.residual_clip if residual else statistics.raw_clip
    sums = statistics.residual_sums if residual else statistics.raw_sums
    query = sums / np.maximum(statistics.counts, plan.k_min)
    if residual:
        query = query + statistics.public_baselines
    values = np.full_like(query, np.nan)
    emitted = np.zeros(query.shape, dtype=bool)
    sensitivity = 4 * clip / plan.k_min
    for epoch in plan.scheduled_epochs:
        noisy = query[epoch] + rng.laplace(0., sensitivity / plan.epsilon_mean, plan.groups)
        if plan.private_eligibility:
            noisy_counts = statistics.counts[epoch] + rng.laplace(0., 2 / plan.epsilon_count, plan.groups)
            threshold = plan.k_min + (2 / plan.epsilon_count) * math.log(.5 / plan.false_release_probability)
            allowed = noisy_counts >= threshold
        else:
            allowed = np.ones(plan.groups, dtype=bool)
        emitted[epoch] = allowed
        values[epoch, allowed] = noisy[allowed]
    return values, emitted


def plan_for_setting(setting: dict, *, steps: int, groups: int, false_release_probability: float) -> FixedReleasePlan:
    count = setting["scheduled_epochs"]
    if not 1 <= count <= steps:
        raise ValueError("scheduled count outside horizon")
    total_per_epoch = setting["epsilon_user_max"] / count
    fraction = setting["count_budget_fraction"] if setting["private_eligibility"] else 0.
    if not 0 <= fraction < 1:
        raise ValueError("invalid count-budget fraction")
    plan = FixedReleasePlan(steps, groups, setting["k_min"], total_per_epoch * (1 - fraction),
        total_per_epoch * fraction, setting["epsilon_user_max"], setting["private_eligibility"],
        false_release_probability)
    if len(plan.scheduled_epochs) != count:
        raise ValueError("floating-point schedule differs from registered release count")
    return plan


def grid_settings(protocol: dict) -> list[dict]:
    return [{"epsilon_user_max": cap, "scheduled_epochs": epochs, "k_min": k,
             "private_eligibility": private, "count_budget_fraction": protocol["count_budget_fraction"],
             "release_mode": mode}
            for cap, epochs, k, private, mode in itertools.product(protocol["epsilon_user_caps"],
                protocol["scheduled_epochs"], protocol["k_min"], protocol["private_eligibility"],
                ("raw", "residual"))]


def evaluate_setting(statistics: ReleaseStatistics, truth: np.ndarray, setting: dict, *, seed: int,
                     burn_in: int, deadline_steps: int = 24, false_release_probability: float = .01) -> dict:
    steps, groups = statistics.counts.shape
    if truth.shape != (steps, groups) or not np.isfinite(truth).all() or not 0 <= burn_in < steps:
        raise ValueError("invalid group truth/scoring domain")
    plan = plan_for_setting(setting, steps=steps, groups=groups, false_release_probability=false_release_probability)
    residual = setting["release_mode"] == "residual"
    values, emitted = replay_release(statistics, plan, residual=residual,
                                     rng=np.random.default_rng(stream_seed(seed, "privacy")))
    scored = emitted.copy()
    scored[:burn_in] = False
    errors = values[scored] - truth[scored]
    scheduled = np.array(plan.scheduled_epochs, dtype=int)
    opportunities = int((scheduled >= burn_in).sum()) * groups
    # A fixed collection deadline closes each acquisition-epoch query; no use of
    # its future arrivals is described as a release at the acquisition time.
    ages = []
    latest = np.full(groups, -1, dtype=int)
    for acquisition_epoch in range(steps):
        latest = np.where(emitted[acquisition_epoch], acquisition_epoch, latest)
        if acquisition_epoch >= burn_in:
            ages.extend((acquisition_epoch + deadline_steps - latest).tolist())
    clip = statistics.residual_clip if residual else statistics.raw_clip
    return {**setting, "epsilon_mean": plan.epsilon_mean, "epsilon_count": plan.epsilon_count,
        "composed_epsilon": plan.composed_epsilon,
        "privacy_budget_violation_count": int(plan.composed_epsilon > plan.epsilon_user_max + 1e-10),
        "release_count": int(scored.sum()), "scheduled_group_opportunities": opportunities,
        "conditional_release_fraction": float(scored.sum() / opportunities) if opportunities else None,
        "all_epoch_release_fraction": float(scored.sum() / ((steps - burn_in) * groups)),
        "release_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "release_mae": float(np.mean(np.abs(errors))) if len(errors) else None,
        "reference_group_rmse": float(np.sqrt(np.mean((statistics.public_baselines[burn_in:] - truth[burn_in:]) ** 2))),
        "mean_age_of_last_release_at_query_closure": float(np.mean(ages)),
        "mean_acquisition_cadence_hours": float(steps / len(scheduled)),
        "public_collection_deadline_hours": deadline_steps,
        "vector_sensitivity": 4 * clip / plan.k_min,
        "privacy_query_admission": "independent-reserved-release-channel",
        "public_baseline_citizen_independent": True,
        "counts_released": False,
        "release_clock": "acquisition_epoch_plus_fixed_collection_deadline",
        "scope": "one-counterfactual-fixed-schedule-release-transcript-not-joint-publication-of-the-grid"}

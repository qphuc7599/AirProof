"""Reviewer-requested matched controls for the fixed-schedule private release.

The historical raw arm clipped a nonnegative pollutant to a symmetric interval
and therefore used a deliberately conservative replacement bound.  This module
adds the matched [0, upper] control.  Moving one user between private groups can
change at most two coordinates by ``upper / k_min`` each, so the vector L1
replacement sensitivity is ``2 * upper / k_min``.
"""
from __future__ import annotations

import numpy as np

from .continual_release import FixedReleasePlan
from .rng import stream_seed
from .v5_experiment import release_evidence, reserve_contributions


def nonnegative_stabilized_query(
    contributions,
    *,
    groups: int,
    upper: float,
    k_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the group vector with per-user aggregation and [0, upper] clipping."""
    if groups < 1 or upper <= 0 or k_min < 1:
        raise ValueError("positive groups, upper bound and stabilizer required")
    values = [[] for _ in range(groups)]
    for user, items in contributions.items():
        if not items or any(
            item.user_id != user or item.group != items[0].group for item in items
        ):
            raise ValueError("expected one canonical group per user")
        group = int(items[0].group)
        if not 0 <= group < groups or not all(np.isfinite(item.value) for item in items):
            raise ValueError("invalid local aggregate")
        values[group].append(float(np.clip(np.mean([item.value for item in items]), 0, upper)))
    counts = np.asarray([len(group_values) for group_values in values], dtype=int)
    query = np.asarray(
        [sum(group_values) / max(len(group_values), k_min) for group_values in values],
        dtype=float,
    )
    return query, counts


def reviewer_release_evidence(world, public, release_arrivals, cfg):
    """Add a matched nonnegative raw arm without changing the frozen v5 mechanism."""
    summary, arrays = release_evidence(world, public, release_arrivals, cfg)
    steps, _ = world.truth.shape
    groups = int(cfg["world"]["groups"])
    burn = int(cfg["world"]["burn_in_steps"])
    privacy = cfg["privacy"]
    upper = float(privacy.get("nonnegative_raw_upper", privacy["clip"]))
    if float(privacy.get("nonnegative_raw_lower", 0.0)) != 0.0:
        raise ValueError("reviewer control is registered on [0, upper]")
    plan = FixedReleasePlan(
        steps,
        groups,
        int(privacy["k_min"]),
        float(privacy["epsilon_mean"]),
        float(privacy.get("epsilon_count", 0.0)),
        float(privacy["epsilon_user_max"]),
        bool(privacy.get("private_eligibility", False)),
        0.01,
    )
    if plan.private_eligibility:
        raise ValueError("matched reviewer control requires fixed public emission support")
    deadline = int(privacy.get("release_deadline_steps", 24))
    contributions = reserve_contributions(
        world.observations,
        release_arrivals,
        steps=steps,
        deadline=deadline,
    )
    truth = arrays["truth"]
    baseline = arrays["baseline"]
    query = np.full((steps, groups), np.nan)
    values = np.full_like(query, np.nan)
    mask = np.zeros_like(query, dtype=bool)
    sensitivity = 2.0 * upper / plan.k_min
    scale = sensitivity / plan.epsilon_mean
    # Common random numbers make the mechanism comparison lower variance while
    # preserving the correct marginal Laplace law for every arm.
    rng = np.random.default_rng(stream_seed(world.seed, "privacy"))
    for epoch in plan.scheduled_epochs:
        query[epoch], _ = nonnegative_stabilized_query(
            contributions.get(epoch, {}),
            groups=groups,
            upper=upper,
            k_min=plan.k_min,
        )
        values[epoch] = query[epoch] + rng.laplace(0.0, scale, groups)
        mask[epoch] = True
    scored = mask.copy()
    scored[:burn] = False
    error = values[scored] - truth[scored]
    deterministic = query[scored] - truth[scored]
    public_error = baseline[scored] - truth[scored]
    summary["raw_nonnegative"] = {
        "domain": [0.0, upper],
        "replacement_vector_l1_sensitivity": sensitivity,
        "laplace_scale": scale,
        "laplace_variance": 2.0 * scale**2,
        "rmse": float(np.sqrt(np.mean(error**2))) if error.size else None,
        "same_support_public_rmse": (
            float(np.sqrt(np.mean(public_error**2))) if error.size else None
        ),
        "deterministic_query_mse": (
            float(np.mean(deterministic**2)) if error.size else None
        ),
        "released_group_queries": int(scored.sum()),
        "scheduled_group_queries": int(scored.sum()),
        "adjacency": "replace one complete user history; private group may change",
        "common_random_numbers_with_other_release_arms": True,
    }
    # Keep the old arm explicit for auditability rather than silently relabeling it.
    summary["raw_symmetric_historical_control"] = summary.pop("raw")
    arrays["raw_nonnegative"] = values
    arrays["raw_nonnegative_query"] = query
    arrays["raw_nonnegative_mask"] = mask
    return summary, arrays

"""User-history DP release with independent admission and a public fixed schedule.

This path must not consume the output of a competitive raw-evidence scheduler.
It assumes authenticated per-credential ingress reservations: removing/replacing
one user's complete history cannot displace another user's release contribution.
Operational metadata and raw twin/audit outputs remain outside the DP transcript.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np

from .privacy import PrivateRelease, stabilized_bounded_mean
from .records import Observation


@dataclass(frozen=True)
class FixedReleasePlan:
    steps: int
    groups: int
    k_min: int
    epsilon_mean: float
    epsilon_count: float
    epsilon_user_max: float
    private_eligibility: bool = False
    false_release_probability: float = .01

    def __post_init__(self):
        if min(self.steps, self.groups, self.k_min) < 1:
            raise ValueError("release plan dimensions must be positive")
        if not all(math.isfinite(v) for v in (
            self.epsilon_mean, self.epsilon_count, self.epsilon_user_max
        )) or self.epsilon_mean <= 0 or self.epsilon_count < 0 or self.epsilon_user_max < 0:
            raise ValueError("invalid release plan privacy parameters")
        if self.private_eligibility and self.epsilon_count <= 0:
            raise ValueError("private eligibility requires count epsilon")
        if not 0 < self.false_release_probability < .5:
            raise ValueError("invalid false release probability")

    @property
    def epsilon_per_epoch(self) -> float:
        return self.epsilon_mean + (self.epsilon_count if self.private_eligibility else 0)

    @property
    def scheduled_epochs(self) -> tuple[int, ...]:
        # Public, outcome-independent spacing; budget exhaustion cannot depend on
        # another user's presence or on any noisy suppression decision.
        count = min(self.steps, int(math.floor(
            (self.epsilon_user_max + 1e-12) / self.epsilon_per_epoch
        )))
        return tuple(int(index * self.steps // count) for index in range(count))

    @property
    def composed_epsilon(self) -> float:
        return len(self.scheduled_epochs) * self.epsilon_per_epoch


def independent_contributions(records: list[Observation], *, steps: int, groups: int,
                              relay: bool, deadline_steps: int) -> dict[int, dict[int, list[Observation]]]:
    """Select at most one group per user/epoch using only that user's records.

    An authenticated sender could submit readings from several regions in an
    epoch. The lowest valid group id is the documented canonical release domain;
    all of that user's readings in that domain are locally aggregated later.
    There is no top-k competition, private global denominator in admission, or
    budget-dependent displacement. Delivery timestamps are inputs to the reserved
    release channel, not outcomes of a shared finite-queue raw relay simulator.
    """
    if steps < 1 or groups < 1 or deadline_steps < 0:
        raise ValueError("invalid independent contribution domain")
    local = defaultdict(list)
    seen = set()
    for item in records:
        if item.source_class != "citizen":
            raise ValueError("release contributions must be authenticated citizen records")
        if (not 0 <= item.epoch < steps or not 0 <= item.group < groups
                or not np.isfinite(item.value)):
            raise ValueError("invalid citizen release contribution")
        arrival = item.relay_arrival if relay else item.direct_arrival
        if arrival is None or not item.epoch <= arrival <= item.epoch + deadline_steps:
            continue
        # Include user and domain: the credential authority, not a caller-chosen
        # colliding string, owns nullifier binding. Duplicate replays have no weight.
        key = (item.user_id, item.epoch, item.group, item.nullifier)
        if key not in seen:
            seen.add(key)
            local[(item.epoch, item.user_id)].append(item)
    result = defaultdict(dict)
    for (epoch, user), items in sorted(local.items()):
        group = min(item.group for item in items)
        result[epoch][user] = [item for item in items if item.group == group]
    return dict(result)


def bounded_epoch_query(contributions: dict[int, list[Observation]], *, groups: int,
                        baselines: np.ndarray, clip: float, k_min: int,
                        residual: bool) -> tuple[np.ndarray, np.ndarray]:
    """Vector query; arbitrary replacement can change at most two groups.

    Add/remove user L1 sensitivity <=2C/k; replacement <=4C/k including a
    private change of region. Count-vector replacement sensitivity <=2.
    """
    centers = np.asarray(baselines, dtype=float)
    if centers.shape != (groups,) or not np.all(np.isfinite(centers)) or clip <= 0:
        raise ValueError("invalid public baseline or clipping radius")
    values = [[] for _ in range(groups)]
    for user, items in contributions.items():
        if not items or any(item.user_id != user or item.group != items[0].group for item in items):
            raise ValueError("expected one canonical group per user")
        group = items[0].group
        if not 0 <= group < groups or not all(np.isfinite(item.value) for item in items):
            raise ValueError("invalid local aggregate")
        center = centers[group] if residual else 0.0
        # Arbitrary local history -> one clipped scalar. Clip after aggregation.
        value = float(np.clip(np.mean([item.value for item in items]) - center, -clip, clip))
        values[group].append(value)
    query = np.array([stabilized_bounded_mean(v, clip=clip, k_min=k_min) for v in values])
    if residual:
        query += centers
    return query, np.array([len(v) for v in values], dtype=int)


def release_epoch(contributions: dict[int, list[Observation]], *, epoch: int,
                  plan: FixedReleasePlan, public_baselines: np.ndarray, clip: float,
                  residual: bool, rng: np.random.Generator) -> list[PrivateRelease]:
    if not 0 <= epoch < plan.steps:
        raise ValueError("epoch outside release plan")
    scope = "user-history-replacement-fixed-schedule"
    if residual:
        scope = "public-residual-" + scope
    sensitivity = 4 * clip / plan.k_min
    if epoch not in plan.scheduled_epochs:
        return [PrivateRelease(g, epoch, False, None, 0, sensitivity,
                               plan.epsilon_mean, plan.epsilon_count, scope, 0)
                for g in range(plan.groups)]
    query, counts = bounded_epoch_query(contributions, groups=plan.groups,
                                       baselines=public_baselines, clip=clip,
                                       k_min=plan.k_min, residual=residual)
    # One vector-Laplace mechanism, not unaccounted per-group composition.
    noisy_values = query + rng.laplace(0, sensitivity / plan.epsilon_mean, plan.groups)
    if plan.private_eligibility:
        noisy_counts = counts + rng.laplace(0, 2 / plan.epsilon_count, plan.groups)
        threshold = plan.k_min + (2 / plan.epsilon_count) * math.log(
            .5 / plan.false_release_probability
        )
        eligible = noisy_counts >= threshold
    else:
        # The default emits every scheduled query, including empty cohorts.
        # k_min stabilizes the denominator; it is NOT an exact-count release gate.
        noisy_counts, threshold = None, None
        eligible = np.ones(plan.groups, dtype=bool)
    return [PrivateRelease(
        g, epoch, bool(eligible[g]), float(noisy_values[g]) if eligible[g] else None,
        int(counts[g]), sensitivity, plan.epsilon_mean,
        plan.epsilon_count if plan.private_eligibility else 0.0, scope, 0,
        None if noisy_counts is None else float(noisy_counts[g]), threshold,
    ) for g in range(plan.groups)]

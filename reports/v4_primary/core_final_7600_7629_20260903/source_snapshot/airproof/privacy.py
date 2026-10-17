from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import numpy as np

from .records import Observation


@dataclass
class PrivacyAccountant:
    epsilon_max: float
    spent: dict[int, float] = field(default_factory=dict)
    events: list[PrivacySpendEvent] = field(default_factory=list)

    def can_spend(self, user_id: int, epsilon: float) -> bool:
        return self.spent.get(user_id, 0.0) + epsilon <= self.epsilon_max + 1e-12

    def spend(
        self,
        user_id: int,
        epsilon: float,
        *,
        group: int | None = None,
        epoch: int | None = None,
        mechanism: str = "unspecified",
    ) -> bool:
        if not self.can_spend(user_id, epsilon):
            return False
        self.spent[user_id] = self.spent.get(user_id, 0.0) + epsilon
        self.events.append(PrivacySpendEvent(user_id, group, epoch, mechanism, epsilon))
        return True

    def remaining(self, user_id: int) -> float:
        return max(0.0, self.epsilon_max - self.spent.get(user_id, 0.0))

    @property
    def max_spent(self) -> float:
        return max(self.spent.values(), default=0.0)


@dataclass(frozen=True)
class PrivacySpendEvent:
    user_id: int
    group: int | None
    epoch: int | None
    mechanism: str
    epsilon: float


@dataclass(frozen=True)
class PrivateRelease:
    group: int
    epoch: int
    released: bool
    value: float | None
    distinct_users: int
    sensitivity: float
    epsilon_mean: float
    epsilon_count: float
    transcript_scope: str
    declined_users: int
    noisy_count: float | None = None
    eligibility_threshold: float | None = None

    def protected_payload(self) -> dict[str, object]:
        """Return only fields covered by the scoped release mechanism.

        The exact contributor count and accountant diagnostics remain internal.
        """
        return {
            "group": self.group,
            "epoch": self.epoch,
            "released": self.released,
            "value": self.value,
            "transcript_scope": self.transcript_scope,
        }


def _one_value_per_user(records: Iterable[Observation], clip: float) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for record in records:
        # Physical validity clipping is per reading; contribution clipping is repeated
        # after aggregation so one user's complete epoch history has sensitivity 2C.
        values.setdefault(record.user_id, []).append(float(np.clip(record.value, -clip, clip)))
    return {user: float(np.clip(np.mean(items), -clip, clip)) for user, items in values.items()}


def laplace_threshold_margin(epsilon_count: float, false_release_probability: float) -> float:
    """Safety margin whose one-sided Laplace tail is ``false_release_probability``."""
    if epsilon_count <= 0:
        raise ValueError("epsilon_count must be positive")
    if not 0 < false_release_probability < 0.5:
        raise ValueError("false_release_probability must lie in (0, 0.5)")
    return math.log(0.5 / false_release_probability) / epsilon_count


def stabilized_bounded_mean(values: Iterable[float], *, clip: float, k_min: int) -> float:
    """Deterministic query to which calibrated Laplace noise is added."""
    if clip <= 0 or k_min <= 0:
        raise ValueError("clip and k_min must be positive")
    bounded = [float(np.clip(value, -clip, clip)) for value in values]
    return float(sum(bounded) / max(len(bounded), k_min))


def release_group_mean(
    records: Iterable[Observation],
    *,
    group: int,
    epoch: int,
    clip: float,
    k_min: int,
    epsilon_mean: float,
    epsilon_count: float,
    accountant: PrivacyAccountant,
    rng: np.random.Generator,
    private_eligibility: bool = False,
    eligibility_safety_margin: float | None = None,
    false_release_probability: float = 1e-6,
) -> PrivateRelease:
    if clip <= 0 or k_min <= 0 or epsilon_mean <= 0:
        raise ValueError("clip, k_min and epsilon_mean must be positive")
    selected = [record for record in records if record.group == group and record.epoch == epoch]
    per_user = _one_value_per_user(selected, clip)
    epoch_epsilon = epsilon_mean + (epsilon_count if private_eligibility else 0.0)
    eligible_users: dict[int, float] = {}
    declined = 0
    for user_id, value in sorted(per_user.items()):
        if accountant.can_spend(user_id, epoch_epsilon):
            eligible_users[user_id] = value
        else:
            declined += 1
    count = len(eligible_users)
    if private_eligibility:
        if epsilon_count <= 0:
            raise ValueError("private eligibility requires epsilon_count > 0")
        noisy_count = float(count + rng.laplace(0.0, 1.0 / epsilon_count))
        safety_margin = (
            laplace_threshold_margin(epsilon_count, false_release_probability)
            if eligibility_safety_margin is None
            else float(eligibility_safety_margin)
        )
        if safety_margin < 0:
            raise ValueError("eligibility_safety_margin must be non-negative")
        threshold = float(k_min + safety_margin)
        release = noisy_count >= threshold
        transcript_scope = "mean+private-eligibility"
    else:
        noisy_count = None
        threshold = None
        release = count >= k_min
        transcript_scope = "mean-only-public-eligibility"
    sensitivity = 2.0 * clip / k_min
    if not release:
        if private_eligibility:
            for user_id in eligible_users:
                if not accountant.spend(
                    user_id,
                    epsilon_count,
                    group=group,
                    epoch=epoch,
                    mechanism="eligibility",
                ):
                    raise RuntimeError("Privacy accountant changed during private eligibility")
        return PrivateRelease(
            group, epoch, False, None, count, sensitivity, epsilon_mean,
            epsilon_count if private_eligibility else 0.0,
            transcript_scope,
            declined,
            noisy_count,
            threshold,
        )
    fixed_mean = stabilized_bounded_mean(eligible_users.values(), clip=clip, k_min=k_min)
    noisy_mean = fixed_mean + rng.laplace(0.0, sensitivity / epsilon_mean)
    charge = epoch_epsilon if private_eligibility else epsilon_mean
    for user_id in eligible_users:
        if not accountant.spend(
            user_id,
            charge,
            group=group,
            epoch=epoch,
            mechanism="eligibility+mean" if private_eligibility else "mean",
        ):
            raise RuntimeError("Privacy accountant changed between eligibility and release")
    return PrivateRelease(
        group, epoch, True, float(noisy_mean), count, sensitivity, epsilon_mean,
        epsilon_count if private_eligibility else 0.0,
        transcript_scope,
        declined,
        noisy_count,
        threshold,
    )


def release_group_residual_mean(
    records: Iterable[Observation],
    *,
    group: int,
    epoch: int,
    public_baseline: float,
    residual_clip: float,
    k_min: int,
    epsilon_mean: float,
    epsilon_count: float,
    accountant: PrivacyAccountant,
    rng: np.random.Generator,
    private_eligibility: bool = False,
    eligibility_safety_margin: float | None = None,
    false_release_probability: float = 1e-6,
) -> PrivateRelease:
    """Release a DP residual correction around a contributor-independent baseline.

    Privacy follows from the same bounded-mean mechanism with clipping radius
    ``residual_clip``. The caller is responsible for ensuring that ``public_baseline``
    does not depend on current protected contributions.
    """
    if not math.isfinite(public_baseline):
        raise ValueError("public_baseline must be finite")
    transformed = [
        replace(record, value=float(record.value) - float(public_baseline))
        for record in records
    ]
    residual_release = release_group_mean(
        transformed,
        group=group,
        epoch=epoch,
        clip=residual_clip,
        k_min=k_min,
        epsilon_mean=epsilon_mean,
        epsilon_count=epsilon_count,
        accountant=accountant,
        rng=rng,
        private_eligibility=private_eligibility,
        eligibility_safety_margin=eligibility_safety_margin,
        false_release_probability=false_release_probability,
    )
    value = (
        None
        if residual_release.value is None
        else float(public_baseline + residual_release.value)
    )
    return replace(
        residual_release,
        value=value,
        transcript_scope=f"residual-{residual_release.transcript_scope}",
    )

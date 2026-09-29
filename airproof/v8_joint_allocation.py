"""Causal lifetime pacing with outcome- and service-aware allocation.

This module is additive to the frozen v6/v7 implementations.  It allocates the
same per-user lifetime exposure budget across a sequence of public decision
epochs.  A decision may use only records that have already arrived and remain
inside the fixed-lag estimator window.  Ranking uses the contemporaneously
available public reference, the submitted value, declared uncertainty, and
contributor service history through the preceding decision.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np

from .records import Observation


def _validate_records(records: tuple[Observation, ...]) -> None:
    for record in records:
        if not isinstance(record.user_id, (int, np.integer)):
            raise TypeError("integer user partition required")
        if not isinstance(record.group, (int, np.integer)) or record.group < 0:
            raise ValueError("nonnegative integer groups required")
        if not np.isfinite(record.value) or not np.isfinite(record.sigma):
            raise ValueError("finite record value and sigma required")
        if record.sigma <= 0:
            raise ValueError("positive sigma required")
        if not np.isfinite(record.quality) or not 0 < record.quality <= 1:
            raise ValueError("quality must be finite and in (0,1]")


def causal_joint_lifetime_exposure_weights(
    observations: Iterable[Observation],
    arrival_map: Mapping[str, int | None],
    public_reference: np.ndarray,
    *,
    lag: int,
    per_user_exposure_budget: float,
    horizon_start: int,
    horizon_end: int,
    decision_interval: int,
    innovation_scale_floor: float,
    innovation_clip: float,
    fairness_strength: float,
    outcome_strength: float = 1.0,
    processing_end: int | None = None,
    group_count: int | None = None,
) -> tuple[tuple[Observation, ...], dict[str, object]]:
    """Allocate fixed lifetime exposure at causal public decision epochs.

    Exposure accrues linearly with public time, exactly as in the v7 paced
    allocator.  Spending occurs every ``decision_interval`` epochs.  At a
    decision, each contributor can spend only accrued exposure and can choose
    only among their already-arrived records whose lag window is still active.
    The outcome score is a clipped standardized discrepancy from the public
    reference.  The fairness score is the normalized deficit in distinct
    contributors served by the record's group, evaluated before the decision.

    ``horizon_end`` is the acquisition/exposure-accrual boundary. Optional
    ``processing_end`` permits arrivals through the fixed-lag drain without
    slowing the public exposure-release curve. The per-user mass bound is an
    accounting invariant. Because the fairness score uses cross-user service
    state, that invariant alone is not a global replacement-sensitivity proof;
    the conditional whole-sequence theorem must not be extended to this
    allocator without an additional stability argument.
    """
    if not isinstance(lag, int) or lag < 0:
        raise ValueError("nonnegative integer lag required")
    if not isinstance(horizon_start, int) or not isinstance(horizon_end, int):
        raise TypeError("integer horizon endpoints required")
    if horizon_end <= horizon_start:
        raise ValueError("horizon_end must exceed horizon_start")
    if not isinstance(decision_interval, int) or decision_interval < 1:
        raise ValueError("positive integer decision interval required")
    finite_positive = (per_user_exposure_budget, innovation_scale_floor, innovation_clip)
    if not all(np.isfinite(value) and value > 0 for value in finite_positive):
        raise ValueError("positive finite budget, scale floor and innovation clip required")
    if not np.isfinite(fairness_strength) or fairness_strength < 0:
        raise ValueError("nonnegative finite fairness strength required")
    if not np.isfinite(outcome_strength) or outcome_strength < 0:
        raise ValueError("nonnegative finite outcome strength required")
    if processing_end is None:
        processing_end = horizon_end
    if not isinstance(processing_end, int) or processing_end < horizon_end:
        raise ValueError("processing_end must be an integer at least horizon_end")

    public = np.asarray(public_reference, dtype=float)
    if public.ndim != 2 or not np.isfinite(public).all():
        raise ValueError("finite two-dimensional public reference required")
    if public.shape[0] < processing_end:
        raise ValueError("public reference does not cover the allocation horizon")

    items = tuple(observations)
    _validate_records(items)
    inferred_groups = max((int(record.group) for record in items), default=-1) + 1
    if group_count is None:
        group_count = inferred_groups
    if (isinstance(group_count, bool) or not isinstance(group_count, (int, np.integer))
            or group_count < inferred_groups or group_count < 1):
        raise ValueError("group_count must cover every nonnegative record group")
    events: defaultdict[int, list[Observation]] = defaultdict(list)
    dropped_unavailable = 0
    for record in items:
        arrival = arrival_map.get(record.nullifier)
        if arrival is None:
            dropped_unavailable += 1
            continue
        arrival = int(arrival)
        if (
            arrival < horizon_start
            or arrival >= processing_end
            or not horizon_start <= int(record.epoch) < horizon_end
            or arrival > int(record.epoch) + lag
        ):
            dropped_unavailable += 1
            continue
        if not 0 <= record.cell < public.shape[1]:
            raise ValueError("record cell outside public reference")
        events[arrival].append(record)

    span = horizon_end - horizon_start
    backlog: defaultdict[int, list[Observation]] = defaultdict(list)
    allocated: defaultdict[int, float] = defaultdict(float)
    served_by_group: defaultdict[int, set[int]] = defaultdict(set)
    selected_by_group: defaultdict[int, int] = defaultdict(int)
    selected: list[Observation] = []
    expired_records = skipped_decisions = 0
    maximum_release_violation = 0.0

    for epoch in range(horizon_start, processing_end):
        for record in sorted(
            events.get(epoch, ()),
            key=lambda item: (int(item.user_id), int(item.epoch), item.nullifier),
        ):
            backlog[int(record.user_id)].append(record)
        for user in tuple(backlog):
            valid = [record for record in backlog[user] if epoch <= record.epoch + lag]
            expired_records += len(backlog[user]) - len(valid)
            if valid:
                backlog[user] = valid
            else:
                del backlog[user]

        decision = ((epoch - horizon_start + 1) % decision_interval == 0)
        decision = decision or epoch == processing_end - 1
        if not decision:
            continue

        released_fraction = min(1.0, (epoch - horizon_start + 1) / span)
        released = float(per_user_exposure_budget) * released_fraction
        groups = list(range(int(group_count)))
        counts_before = {group: len(served_by_group[group]) for group in groups}
        maximum_count = max(counts_before.values(), default=0)
        denomin = max(1, maximum_count)

        for user in sorted(backlog):
            available = max(0.0, released - allocated[user])
            if available <= 1e-15:
                skipped_decisions += 1
                continue

            def rank(record: Observation) -> tuple[float, float, int, int, str]:
                scale = max(float(record.sigma), float(innovation_scale_floor))
                standardized = min(
                    float(innovation_clip),
                    abs(float(record.value) - float(public[record.epoch, record.cell])) / scale,
                ) / float(innovation_clip)
                deficit = (maximum_count - counts_before[int(record.group)]) / denomin
                score = outcome_strength * standardized + fairness_strength * deficit
                return (
                    score,
                    standardized,
                    int(record.epoch),
                    -int(record.cell),
                    record.nullifier,
                )

            eligible = list(backlog[user])
            if not eligible:
                skipped_decisions += 1
                continue
            choice = max(eligible, key=rank)
            multiplicity = int(choice.epoch) + lag - epoch + 1
            requested = float(choice.quality) * multiplicity
            granted = min(requested, available)
            if granted <= 1e-15:
                skipped_decisions += 1
                continue
            selected.append(replace(choice, quality=granted / multiplicity))
            allocated[user] += granted
            served_by_group[int(choice.group)].add(user)
            selected_by_group[int(choice.group)] += 1
            backlog[user].remove(choice)
            if not backlog[user]:
                del backlog[user]
            maximum_release_violation = max(
                maximum_release_violation, allocated[user] - released
            )

    maximum = max(allocated.values(), default=0.0)
    if maximum > per_user_exposure_budget + 1e-10:
        raise AssertionError("lifetime exposure budget exceeded")
    if maximum_release_violation > 1e-10:
        raise AssertionError("paced cumulative exposure exceeded public release curve")
    selected.sort(
        key=lambda item: (int(item.epoch), int(item.cell), int(item.user_id), item.nullifier)
    )
    contributor_counts = {
        str(group): len(served_by_group[group]) for group in range(int(group_count))
    }
    all_counts = list(contributor_counts.values())
    maximum_contributors = max(all_counts, default=0)
    minimum_contributors = min(all_counts, default=0)
    contributor_gap = (
        0.0
        if maximum_contributors == 0
        else 1.0 - minimum_contributors / maximum_contributors
    )
    return tuple(selected), {
        "allocation_policy": "causal_joint_outcome_service_pacing_v1",
        "input_records": len(items),
        "effective_records": len(selected),
        "dropped_unavailable_records": dropped_unavailable,
        "expired_unselected_records": expired_records,
        "skipped_user_decisions_without_released_exposure": skipped_decisions,
        "users_with_exposure": len(allocated),
        "maximum_user_lifetime_exposure": float(maximum),
        "total_lifetime_exposure": float(sum(allocated.values())),
        "per_user_exposure_budget": float(per_user_exposure_budget),
        "maximum_release_curve_violation": float(maximum_release_violation),
        "decision_interval": decision_interval,
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "processing_end": processing_end,
        "lag": lag,
        "innovation_scale_floor": float(innovation_scale_floor),
        "innovation_clip": float(innovation_clip),
        "outcome_strength": float(outcome_strength),
        "fairness_strength": float(fairness_strength),
        "group_count": int(group_count),
        "effective_contributors_by_group": contributor_counts,
        "effective_records_by_group": {
            str(group): count for group, count in sorted(selected_by_group.items())
        },
        "effective_contributor_gap": float(contributor_gap),
    }

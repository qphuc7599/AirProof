from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .records import Observation


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[Observation, ...]
    spent_bytes: int
    counts: dict[int, int]
    deficits: dict[int, float]
    infeasible_groups: tuple[int, ...]
    opportunity_shortfall_groups: tuple[int, ...]
    budget_infeasible_groups: tuple[int, ...]
    max_deficit: float
    effective_targets: dict[int, int]
    globally_feasible: bool
    constraint_satisfied: bool
    constraint_mode: str


def service_deficit(count: int, target: int) -> float:
    if target <= 0:
        raise ValueError("Service targets must be positive")
    return max(0.0, 1.0 - count / target)


def population_targets(populations: dict[int, int], service_per_person: float) -> dict[int, int]:
    """Convert physical population counts into contributor service targets."""
    if service_per_person < 0:
        raise ValueError("service_per_person must be non-negative")
    if any(population < 0 for population in populations.values()):
        raise ValueError("populations must be non-negative")
    return {
        group: max(1, math.ceil(population * service_per_person))
        for group, population in populations.items()
    }


def policy_weight_targets(weights: dict[int, float], total_slots: int) -> dict[int, int]:
    """Allocate a separate policy-weight target model without calling weights population."""
    if total_slots < len(weights) or not weights or any(weight < 0 for weight in weights.values()):
        raise ValueError("total_slots must cover all groups and weights must be non-negative")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("at least one policy weight must be positive")
    raw = {group: total_slots * weight / weight_sum for group, weight in weights.items()}
    targets = {group: int(value) for group, value in raw.items()}
    remainder = total_slots - sum(targets.values())
    order = sorted(weights, key=lambda group: (raw[group] - targets[group], -group), reverse=True)
    for group in order[:remainder]:
        targets[group] += 1
    return {group: max(1, target) for group, target in targets.items()}


def _utility(record: Observation, epoch: int, *, causal_arrival: bool = False) -> float:
    if causal_arrival:
        freshness = 1.0 / (1.0 + max(0, epoch - record.epoch))
    else:
        # Legacy acquisition-time diagnostic only. Final online allocation passes
        # the actual admission clock and never ranks by a future realized arrival.
        arrival = record.relay_arrival if record.relay_arrival is not None else record.direct_arrival
        freshness = 0.0 if arrival is None else 1.0 / (1.0 + max(0, arrival - epoch))
    uncertainty_gain = 1.0 / max(record.sigma * record.sigma, 1e-9)
    return 0.45 * uncertainty_gain + 0.35 * freshness + 0.20 * record.quality


def select_evidence(
    candidates: Iterable[Observation],
    *,
    budget_bytes: int,
    reserve_fraction: float,
    targets: dict[int, int],
    fairness: bool = True,
    fairness_strength: float = 1.0,
    constraint_mode: str = "reserved",
    allocation_epoch: int | None = None,
) -> SelectionResult:
    if budget_bytes < 0 or not 0 <= reserve_fraction <= 1 or fairness_strength < 0:
        raise ValueError("Invalid budget or reserve fraction")
    if constraint_mode not in {"reserved", "hard_if_feasible"}:
        raise ValueError("constraint_mode must be reserved or hard_if_feasible")

    def utility(item):
        return _utility(item, item.epoch if allocation_epoch is None else allocation_epoch,
                        causal_arrival=allocation_epoch is not None)

    deduplicated: dict[tuple[int, int, int], Observation] = {}
    for item in candidates:
        if allocation_epoch is not None and item.epoch > allocation_epoch:
            raise ValueError("future-acquired evidence cannot enter the current allocation")
        key = (item.user_id, item.group, item.epoch if allocation_epoch is None else allocation_epoch)
        incumbent = deduplicated.get(key)
        if incumbent is None or utility(item) > utility(incumbent):
            deduplicated[key] = item
    remaining = list(deduplicated.values())
    selected: list[Observation] = []
    counts = {group: 0 for group in targets}
    spent = 0
    strength = float(fairness_strength) if fairness else 0.0
    effective_targets = {
        group: (max(1, math.ceil(target * min(strength, 1.0))) if strength > 0 else 0)
        for group, target in targets.items()
    }
    globally_feasible = False

    if fairness and strength > 0 and constraint_mode == "hard_if_feasible":
        required: list[Observation] = []
        enough_candidates = True
        for group, target in effective_targets.items():
            group_items = sorted(
                (item for item in remaining if item.group == group),
                key=lambda item: (item.size_bytes, -utility(item), item.user_id),
            )
            if len(group_items) < target:
                enough_candidates = False
                break
            required.extend(group_items[:target])
        globally_feasible = enough_candidates and sum(item.size_bytes for item in required) <= budget_bytes
        if globally_feasible:
            for item in required:
                selected.append(item)
                remaining.remove(item)
                counts[item.group] += 1
                spent += item.size_bytes

    if constraint_mode == "hard_if_feasible" and fairness and strength > 0:
        reserve = budget_bytes
    else:
        reserve = int(budget_bytes * reserve_fraction * min(strength, 1.0))

    # Recompute the currently maximum deficit after every distinct contributor.
    while remaining and spent < reserve and not globally_feasible:
        feasible_groups = {
            group
            for group, target in effective_targets.items()
            if target > 0
            if any(item.group == group and spent + item.size_bytes <= reserve for item in remaining)
        }
        deficient = [
            group
            for group in feasible_groups
            if service_deficit(counts[group], effective_targets[group]) > 0
        ]
        if not deficient:
            break
        max_deficit = max(
            service_deficit(counts[group], effective_targets[group]) for group in deficient
        )
        group = min(
            group
            for group in deficient
            if service_deficit(counts[group], effective_targets[group]) == max_deficit
        )
        group_items = [
            item for item in remaining if item.group == group and spent + item.size_bytes <= reserve
        ]
        if not group_items:
            break
        choice = max(group_items, key=lambda item: (utility(item) / item.size_bytes, -item.user_id))
        selected.append(choice)
        remaining.remove(choice)
        counts[group] += 1
        spent += choice.size_bytes

    ranked = sorted(
        remaining,
        key=lambda item: (
            utility(item)
            + (
                strength * service_deficit(counts[item.group], effective_targets[item.group])
                if effective_targets.get(item.group, 0) > 0
                else 0.0
            ),
            -item.size_bytes,
            -item.user_id,
        ),
        reverse=True,
    )
    for item in ranked:
        if spent + item.size_bytes <= budget_bytes:
            selected.append(item)
            counts[item.group] = counts.get(item.group, 0) + 1
            spent += item.size_bytes
    deficits = {group: service_deficit(counts[group], target) for group, target in targets.items()}
    constraint_satisfied = bool(
        fairness
        and strength > 0
        and all(counts[group] >= target for group, target in effective_targets.items())
    )
    candidates_by_group = defaultdict(int)
    for item in deduplicated.values():
        candidates_by_group[item.group] += 1
    opportunity_shortfall = tuple(
        sorted(group for group, target in targets.items() if candidates_by_group[group] < target)
    )
    selected_ids = {id(item) for item in selected}
    budget_infeasible = tuple(
        sorted(
            group
            for group in targets
            if deficits[group] > 0
            and candidates_by_group[group] > 0
            and not any(
                item.group == group
                and id(item) not in selected_ids
                and spent + item.size_bytes <= budget_bytes
                for item in deduplicated.values()
            )
        )
    )
    infeasible = tuple(sorted(set(opportunity_shortfall) | set(budget_infeasible)))
    return SelectionResult(
        tuple(selected),
        spent,
        counts,
        deficits,
        infeasible,
        opportunity_shortfall,
        budget_infeasible,
        max(deficits.values(), default=0.0),
        effective_targets,
        globally_feasible,
        constraint_satisfied,
        constraint_mode,
    )

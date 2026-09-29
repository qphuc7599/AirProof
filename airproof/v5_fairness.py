"""Exact minimax avoidable service deficit on causal representative partitions."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from .records import Observation
from .scheduler import SelectionResult, _utility, service_deficit


@dataclass(frozen=True)
class MinimaxSelectionResult(SelectionResult):
    available_targets: dict[int, int]
    unavoidable_deficits: dict[int, float]
    avoidable_deficits: dict[int, float]
    optimum_numerator: int
    optimum_denominator: int
    certificate_counts: dict[int, int]
    certificate_bytes: int
    violations: tuple[str, ...]


def _identity(item: Observation) -> tuple:
    # Tie-break only: no future realized arrival enters ranking.
    return (item.user_id, item.group, item.epoch, item.nullifier, item.cell,
            item.size_bytes, item.value, item.sigma, item.quality, item.source_class)


def causal_representatives(candidates: Iterable[Observation], allocation_epoch: int) -> tuple[Observation, ...]:
    """Same causal user/group partition as scheduler, with deterministic utility ties."""
    representatives: dict[tuple[int, int], Observation] = {}
    for item in candidates:
        if item.epoch > allocation_epoch:
            raise ValueError("future-acquired evidence cannot enter allocation")
        if not isinstance(item.size_bytes, int) or item.size_bytes < 0:
            raise ValueError("costs must be nonnegative integer bytes")
        if not all(math.isfinite(x) for x in (item.sigma, item.quality, item.value)):
            raise ValueError("record numerical fields must be finite")
        key = (item.user_id, item.group)
        previous = representatives.get(key)
        rank = (-_utility(item, allocation_epoch, causal_arrival=True), _identity(item))
        if previous is None or rank < (-_utility(previous, allocation_epoch, causal_arrival=True), _identity(previous)):
            representatives[key] = item
    return tuple(sorted(representatives.values(), key=_identity))


def select_minimax_evidence(
    candidates: Iterable[Observation], *, budget_bytes: int,
    targets: dict[int, int], allocation_epoch: int, fairness: bool = True,
) -> MinimaxSelectionResult:
    """Select a least-cost minimax certificate, then fill by causal utility/byte.

    Availability means an authenticated candidate has already arrived; the caller
    owns authentication and arrival filtering. Untargeted groups may use fill
    capacity but do not enter the fairness objective. Disabling fairness skips
    the certificate; the reported optimum still describes the feasible pool.
    """
    if not isinstance(budget_bytes, int) or budget_bytes < 0:
        raise ValueError("budget must be nonnegative integer bytes")
    if any(not isinstance(q, int) or q <= 0 for q in targets.values()):
        raise ValueError("targets must be positive integers")
    pool = causal_representatives(candidates, allocation_epoch)
    utility = lambda item: _utility(item, allocation_epoch, causal_arrival=True)
    groups: dict[int, list[Observation]] = defaultdict(list)
    for item in pool:
        groups[item.group].append(item)
    prefixes = {}
    available = {}
    for group, target in sorted(targets.items()):
        groups[group].sort(key=lambda item: (item.size_bytes, -utility(item), _identity(item)))
        available[group] = min(target, len(groups[group]))
        costs = [0]
        for item in groups[group][:available[group]]:
            costs.append(costs[-1] + item.size_bytes)
        prefixes[group] = costs
    breaks = sorted({Fraction(0)} | {
        Fraction(available[g] - k, q)
        for g, q in targets.items() for k in range(available[g] + 1)
    })

    def certificate(d: Fraction) -> tuple[dict[int, int], int]:
        counts = {}
        for g, q in targets.items():
            # ceil(a - d*q), exactly; avoids threshold floating-point errors.
            numerator = available[g] * d.denominator - d.numerator * q
            counts[g] = max(0, -(-numerator // d.denominator))
        return counts, sum(prefixes[g][k] for g, k in counts.items())

    low, high = 0, len(breaks) - 1
    while low < high:
        middle = (low + high) // 2
        if certificate(breaks[middle])[1] <= budget_bytes:
            high = middle
        else:
            low = middle + 1
    optimum = breaks[low]
    cert_counts, cert_cost = certificate(optimum)
    selected = [item for g in sorted(targets) for item in groups[g][:cert_counts[g]]] if fairness else []
    selected_ids = {id(item) for item in selected}
    spent = sum(item.size_bytes for item in selected)
    # Free evidence has infinite utility/byte; stable identity resolves ties.
    ranked = sorted((item for item in pool if id(item) not in selected_ids),
                    key=lambda item: (-(utility(item) / item.size_bytes if item.size_bytes else math.inf), _identity(item)))
    for item in ranked:
        if spent + item.size_bytes <= budget_bytes:
            selected.append(item)
            spent += item.size_bytes
    counts = dict.fromkeys(targets, 0)
    for item in selected:
        counts[item.group] = counts.get(item.group, 0) + 1
    deficits = {g: service_deficit(counts[g], q) for g, q in targets.items()}
    unavoidable = {g: (q - available[g]) / q for g, q in targets.items()}
    avoidable_exact = {g: Fraction(max(0, available[g] - counts[g]), q) for g, q in targets.items()}
    opportunity = tuple(sorted(g for g in targets if available[g] < targets[g]))
    budget_shortfall = tuple(sorted(g for g in targets if counts[g] < available[g]))
    globally_feasible = not opportunity and sum(prefixes[g][available[g]] for g in targets) <= budget_bytes
    violations = []
    if spent > budget_bytes:
        violations.append("budget")
    if fairness and max(avoidable_exact.values(), default=Fraction(0)) != optimum:
        violations.append("minimax_certificate")
    if fairness and globally_feasible and any(counts[g] < q for g, q in targets.items()):
        violations.append("feasible_floor")
    return MinimaxSelectionResult(
        selected=tuple(selected), spent_bytes=spent, counts=counts, deficits=deficits,
        infeasible_groups=tuple(sorted(set(opportunity) | set(budget_shortfall))),
        opportunity_shortfall_groups=opportunity, budget_infeasible_groups=budget_shortfall,
        max_deficit=max(deficits.values(), default=0.), effective_targets=dict(targets),
        globally_feasible=globally_feasible,
        constraint_satisfied=fairness and all(counts[g] >= q for g, q in targets.items()),
        constraint_mode="minimax_avoidable", available_targets=available,
        unavoidable_deficits=unavoidable, avoidable_deficits={g: float(d) for g, d in avoidable_exact.items()},
        optimum_numerator=optimum.numerator, optimum_denominator=optimum.denominator,
        certificate_counts=cert_counts if fairness else dict.fromkeys(targets, 0),
        certificate_bytes=cert_cost if fairness else 0, violations=tuple(violations),
    )

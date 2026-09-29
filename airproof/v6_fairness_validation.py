"""Shared-trace v6 fairness validation helpers.

This module evaluates allocation mechanisms; it does not select an estimator or
authorize primary inference.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from types import SimpleNamespace

import numpy as np

from .metrics import coverage_metrics
from .scheduler import select_evidence
from .v6_fairness import CumulativeServiceState, select_cumulative_evidence


ALLOCATORS = ("v6_minimax", "utility_only", "v4_fallback")


@dataclass
class AllocationTrace:
    selected: list
    selection_times: dict[str, int]
    allocations: list[dict]
    counts: np.ndarray
    opportunities: np.ndarray
    metrics: dict


def _select(mode, state, candidates, *, budget, targets, epoch):
    if mode == "v6_minimax":
        return state.allocate(candidates, budget_bytes=budget, targets=targets,
                              allocation_epoch=epoch, fairness=True)
    if mode == "utility_only":
        return select_cumulative_evidence(candidates, budget_bytes=budget,
            targets=targets, allocation_epoch=epoch, fairness=False)
    if mode == "v4_fallback":
        return select_evidence(candidates, budget_bytes=budget, reserve_fraction=1.,
            targets=targets, fairness=True, constraint_mode="hard_if_feasible",
            allocation_epoch=epoch)
    raise ValueError(f"unknown allocator {mode}")


def _longest(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def allocate_shared_arrivals(world, arrivals, cfg, mode, *, window=24):
    """Allocate one immutable arrival map with q=30 and a finite byte budget."""
    if mode not in ALLOCATORS:
        raise ValueError(f"unknown allocator {mode}")
    steps = int(cfg["world"]["steps"]); burn = int(cfg["world"]["burn_in_steps"])
    lag = int(cfg["twin"]["fixed_lag"]); horizon = steps + lag
    groups = int(cfg["world"]["groups"])
    budget = int(cfg["scheduler"]["budget_bytes_per_epoch"])
    q = int(cfg["scheduler"]["target_contributors"])
    targets = {g: q for g in range(groups)}
    events = defaultdict(list)
    for item in world.observations:
        arrival = arrivals.get(item.nullifier)
        if arrival is not None and item.epoch <= arrival < horizon and arrival-item.epoch <= lag:
            events[arrival].append(item)
    backlog = {}; selected = []; selection_times = {}; allocations = []
    counts = np.zeros((horizon, groups), dtype=int)
    opportunities = np.zeros_like(counts)
    state = CumulativeServiceState()
    byte_violations = floor_violations = certificate_violations = 0
    for epoch in range(horizon):
        backlog.update((item.nullifier, item) for item in events.get(epoch, ()))
        backlog = {key: item for key, item in backlog.items() if epoch <= item.epoch + lag}
        result = _select(mode, state, backlog.values(), budget=budget, targets=targets, epoch=epoch)
        byte_violations += int(result.spent_bytes > budget)
        floor_violations += int(result.globally_feasible and mode != "utility_only" and not result.constraint_satisfied)
        certificate_violations += len(getattr(result, "violations", ()))
        counts[epoch] = [result.counts.get(g, 0) for g in range(groups)]
        available = getattr(result, "available_targets", None)
        if available is None:
            representatives = {(item.user_id, item.group) for item in backlog.values()}
            available = {g: min(q, sum(group == g for _, group in representatives)) for g in targets}
        opportunities[epoch] = [available[g] for g in range(groups)]
        unavoidable = getattr(result, "unavoidable_deficits",
            {g: (q-available[g])/q for g in targets})
        avoidable = getattr(result, "avoidable_deficits",
            {g: max(0, available[g]-result.counts.get(g, 0))/q for g in targets})
        allocations.append({"epoch": epoch, "spent": result.spent_bytes,
            "counts": dict(result.counts), "available": available,
            "avoidable": avoidable, "unavoidable": unavoidable,
            "feasible": result.globally_feasible})
        for item in result.selected:
            backlog.pop(item.nullifier, None)
            selection_times[item.nullifier] = epoch
        selected.extend(result.selected)
    scored_counts = counts[burn:steps]; scored_opportunities = opportunities[burn:steps]
    scored = [item for item in selected if item.epoch >= burn]
    rolling = {str(g): min((float(scored_counts[start:start+window, g].clip(max=q).sum()/(window*q))
                            for start in range(max(1, len(scored_counts)-window+1))), default=0.)
               for g in range(groups)}
    populations = {g: len({item.user_id for item in world.observations if item.group == g}) for g in range(groups)}
    unique = {g: len({item.user_id for item in scored if item.group == g}) for g in range(groups)}
    distinct_debt = {g: max(0, populations[g]-unique[g]) for g in range(groups)}
    metrics = {**coverage_metrics(scored, groups),
        "selected_records": len(scored), "selected_distinct_by_group": unique,
        "participant_population_by_group": populations, "distinct_user_debt_by_group": distinct_debt,
        "minimum_24epoch_capped_service_by_group": rolling,
        "minimum_24epoch_capped_service": min(rolling.values(), default=0.),
        "worst_zero_service_streak": max((_longest(scored_counts[:, g] == 0) for g in range(groups)), default=0),
        "worst_available_but_unserved_streak": max((_longest((scored_opportunities[:, g] > 0) & (scored_counts[:, g] == 0)) for g in range(groups)), default=0),
        "mean_max_opportunity_deficit": float(np.mean(np.max((q-scored_opportunities)/q, axis=1))),
        "mean_max_allocation_deficit": float(np.mean(np.max(np.maximum(0, scored_opportunities-scored_counts)/q, axis=1))),
        "max_spent_bytes": max(a["spent"] for a in allocations),
        "processing_budget_bytes": budget, "byte_violations": byte_violations,
        "feasible_floor_violations": floor_violations,
        "certificate_violations": certificate_violations,
        "globally_feasible_epochs": sum(a["feasible"] for a in allocations[burn:steps])}
    return AllocationTrace(selected, selection_times, allocations, counts, opportunities, metrics)


def outcome_strata_39(world, public, arrivals, live, reconstructed, selected, burn):
    """Return the fixed-prefix 39 spatial masks with error and service outcomes."""
    cells = world.truth.shape[1]; side = int(np.sqrt(cells))
    rc = np.indices((side, side)).reshape(2, -1); edge = max(1, side//8)
    peripheral = np.minimum.reduce([rc[0], rc[1], side-1-rc[0], side-1-rc[1]]) < edge
    ingress = np.zeros(cells, int)
    for record in world.observations:
        arrival = arrivals.get(record.nullifier)
        if arrival is not None and record.epoch <= arrival < burn and arrival-record.epoch <= 6:
            ingress[record.cell] += 1
    quarter = int(np.ceil(cells/4)); high = np.zeros(cells, bool); low = np.zeros(cells, bool)
    high[np.lexsort((np.arange(cells), public[:burn].mean(axis=0)))[-quarter:]] = True
    low[np.lexsort((np.arange(cells), ingress))[:quarter]] = True
    attrs = {"peripheral": peripheral, "high_public_exposure": high, "low_prefix_ingress": low}
    masks = {}
    for n in (1, 2, 3):
        for names in combinations(attrs, n):
            masks[" & ".join(names)] = np.logical_and.reduce([attrs[name] for name in names])
    for group in np.unique(world.cell_groups):
        for bits in product((0, 1), repeat=3):
            mask = world.cell_groups == group
            for name, bit in zip(attrs, bits):
                mask &= attrs[name] if bit else ~attrs[name]
            masks[f"zone={group}|" + "|".join(f"{name}={bit}" for name, bit in zip(attrs, bits))] = mask
    if len(masks) != 39:
        raise AssertionError(f"expected 39 strata, got {len(masks)}")
    truth = world.truth[burn:]; live_error = (live-truth)**2; error = (reconstructed-truth)**2
    scored = [item for item in selected if item.epoch >= burn]
    answer = {}
    for name, mask in masks.items():
        records = [item for item in scored if mask[item.cell]]
        answer[name] = {"cells": int(mask.sum()), "supported": int(mask.sum()) >= 8,
            "live_rmse": float(np.sqrt(live_error[:, mask].mean())) if mask.any() else None,
            "rmse": float(np.sqrt(error[:, mask].mean())) if mask.any() else None,
            "selected_records": len(records),
            "selected_distinct_contributors": len({(item.user_id, item.group) for item in records})}
    return answer


def collector_view(trace):
    return SimpleNamespace(selected=trace.selected, selection_times=trace.selection_times,
                           allocations=trace.allocations)

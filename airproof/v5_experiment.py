"""Integrated v5 numerical evaluations; original observations stay immutable.

Transport results are separate metadata. Historical v4 execution is never
substituted for a changed mechanism's outcome.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import time

import numpy as np

from .config import with_overrides, config_hash
from .continual_release import FixedReleasePlan, bounded_epoch_query, release_epoch
from .experiment import _v4_public_components
from .field import grid_laplacian
from .meteorology import grid_coordinates
from .metrics import prediction_metrics, coverage_metrics
from .rng import stream_seed
from .simulator import generate_world
from .v5_estimator import EstimatorConfig, enumerate_candidates, estimate_public_field
from .v5_fairness import select_minimax_evidence


CELLS = {
    "anchor_clean": {"network.availability": 1., "network.outage_median_hours": 0,
                     "network.outage_p95_hours": 0, "world.participation_skew": 1.,
                     "attack.kind": "clean", "attack.fraction": 0.},
    "outage_clean": {"network.availability": .75, "network.outage_median_hours": 6,
                     "network.outage_p95_hours": 24, "world.participation_skew": 5.,
                     "attack.kind": "clean", "attack.fraction": 0.},
    "severe_clean": {"attack.kind": "clean", "attack.fraction": .2},
    "severe_drift": {"attack.kind": "adversarial_drift", "attack.fraction": .2},
    "severe_hotspot": {"attack.kind": "hotspot_suppression", "attack.fraction": .2},
    "severe_inlier": {"attack.kind": "inlier", "attack.fraction": .2},
}


def cell_configuration(base: dict, cell: str, overrides: dict | None = None) -> dict:
    if cell not in CELLS:
        raise ValueError(f"unknown physical condition {cell}")
    return with_overrides(base, {**CELLS[cell], **(overrides or {})})


def method_specs(selected: EstimatorConfig, cell: str, *, factorial=True) -> list[dict]:
    specs = [dict(method="AP", fairness=True, relay=True, estimator=selected),
             dict(method="SQ", fairness=True, relay=True,
                  estimator=replace(selected, input_clip=False, output_cap=False)),
             dict(method="PUBLIC", fairness=True, relay=True, estimator=None),
             dict(method="HUBER", fairness=True, relay=True,
                  estimator=replace(selected, input_clip=False, output_cap=False, loss="huber", max_irls=max(200, selected.max_irls)))]
    if factorial and cell in ("severe_clean", "severe_hotspot"):
        for fairness, relay, robust in itertools.product((False, True), repeat=3):
            if fairness and relay:
                continue  # already AP/SQ; never count duplicate evaluations
            name = f"F{int(fairness)}R{int(relay)}B{int(robust)}"
            specs.append(dict(method=name, fairness=fairness, relay=relay,
                estimator=selected if robust else replace(selected, input_clip=False, output_cap=False)))
    return specs


def development_specs() -> list[dict]:
    specs, controls = [], set()
    for candidate in enumerate_candidates():
        specs.append(dict(method=candidate.candidate_id, fairness=True, relay=True, estimator=candidate))
        control_id = f"SQ_{candidate.objective}_reg{candidate.regularization_multiplier:g}"
        if control_id not in controls:
            controls.add(control_id)
            specs.append(dict(method=control_id, fairness=True, relay=True,
                              estimator=replace(candidate, input_clip=False, output_cap=False)))
    specs.append(dict(method="PUBLIC", fairness=True, relay=True, estimator=None))
    return specs


def reserve_contributions(records, release_arrivals: dict, *, steps: int, deadline: int = 24):
    """Source-local canonical mapping; delivery is a separate reserved-channel fact."""
    local, seen = defaultdict(list), set()
    for record in records:
        if record.source_class != "citizen" or not np.isfinite(record.value):
            raise ValueError("reserved release requires finite authenticated citizen values")
        key = (record.user_id, record.epoch, record.group, record.nullifier)
        if key in seen:
            continue
        seen.add(key)
        arrival = release_arrivals.get((record.user_id, record.epoch))
        if arrival is None or not record.epoch <= arrival <= record.epoch + deadline:
            continue
        if not 0 <= record.epoch < steps:
            raise ValueError("reserved contribution outside acquisition horizon")
        local[(record.epoch, record.user_id)].append(record)
    result = defaultdict(dict)
    for (epoch, user), items in sorted(local.items()):
        group = min(item.group for item in items)
        result[epoch][user] = [item for item in items if item.group == group]
    return dict(result)


def release_evidence(world, public: np.ndarray, release_arrivals: dict, cfg: dict, *,
                     dense_calibration: bool = False) -> tuple[dict, dict]:
    steps, cells = world.truth.shape
    groups, p = int(cfg["world"]["groups"]), cfg["privacy"]
    burn = int(cfg["world"]["burn_in_steps"])
    plan = FixedReleasePlan(steps, groups, int(p["k_min"]), float(p["epsilon_mean"]),
        float(p.get("epsilon_count", 0)), float(p["epsilon_user_max"]), bool(p.get("private_eligibility", False)), .01)
    deadline = int(p.get("release_deadline_steps", 24))
    contrib = reserve_contributions(world.observations, release_arrivals, steps=steps, deadline=deadline)
    baseline = np.column_stack([public[:, world.cell_groups == g].mean(axis=1) for g in range(groups)])
    truth = np.column_stack([world.truth[:, world.cell_groups == g].mean(axis=1) for g in range(groups)])
    outputs, summary = {}, {}
    for residual in (False, True):
        clipping = float(p["residual_clip"] if residual else p["clip"])
        rng = np.random.default_rng(stream_seed(world.seed, "privacy"))
        values = np.full((steps, groups), np.nan)
        query = np.full_like(values, np.nan)
        mask = np.zeros_like(values, dtype=bool)
        for epoch in plan.scheduled_epochs:
            clipping = float(p["residual_clip"] if residual else p["clip"])
            query[epoch], _ = bounded_epoch_query(contrib.get(epoch, {}), groups=groups,
                baselines=baseline[epoch], clip=clipping, k_min=plan.k_min, residual=residual)
            released = release_epoch(contrib.get(epoch, {}), epoch=epoch, plan=plan,
                public_baselines=baseline[epoch], clip=clipping, residual=residual, rng=rng)
            for item in released:
                if item.released:
                    mask[epoch, item.group] = True
                    values[epoch, item.group] = item.value
        scored = mask.copy(); scored[:burn] = False
        name = "residual" if residual else "raw"
        errors = values[scored] - truth[scored]
        deterministic = query[scored] - truth[scored]
        public_error = baseline[scored] - truth[scored]
        summary[name] = {
            "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            "same_support_public_rmse": float(np.sqrt(np.mean(public_error**2))) if errors.size else None,
            "deterministic_query_mse": float(np.mean(deterministic**2)) if errors.size else None,
            "laplace_variance": 2 * (4*clipping/(plan.k_min*plan.epsilon_mean))**2,
            "released_group_queries": int(scored.sum()),
            "scheduled_group_queries": int(sum(e >= burn for e in plan.scheduled_epochs)*groups),
        }
        outputs[name] = values
        outputs[f"{name}_query"] = query
        outputs[f"{name}_mask"] = mask
    if dense_calibration:
        # Synthetic calibration instrumentation only: no extra release, noise,
        # privacy spend, or admission. Never interpolate sparse scheduled queries.
        outputs["residual_query_dense"] = np.stack([
            bounded_epoch_query(contrib.get(epoch, {}), groups=groups,
                baselines=baseline[epoch], clip=float(p["residual_clip"]),
                k_min=plan.k_min, residual=True)[0]
            for epoch in range(steps)])
        summary["dense_query_calibration_only"] = True
        summary["dense_query_definition"] = "hourly bounded_epoch_query on same reserved contributions and deadline; no DP noise"
    summary.update({"epsilon_history": plan.composed_epsilon,
        "privacy_budget_violations": int(plan.composed_epsilon > 8+1e-10),
        "scheduled_epochs": list(plan.scheduled_epochs), "deadline": deadline,
        "admitted_user_epochs": sum(len(users) for users in contrib.values()),
        "scope": "release-only-user-history-given-public-topology; raw/residual-counterfactual-comparison"})
    return summary, {**outputs, "baseline": baseline, "truth": truth}


def select_arrived(world, arrivals: dict[str, int], cfg: dict, *, fairness: bool):
    steps = int(cfg["world"]["steps"]); lag = int(cfg["twin"]["fixed_lag"])
    burn = int(cfg["world"]["burn_in_steps"]); groups = int(cfg["world"]["groups"])
    horizon = steps + lag
    events, seen = defaultdict(list), set()
    retrospective = 0
    for item in world.observations:
        arrival = arrivals.get(item.nullifier)
        if arrival is None or arrival >= horizon or item.nullifier in seen:
            continue
        if arrival < item.epoch:
            raise ValueError("transport returned future-information arrival")
        seen.add(item.nullifier)
        if arrival-item.epoch > lag:
            retrospective += 1
        else:
            events[arrival].append(item)
    selected = []; counts = np.zeros((horizon, groups), dtype=int)
    feasible = violations = 0; avoidable = []; unavoidable = []
    for epoch in range(horizon):
        result = select_minimax_evidence(events.get(epoch, []),
            budget_bytes=int(cfg["scheduler"]["budget_bytes_per_epoch"]),
            targets={g: int(cfg["scheduler"]["target_contributors"]) for g in range(groups)},
            allocation_epoch=epoch, fairness=fairness)
        selected.extend(result.selected)
        counts[epoch] = [result.counts[g] for g in range(groups)]
        if burn <= epoch < steps:
            feasible += int(result.globally_feasible)
            violations += int(result.globally_feasible and fairness and not result.constraint_satisfied)
            avoidable.append(max(result.avoidable_deficits.values(), default=0))
            unavoidable.append(max(result.unavoidable_deficits.values(), default=0))
    scored = [item for item in selected if item.epoch >= burn]
    coverage = coverage_metrics(scored, groups)
    zero_streak = np.zeros(groups, dtype=int); worst_streak = 0
    for row in counts[burn:steps]:
        zero_streak = np.where(row == 0, zero_streak+1, 0)
        worst_streak = max(worst_streak, int(zero_streak.max()))
    return selected, {**coverage, "globally_feasible_epochs": feasible,
        "feasible_floor_violations": violations, "scored_epochs": steps-burn,
        "mean_max_avoidable_deficit": float(np.mean(avoidable)),
        "mean_max_unavoidable_deficit": float(np.mean(unavoidable)),
        "worst_zero_service_streak": worst_streak, "retrospective_ingress": retrospective}, counts


def outcome_strata(world, public, arrivals, live, reconstructed, burn: int) -> dict:
    """Fixed-prefix 39-mask audit using ACTUAL v5 arrivals, not v4 synthetic timestamps."""
    cells = world.truth.shape[1]; side = int(np.sqrt(cells))
    rc = np.indices((side, side)).reshape(2, -1); edge = max(1, side//8)
    peripheral = np.minimum.reduce([rc[0], rc[1], side-1-rc[0], side-1-rc[1]]) < edge
    ingress = np.zeros(cells, int)
    for record in world.observations:
        a = arrivals.get(record.nullifier)
        if a is not None and record.epoch <= a < burn and a-record.epoch <= 6:
            ingress[record.cell] += 1
    high = np.zeros(cells, bool); low = high.copy(); quarter = int(np.ceil(cells/4))
    high[np.lexsort((np.arange(cells), public[:burn].mean(axis=0)))[-quarter:]] = True
    low[np.lexsort((np.arange(cells), ingress))[:quarter]] = True
    attrs = {"peripheral": peripheral, "high_public_exposure": high, "low_prefix_ingress": low}
    masks = {}
    for n in (1, 2, 3):
        for names in itertools.combinations(attrs, n):
            masks[" & ".join(names)] = np.logical_and.reduce([attrs[name] for name in names])
    for group in np.unique(world.cell_groups):
        for bits in itertools.product((0, 1), repeat=3):
            mask = world.cell_groups == group
            for name, bit in zip(attrs, bits):
                mask = mask & (attrs[name] if bit else ~attrs[name])
            masks[f"zone={group}|" + "|".join(f"{name}={bit}" for name, bit in zip(attrs, bits))] = mask
    e1, e2 = (live[burn:]-world.truth[burn:])**2, (reconstructed[burn:]-world.truth[burn:])**2
    return {name: {"cells": int(mask.sum()), "supported": int(mask.sum()) >= 8,
        "live_rmse": float(np.sqrt(e1[:, mask].mean())) if mask.any() else None,
        "rmse": float(np.sqrt(e2[:, mask].mean())) if mask.any() else None} for name, mask in masks.items()}


def evaluate_prepared(world, public, operators, cfg, transport_by_policy, spec, *, strata=True, selection_cache=None):
    started = time.perf_counter()
    route = "airproof_deadline" if spec["relay"] else "direct"
    transport = transport_by_policy[route]
    selection_cache = {} if selection_cache is None else selection_cache
    key = (route, spec["fairness"])
    if key not in selection_cache:
        selection_cache[key] = select_arrived(world, transport.raw_arrivals, cfg, fairness=spec["fairness"])
    selected, scheduler, counts = selection_cache[key]
    arrivals = dict(transport.raw_arrivals)
    for record in world.reference_observations:
        arrivals[record.nullifier] = record.epoch
    if spec["estimator"] is None:
        live, reconstructed, diagnostics = public, public, {"solver_failure_rate": 0., "public_only": True}
    else:
        # Drain raw lag using public forecasts only; no future reference or truth.
        lag = int(cfg["twin"]["fixed_lag"])
        tail = []; last = public[-1].copy()
        for _ in range(lag):
            last = last.copy() if operators is None else np.asarray(operators[-1] @ last).ravel()
            tail.append(last.copy())
        extended = np.concatenate([public, np.asarray(tail)], axis=0) if lag else public
        extended_operators = None if operators is None else list(operators) + [operators[-1]] * lag
        result = estimate_public_field(extended, grid_coordinates(int(cfg["world"]["grid_side"])),
            [*selected, *world.reference_observations], spec["estimator"], arrival_map=arrivals,
            spatial_laplacian=grid_laplacian(int(cfg["world"]["grid_side"])), transitions=extended_operators)
        live, reconstructed, diagnostics = result.live[:len(public)], result.reconstructed[:len(public)], result.diagnostics
    burn = int(cfg["world"]["burn_in_steps"])
    reconstruction_metrics = prediction_metrics(world.truth[burn:], reconstructed[burn:], world.cell_groups)
    live_metrics = prediction_metrics(world.truth[burn:], live[burn:], world.cell_groups)
    threshold = float(np.quantile(world.truth[:burn], .95))
    events = world.truth[burn:] >= threshold
    event_count = int(events.sum())
    event_recall = float(np.mean(live[burn:][events] >= threshold)) if event_count else None
    row = {"seed": world.seed, "method": spec["method"], "config_hash": config_hash(cfg),
        "estimator": asdict(spec["estimator"]) if spec["estimator"] else None,
        "fairness": spec["fairness"], "relay": spec["relay"],
        "metrics": {**reconstruction_metrics, **scheduler,
                    **{f"live_{key}": value for key, value in live_metrics.items()},
                    "event_recall": event_recall, "event_support": event_count,
                    "event_threshold": threshold, "solver_failure_rate": diagnostics["solver_failure_rate"]},
        "diagnostics": {key: value for key, value in diagnostics.items() if key != "epoch_terms"},
        "transport": transport.metrics,
        "elapsed_seconds": time.perf_counter()-started}
    if strata:
        row["strata"] = outcome_strata(world, public, transport_by_policy["airproof_deadline"].raw_arrivals,
                                       live, reconstructed, burn)
    return row, {"live": live, "reconstructed": reconstructed, "counts": counts,
                 "terms": diagnostics.get("epoch_terms", [])}

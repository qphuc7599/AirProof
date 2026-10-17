from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
import tracemalloc
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from . import __version__
from .config import config_hash, validate_config
from .continual_release import FixedReleasePlan, independent_contributions, release_epoch
from .field import grid_laplacian
from .integrity import AtomicNullifierStore, MerkleTree
from .intersectional import audit_intersections
from .metrics import coverage_metrics, network_metrics, prediction_metrics
from .privacy import PrivacyAccountant, release_group_mean, release_group_residual_mean
from .records import Observation, canonical_json
from .reference import calibrated_public_reference, public_reference_fields
from .meteorology import grid_coordinates, transition_series
from .rng import stream_seed
from .scheduler import policy_weight_targets, population_targets, select_evidence
from .simulator import SyntheticWorld, generate_world
from .twin import FixedLagTwin


@dataclass(frozen=True)
class Method:
    name: str
    fairness: bool
    relay: bool
    huber: bool
    privacy: bool
    ledger: bool
    adaptive_huber: bool = False
    gated_correction: bool = False
    residual_correction: bool = False
    guarded_correction: bool = False
    predictive_residual: bool = False
    reference_anchored: bool = False
    final_architecture: bool = False


METHODS = {
    "airproof": Method("airproof", True, True, True, True, True),
    "centralized": Method("centralized", False, False, False, False, False),
    "air_quality_dt": Method("air_quality_dt", False, False, False, False, False),
    "qpta_adaptation": Method("qpta_adaptation", False, False, False, True, False),
    "opportunistic_ttl": Method("opportunistic_ttl", False, True, False, False, False),
    "blockchain_dt": Method("blockchain_dt", False, False, False, False, True),
    "no_fairness": Method("no_fairness", False, True, True, True, True),
    "no_relay": Method("no_relay", True, False, True, True, True),
    "squared_loss": Method("squared_loss", True, True, False, True, True),
    "airproof_adaptive": Method("airproof_adaptive", True, True, True, True, True, True),
    "airproof_gated": Method("airproof_gated", True, True, True, True, True, False, True),
    "airproof_residual": Method(
        "airproof_residual", True, True, True, True, True, residual_correction=True
    ),
    "airproof_residual_adaptive": Method(
        "airproof_residual_adaptive",
        True,
        True,
        True,
        True,
        True,
        adaptive_huber=True,
        residual_correction=True,
    ),
    "airproof_residual_guarded": Method(
        "airproof_residual_guarded",
        True,
        True,
        True,
        True,
        True,
        residual_correction=True,
        guarded_correction=True,
    ),
    "airproof_predictive_residual": Method(
        "airproof_predictive_residual",
        True,
        True,
        True,
        True,
        True,
        predictive_residual=True,
    ),
    "airproof_predictive_no_fairness": Method(
        "airproof_predictive_no_fairness",
        False,
        True,
        True,
        True,
        True,
        predictive_residual=True,
    ),
    "airproof_reference_anchored": Method(
        "airproof_reference_anchored", True, True, True, True, True,
        predictive_residual=True, reference_anchored=True,
    ),
    "airproof_v4": Method(
        "airproof_v4", True, True, True, True, True,
        predictive_residual=True, reference_anchored=True, final_architecture=True,
    ),
    "airproof_v4_no_fairness": Method(
        "airproof_v4_no_fairness", False, True, True, True, True,
        predictive_residual=True, reference_anchored=True, final_architecture=True,
    ),
    "airproof_v4_no_relay": Method(
        "airproof_v4_no_relay", True, False, True, True, True,
        predictive_residual=True, reference_anchored=True, final_architecture=True,
    ),
    "squared_loss_v4": Method(
        "squared_loss_v4", True, True, False, True, True, final_architecture=True,
    ),
    **{
        f"factorial_{fair}{relay}{huber}": Method(
            f"factorial_{fair}{relay}{huber}", bool(fair), bool(relay), bool(huber), True, True
        )
        for fair in (0, 1)
        for relay in (0, 1)
        for huber in (0, 1)
    },
}


@lru_cache(maxsize=1)
def source_tree_digest() -> str:
    """Hash executable Python sources so run IDs change even without a Git commit."""
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def registry_digest() -> str:
    """Bind result IDs to the claim, experiment and baseline registries."""
    repo_root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for name in ("claims_registry.yaml", "experiment_registry.yaml", "baseline_registry.yaml"):
        path = repo_root / "configs" / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _targets(config: dict[str, Any]) -> dict[int, int]:
    groups = int(config["world"].get("groups", 4))
    scheduler = config["scheduler"]
    mode = str(scheduler.get("target_mode", "constant"))
    if mode == "population":
        populations = {
            group: int(scheduler["populations"].get(str(group), scheduler["populations"].get(group)))
            for group in range(groups)
        }
        return population_targets(populations, float(scheduler["service_per_person"]))
    if mode == "policy_weight":
        weights = {
            group: float(scheduler["policy_weights"].get(str(group), scheduler["policy_weights"].get(group)))
            for group in range(groups)
        }
        return policy_weight_targets(weights, int(scheduler["target_total_slots"]))
    if mode != "constant":
        raise ValueError(f"unknown scheduler target mode: {mode}")
    target = int(config["scheduler"].get("target_contributors", 10))
    return {group: target for group in range(groups)}


def _select(
    world: SyntheticWorld, config: dict[str, Any], method: Method
) -> tuple[list[Observation], dict[str, float | int]]:
    by_epoch: dict[int, list[Observation]] = {}
    allocation_clock = str(config["scheduler"].get(
        "allocation_clock", "arrival" if method.final_architecture else "legacy_acquisition"
    ))
    retrospective_ingress = 0
    if allocation_clock == "arrival":
        events = []
        steps = int(config["world"]["steps"])
        for item in world.observations:
            arrival = item.relay_arrival if method.relay else item.direct_arrival
            if arrival is not None and item.epoch <= arrival < steps:
                events.append((arrival, item))
        events.sort(key=lambda pair: (pair[0], pair[1].epoch, pair[1].user_id, pair[1].nullifier))
        dedup = AtomicNullifierStore()
        try:
            for arrival, item in events:
                if not dedup.accept_once(item.nullifier):
                    continue
                if arrival - item.epoch > int(config["twin"]["fixed_lag"]):
                    retrospective_ingress += 1
                    continue
                by_epoch.setdefault(arrival, []).append(item)
        finally:
            dedup.close()
    elif allocation_clock == "legacy_acquisition":
        for item in world.observations:
            by_epoch.setdefault(item.epoch, []).append(item)
    else:
        raise ValueError("unknown scheduler allocation clock")
    selected: list[Observation] = []
    budget = int(config["scheduler"]["budget_bytes_per_epoch"])
    reserve = float(config["scheduler"]["reserve_fraction"])
    fairness_strength = float(config["scheduler"].get("fairness_strength", 1.0))
    constraint_mode = str(config["scheduler"].get("constraint_mode", "reserved"))
    max_deficits: list[float] = []
    opportunity_shortfalls = 0
    budget_infeasible = 0
    globally_feasible_epochs = 0
    constraint_satisfied_epochs = 0
    constraint_violation_when_feasible_epochs = 0
    burn_in = int(config["world"].get("burn_in_steps", 0))
    for epoch in range(int(config["world"]["steps"])):
        result = select_evidence(
            by_epoch.get(epoch, []),
            budget_bytes=budget,
            reserve_fraction=reserve,
            targets=_targets(config),
            fairness=method.fairness,
            fairness_strength=fairness_strength,
            constraint_mode=constraint_mode,
            allocation_epoch=epoch if allocation_clock == "arrival" else None,
        )
        selected.extend(result.selected)
        if epoch >= burn_in:
            max_deficits.append(result.max_deficit)
            opportunity_shortfalls += len(result.opportunity_shortfall_groups)
            budget_infeasible += len(result.budget_infeasible_groups)
            globally_feasible_epochs += int(result.globally_feasible)
            constraint_satisfied_epochs += int(result.constraint_satisfied)
            constraint_violation_when_feasible_epochs += int(
                result.globally_feasible and not result.constraint_satisfied
            )
    return selected, {
        "max_service_deficit": max(max_deficits, default=0.0),
        "mean_max_service_deficit": float(np.mean(max_deficits)) if max_deficits else 0.0,
        "opportunity_shortfall_group_epochs": opportunity_shortfalls,
        "budget_infeasible_group_epochs": budget_infeasible,
        "globally_feasible_epochs": globally_feasible_epochs,
        "constraint_satisfied_epochs": constraint_satisfied_epochs,
        "constraint_violation_when_feasible_epochs": constraint_violation_when_feasible_epochs,
        "scheduler_allocation_clock": allocation_clock,
        "scheduler_retrospective_ingress_records": retrospective_ingress,
        "scheduler_candidate_scope": "arrived-within-fixed-lag" if allocation_clock == "arrival" else "legacy-acquisition-census",
    }


def _deliver(
    selected: Iterable[Observation], method: Method, steps: int
) -> tuple[list[Observation], dict[str, int]]:
    arrivals_by_epoch: dict[int, list[Observation]] = {epoch: [] for epoch in range(steps)}
    arrival_lookup: dict[str, int] = {}
    accepted: list[Observation] = []
    nullifiers = AtomicNullifierStore()
    try:
        for item in selected:
            arrival = item.relay_arrival if method.relay else item.direct_arrival
            if arrival is None or arrival >= steps:
                continue
            if nullifiers.accept_once(item.nullifier):
                arrivals_by_epoch[arrival].append(item)
                arrival_lookup[item.nullifier] = arrival
                accepted.append(item)
    finally:
        nullifiers.close()
    return accepted, arrival_lookup


def _privacy_release_metrics(
    world: SyntheticWorld,
    accepted: list[Observation],
    config: dict[str, Any],
    *,
    enabled: bool,
    public_baselines: np.ndarray | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {
            "release_count": 0,
            "suppressed_release_count": 0,
            "release_rmse": None,
            "declined_privacy_users": 0,
            "privacy_transcript_scope": "not-applicable",
            "max_composed_user_epsilon": 0.0,
            "privacy_accountant_events": 0,
        }
    privacy_cfg = config["privacy"]
    accountant = PrivacyAccountant(float(privacy_cfg["epsilon_user_max"]))
    privacy_rng = np.random.default_rng(stream_seed(world.seed, "privacy"))
    release_errors: list[float] = []
    released = 0
    suppressed = 0
    total_released = 0
    total_suppressed = 0
    declined = 0
    total_declined = 0
    scope = ""
    steps = int(config["world"]["steps"])
    burn_in = int(config["world"].get("burn_in_steps", 0))
    groups = int(config["world"].get("groups", 4))
    release_mode = str(privacy_cfg.get("release_mode", "raw"))
    by_epoch: dict[int, list[Observation]] = {epoch: [] for epoch in range(steps)}
    for item in accepted:
        by_epoch[item.epoch].append(item)
    for epoch in range(steps):
        epoch_records = by_epoch[epoch]
        for group in range(groups):
            common = {
                "group": group,
                "epoch": epoch,
                "k_min": int(privacy_cfg["k_min"]),
                "epsilon_mean": float(privacy_cfg["epsilon_mean"]),
                "epsilon_count": float(privacy_cfg.get("epsilon_count", 0.0)),
                "accountant": accountant,
                "rng": privacy_rng,
                "private_eligibility": bool(privacy_cfg.get("private_eligibility", False)),
                "eligibility_safety_margin": privacy_cfg.get("eligibility_safety_margin"),
                "false_release_probability": float(
                    privacy_cfg.get("eligibility_false_release_probability", 1e-6)
                ),
            }
            if release_mode == "residual":
                cells = world.cell_groups == group
                if public_baselines is None:
                    public_baseline = float(
                        privacy_cfg.get(
                            "public_baseline_value", config["world"].get("baseline", 0.0)
                        )
                    )
                else:
                    public_baseline = float(np.mean(public_baselines[epoch, cells]))
                release = release_group_residual_mean(
                    epoch_records,
                    public_baseline=public_baseline,
                    residual_clip=float(privacy_cfg["residual_clip"]),
                    **common,
                )
            else:
                release = release_group_mean(
                    epoch_records,
                    clip=float(privacy_cfg["clip"]),
                    **common,
                )
            scope = release.transcript_scope
            total_declined += release.declined_users
            if release.released:
                total_released += 1
            else:
                total_suppressed += 1
            if epoch < burn_in:
                continue
            declined += release.declined_users
            if release.released:
                released += 1
                cells = world.cell_groups == group
                truth_mean = float(np.mean(world.truth[epoch, cells]))
                release_errors.append(float(release.value - truth_mean))
            else:
                suppressed += 1
    return {
        "release_count": released,
        "suppressed_release_count": suppressed,
        "release_rmse": float(np.sqrt(np.mean(np.square(release_errors))))
        if release_errors
        else None,
        "declined_privacy_users": declined,
        "privacy_transcript_scope": scope,
        "privacy_release_mode": release_mode,
        "max_composed_user_epsilon": accountant.max_spent,
        "privacy_accountant_events": len(accountant.events),
        "total_release_count_including_burn_in": total_released,
        "total_suppressed_release_count_including_burn_in": total_suppressed,
        "total_declined_privacy_users_including_burn_in": total_declined,
    }


def _v4_public_components(world: SyntheticWorld, config: dict[str, Any]):
    twin_cfg = config["twin"]
    side, steps = int(config["world"]["grid_side"]), int(config["world"]["steps"])
    operators = None
    transition_mode = str(twin_cfg.get("transition_mode", "public_weather"))
    if transition_mode == "public_weather":
        if world.public_meteorology is None:
            raise ValueError("v4 weather transition requires public weather covariates")
        world.public_meteorology.validate(steps)
        adjacency = -grid_laplacian(side)
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        operators = transition_series(
            adjacency, grid_coordinates(side), world.public_meteorology,
            transport=float(twin_cfg.get("weather_transport", .12)),
            directionality=float(twin_cfg.get("weather_directionality", 1.0)),
        )
    elif transition_mode != "identity":
        raise ValueError("unknown v4 transition mode")
    backbone = calibrated_public_reference(
        world.reference_observations, side=side, steps=steps,
        initial=float(config["world"].get("baseline", 12)),
        calibration_epochs=int(twin_cfg.get("reference_calibration_epochs", 48)),
        smoothing=float(twin_cfg.get("reference_smoothing", .001)), transitions=operators,
    )
    return backbone.fields, operators, {
        **backbone.diagnostics, "transition_mode": transition_mode,
        "meteorology_source": None if world.public_meteorology is None else world.public_meteorology.source,
    }


def _fixed_release_metrics(world: SyntheticWorld, config: dict[str, Any], method: Method,
                           public_baselines: np.ndarray) -> dict[str, Any]:
    cfg = config["privacy"]
    steps, groups = int(config["world"]["steps"]), int(config["world"].get("groups", 4))
    release_packet_bytes = int(cfg.get("reserved_bytes_per_user_epoch", 256))
    enrolled_users = int(config["world"]["agents"])
    reserved_capacity = int(cfg.get("reserved_budget_bytes_per_epoch", enrolled_users * release_packet_bytes))
    if release_packet_bytes < 1 or reserved_capacity < enrolled_users * release_packet_bytes:
        raise ValueError("independent DP ingress requires capacity reserved for every enrolled user")
    if public_baselines.shape != world.truth.shape or not np.all(np.isfinite(public_baselines)):
        raise ValueError("v4 release requires a complete public baseline field")
    plan = FixedReleasePlan(
        steps, groups, int(cfg["k_min"]), float(cfg["epsilon_mean"]),
        float(cfg.get("epsilon_count", 0)), float(cfg["epsilon_user_max"]),
        bool(cfg.get("private_eligibility", False)),
        float(cfg.get("eligibility_false_release_probability", .01)),
    )
    inputs = independent_contributions(
        list(world.observations), steps=steps, groups=groups, relay=method.relay,
        deadline_steps=int(cfg.get("release_deadline_steps", config["network"]["ttl_steps"])),
    )
    group_baselines = np.column_stack([
        public_baselines[:, world.cell_groups == group].mean(axis=1) for group in range(groups)
    ])
    residual = cfg.get("release_mode", "residual") == "residual"
    clip = float(cfg["residual_clip"] if residual else cfg["clip"])
    rng = np.random.default_rng(stream_seed(world.seed, "privacy"))
    burn_in = int(config["world"].get("burn_in_steps", 0))
    errors, total, scored = [], 0, 0
    scope = ""
    for epoch in range(steps):
        releases = release_epoch(inputs.get(epoch, {}), epoch=epoch, plan=plan,
                                 public_baselines=group_baselines[epoch], clip=clip,
                                 residual=residual, rng=rng)
        for release in releases:
            scope = release.transcript_scope
            total += int(release.released)
            if epoch >= burn_in and release.released:
                scored += 1
                truth = float(world.truth[epoch, world.cell_groups == release.group].mean())
                errors.append(float(release.value) - truth)
    return {
        "release_count": scored, "suppressed_release_count": (steps - burn_in) * groups - scored,
        "release_rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "declined_privacy_users": 0, "privacy_transcript_scope": scope,
        "privacy_release_mode": "residual" if residual else "raw",
        "max_composed_user_epsilon": plan.composed_epsilon,
        "privacy_accountant_events": len(plan.scheduled_epochs),
        "privacy_accounting_unit": "public-scheduled-vector-query-worst-case-user-history",
        "privacy_adjacency": "replacement-of-one-user-complete-history-including-group-changes",
        "privacy_query_admission": "independent-reserved-release-channel",
        "privacy_eligibility": "private-noisy-count" if plan.private_eligibility else "public-fixed-schedule-no-count-gate",
        "privacy_vector_sensitivity": 4 * clip / plan.k_min,
        "privacy_budget_violation_count": int(plan.composed_epsilon > plan.epsilon_user_max + 1e-10),
        "privacy_release_contributor_epochs": sum(len(users) for users in inputs.values()),
        "privacy_scheduled_epochs": len(plan.scheduled_epochs),
        "privacy_release_clock": "acquisition-epoch-plus-fixed-collection-deadline",
        "privacy_collection_deadline_epochs": int(cfg.get("release_deadline_steps", config["network"]["ttl_steps"])),
        "privacy_raw_scheduler_independent": True,
        "privacy_reserved_bytes_per_user_epoch": release_packet_bytes,
        "privacy_reserved_budget_bytes_per_epoch": reserved_capacity,
        "privacy_ingress_resource_model": "additional-per-credential-reserved-control-channel",
        "release_baseline_citizen_independent": True,
        "total_release_count_including_burn_in": total,
        "total_suppressed_release_count_including_burn_in": steps * groups - total,
        "total_declined_privacy_users_including_burn_in": 0,
    }


def run_method(world: SyntheticWorld, config: dict[str, Any], method_name: str) -> dict[str, Any]:
    validate_config(config)
    if method_name not in METHODS:
        raise KeyError(f"Unknown method {method_name!r}")
    method = METHODS[method_name]
    if method.final_architecture:
        if bool(config["twin"].get("predictive_adaptive_delta", False)):
            raise ValueError("final v4 architecture requires a single correction radius")
        if config["privacy"].get("admission_mode") != "independent_fixed_schedule":
            raise ValueError("final v4 architecture requires independent fixed-schedule release")
        if int(config["twin"].get("reference_calibration_epochs", 48)) > int(
            config["world"].get("burn_in_steps", 0)
        ):
            raise ValueError("public calibration must close before the scoring interval")
    trace_python_allocations = bool(config.get("execution", {}).get("trace_python_allocations", True))
    if trace_python_allocations:
        tracemalloc.start()
    elif tracemalloc.is_tracing():
        raise ValueError("unprofiled execution requires a worker without ambient allocation tracing")
    process = psutil.Process(os.getpid())
    peak_rss = [process.memory_info().rss]
    monitor_stop = threading.Event()

    def monitor_memory() -> None:
        while not monitor_stop.wait(0.01):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    started = time.perf_counter()
    selected, scheduler_metrics = _select(world, config, method)
    steps = int(config["world"]["steps"])
    accepted, arrival_lookup = _deliver(selected, method, steps)
    arrivals_by_epoch: dict[int, list[Observation]] = {epoch: [] for epoch in range(steps)}
    for item in accepted:
        arrivals_by_epoch[arrival_lookup[item.nullifier]].append(item)
    for item in world.reference_observations:
        arrivals_by_epoch[item.epoch].append(item)

    twin_cfg = config["twin"]
    side = int(config["world"]["grid_side"])
    initial = np.full(side * side, float(config["world"].get("baseline", 12.0)))
    v4_reference, transitions, public_diagnostics = (
        _v4_public_components(world, config) if method.final_architecture else (None, None, None)
    )
    twin = FixedLagTwin(
        side=side,
        steps=steps,
        fixed_lag=int(twin_cfg["fixed_lag"]),
        huber=method.huber,
        delta=float(twin_cfg["huber_delta"]),
        lambda_prior=float(twin_cfg["lambda_prior"]),
        lambda_spatial=float(twin_cfg["lambda_spatial"]),
        max_irls=int(twin_cfg["max_irls"]),
        tolerance=float(twin_cfg["tolerance"]),
        initial_state=initial,
        adaptive_huber=method.adaptive_huber,
        gated_correction=method.gated_correction,
        residual_correction=method.residual_correction,
        guarded_correction=method.guarded_correction,
        predictive_residual=method.predictive_residual,
        predictive_adaptive_delta=bool(twin_cfg.get("predictive_adaptive_delta", False)),
        predictive_clean_delta=float(twin_cfg.get("predictive_clean_delta", 4.0)),
        predictive_robust_delta=float(twin_cfg.get("predictive_robust_delta", 2.9)),
        predictive_activation_gain=float(
            twin_cfg.get("predictive_activation_gain", 1.0)
        ),
        predictive_calibrated_gate=bool(
            twin_cfg.get("predictive_calibrated_gate", False)
        ),
        predictive_calibration_epochs=int(
            twin_cfg.get("predictive_calibration_epochs", 48)
        ),
        predictive_excess_gate_start=float(
            twin_cfg.get("predictive_excess_gate_start", 0.01)
        ),
        predictive_excess_gate_full=float(
            twin_cfg.get("predictive_excess_gate_full", 0.04)
        ),
        clean_delta=float(twin_cfg.get("adaptive_clean_delta", 6.0)),
        robust_delta=float(twin_cfg.get("adaptive_robust_delta", twin_cfg["huber_delta"])),
        adaptive_tail_z=float(twin_cfg.get("adaptive_tail_z", 2.5)),
        adaptive_clean_tail_fraction=float(
            twin_cfg.get("adaptive_clean_tail_fraction", 0.02)
        ),
        adaptive_contaminated_tail_fraction=float(
            twin_cfg.get("adaptive_contaminated_tail_fraction", 0.15)
        ),
        innovation_gate_start=float(twin_cfg.get("innovation_gate_start", 0.02)),
        innovation_gate_full=float(twin_cfg.get("innovation_gate_full", 0.12)),
        innovation_gate_tail_z=float(twin_cfg.get("innovation_gate_tail_z", 3.5)),
        correction_delta=float(twin_cfg.get("correction_delta", 3.0)),
        lambda_correction=float(twin_cfg.get("lambda_correction", 0.5)),
        correction_clip=float(twin_cfg.get("correction_clip", 8.0)),
        correction_clean_gain=float(twin_cfg.get("correction_clean_gain", 0.25)),
        correction_attack_gain=float(twin_cfg.get("correction_attack_gain", 3.0)),
        correction_gate_start=float(twin_cfg.get("correction_gate_start", 0.15)),
        correction_gate_full=float(twin_cfg.get("correction_gate_full", 0.22)),
        correction_gate_ewma=float(twin_cfg.get("correction_gate_ewma", 0.25)),
        transitions=transitions,
    )
    epoch_update_seconds: list[float] = []
    public_reference = (
        public_reference_fields(
            world.reference_observations,
            side=side,
            steps=steps,
            initial=float(config["world"].get("baseline", 12.0)),
            smoothing=float(twin_cfg.get("reference_smoothing", 0.001)),
        )
        if method.reference_anchored and not method.final_architecture else None
    )
    if method.final_architecture:
        public_reference = v4_reference if method.reference_anchored else None
    live_states = np.empty_like(twin.states) if method.final_architecture else None
    for epoch in range(steps):
        epoch_started = time.perf_counter()
        start = max(0, epoch - twin.fixed_lag)
        twin.ingest_at(
            epoch, arrivals_by_epoch[epoch],
            external_predictor=(public_reference[start : epoch + 1]
                                if public_reference is not None else None),
        )
        epoch_update_seconds.append(time.perf_counter() - epoch_started)
        if live_states is not None:
            live_states[epoch] = twin.states[epoch]

    privacy_metrics = (_fixed_release_metrics(world, config, method, v4_reference)
                       if method.final_architecture else _privacy_release_metrics(
        world,
        accepted,
        config,
        enabled=method.privacy,
        public_baselines=twin.release_baselines,
    ))

    root = None
    if method.ledger:
        root = MerkleTree(canonical_json(item.public_dict()) for item in accepted).root_hex
    elapsed = time.perf_counter() - started
    monitor_stop.set()
    monitor.join(timeout=1.0)
    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    python_peak = None
    if trace_python_allocations:
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    metrics: dict[str, Any] = {}
    burn_in = int(config["world"].get("burn_in_steps", 0))
    scoring_selected = [item for item in selected if item.epoch >= burn_in]
    scoring_accepted = [item for item in accepted if item.epoch >= burn_in]
    metrics.update(
        prediction_metrics(world.truth[burn_in:], twin.states[burn_in:], world.cell_groups)
    )
    if method.final_architecture:
        live_metrics = prediction_metrics(world.truth[burn_in:], live_states[burn_in:], world.cell_groups)
        metrics.update({f"live_{key}": value for key, value in live_metrics.items()})
        metrics["prediction_clock"] = "bounded-fixed-lag-reconstruction-at-horizon-end"
        metrics["live_prediction_clock"] = "frozen-at-own-epoch-before-future-arrivals"
        metrics["maximum_reconstruction_lag_epochs"] = twin.fixed_lag
        audit_started = time.perf_counter()
        metrics["intersectional_audit"] = audit_intersections(
            world, twin.states, live_states, accepted, v4_reference,
            burn_in=burn_in, fixed_lag=twin.fixed_lag,
        )
        metrics["intersectional_audit_seconds"] = time.perf_counter() - audit_started
    if method.residual_correction or method.predictive_residual:
        backbone_metrics = prediction_metrics(
            world.truth[burn_in:], twin.predictive_states[burn_in:], world.cell_groups
        )
        metrics.update(
            {
                "backbone_rmse": backbone_metrics["rmse"],
                "backbone_worst_group_rmse": backbone_metrics["worst_group_rmse"],
            }
        )
    metrics.update(network_metrics(scoring_selected, scoring_accepted, arrival_lookup, steps))
    metrics.update(
        coverage_metrics(scoring_accepted, int(config["world"].get("groups", 4)))
    )
    metrics.update(scheduler_metrics)
    metrics.update(privacy_metrics)
    malicious = sum(item.corrupted for item in accepted)
    metrics.update(
        {
            "accepted_malicious_fraction": malicious / len(accepted) if accepted else 0.0,
            "solver_failure_rate": twin.failure_rate,
            "epoch_update_p50_seconds": float(np.percentile(epoch_update_seconds, 50)),
            "epoch_update_p95_seconds": float(np.percentile(epoch_update_seconds, 95)),
            "epoch_update_max_seconds": max(epoch_update_seconds),
            "p95_epoch_update_lt_epoch_duration": bool(
                np.percentile(epoch_update_seconds, 95)
                < float(config["world"].get("epoch_hours", 1.0)) * 3600
            ),
            "retrospective_late_records": len(twin.retrospective_records),
            "adaptive_huber": method.adaptive_huber,
            "gated_robust_correction": method.gated_correction,
            "residual_robust_correction": method.residual_correction,
            "guarded_robust_correction": method.guarded_correction,
            "predictive_residual_assimilation": method.predictive_residual,
            "reference_anchored_assimilation": method.reference_anchored,
            "final_public_architecture": method.final_architecture,
            "public_backbone_calibration": public_diagnostics,
            "release_baseline_source": (
                "public_regulatory_calibrated_weather" if method.final_architecture
                else "public_regulatory_thin_plate" if method.reference_anchored
                else "internal_prebatch_history"
            ),
            "release_baseline_citizen_independent": method.reference_anchored or method.final_architecture,
            "predictive_adaptive_delta": twin.predictive_adaptive_delta,
            "predictive_delta_mean": float(np.mean(twin.predictive_delta_history)),
            "predictive_delta_min": min(twin.predictive_delta_history, default=0.0),
            "predictive_delta_max": max(twin.predictive_delta_history, default=0.0),
            "predictive_tail_fraction_mean": float(
                np.mean(twin.predictive_tail_fraction_history)
            ),
            "predictive_robust_activation_mean": float(
                np.mean(twin.predictive_activation_by_epoch)
            ),
            "adaptive_huber_delta_mean": float(
                np.mean([item.delta for item in twin.adaptive_history])
            ),
            "adaptive_huber_contamination_mean": float(
                np.mean([item.estimated_contamination for item in twin.adaptive_history])
            ),
            "adaptive_huber_location_abs_mean": float(
                np.mean([abs(item.residual_location) for item in twin.adaptive_history])
            ),
            "robust_gate_mean": float(np.mean(twin.robust_gate_history)),
            "innovation_contamination_mean": float(
                np.mean(twin.innovation_contamination_history)
            ),
            "correction_abs_mean": float(
                np.mean(twin.correction_abs_mean_history)
            )
            if twin.correction_abs_mean_history
            else 0.0,
            "correction_abs_max": max(twin.correction_abs_max_history, default=0.0),
            "correction_tail_fraction_mean": float(
                np.mean(twin.correction_tail_fraction_history)
            )
            if twin.correction_tail_fraction_history
            else 0.0,
            "correction_activation_mean": float(
                np.mean(twin.correction_activation_history)
            ),
            "closed_tail_ewma_mean": float(np.mean(twin.closed_tail_ewma_history)),
            "burn_in_steps": burn_in,
            "scoring_steps": steps - burn_in,
            "runtime_seconds": elapsed,
            "peak_memory_mb": peak_rss[0] / (1024 * 1024),
            "python_peak_alloc_mb": None if python_peak is None else python_peak / (1024 * 1024),
            "python_allocation_tracing_enabled": trace_python_allocations,
            "rss_monitor_sampling_interval_seconds": 0.01,
            "merkle_root": root,
        }
    )
    cfg_hash = config_hash(config)
    source_digest = source_tree_digest()
    registry_sha = registry_digest()
    run_id = hashlib.sha256(
        f"{__version__}:{source_digest}:{registry_sha}:{cfg_hash}:{world.seed}:{method.name}".encode()
    ).hexdigest()[:20]
    return {
        "run_id": run_id,
        "code_version": __version__,
        "source_tree_sha256": source_digest,
        "registry_sha256": registry_sha,
        "config_hash": cfg_hash,
        "seed": world.seed,
        "method": method.name,
        "scenario": {
            "agents": int(config["world"]["agents"]),
            "reference_stations": int(config["world"].get("reference_station_count", 0)),
            "grid_side": int(config["world"]["grid_side"]),
            "steps": int(config["world"]["steps"]),
            "participation_skew": float(config["world"].get("participation_skew", 1.0)),
            "availability": float(config["network"].get("availability", 1.0)),
            "outage_median_hours": float(
                config["network"].get("outage_median_hours", 1.0)
            ),
            "outage_p95_hours": float(config["network"].get("outage_p95_hours", 1.0)),
            "attack_fraction": float(config["attack"].get("fraction", 0.0)),
            "attack_kind": str(config["attack"].get("kind", "clean")),
            "epsilon_mean": float(config["privacy"]["epsilon_mean"]),
            "epsilon_count": float(config["privacy"].get("epsilon_count", 0.0)),
            "epsilon_user_max": float(config["privacy"]["epsilon_user_max"]),
            "privacy_release_mode": str(config["privacy"].get("release_mode", "raw")),
            "residual_clip": float(config["privacy"].get("residual_clip", 0.0)),
            "private_eligibility": bool(config["privacy"].get("private_eligibility", False)),
            "k_min": int(config["privacy"]["k_min"]),
            "huber_delta": float(config["twin"]["huber_delta"]),
            "lambda_prior": float(config["twin"]["lambda_prior"]),
            "lambda_spatial": float(config["twin"]["lambda_spatial"]),
            "fixed_lag": int(config["twin"]["fixed_lag"]),
            "residual_correction": method.residual_correction,
            "predictive_residual": method.predictive_residual,
            "correction_delta": float(config["twin"].get("correction_delta", 3.0)),
            "predictive_adaptive_delta": bool(
                config["twin"].get("predictive_adaptive_delta", False)
            ),
            "predictive_clean_delta": float(
                config["twin"].get("predictive_clean_delta", 4.0)
            ),
            "predictive_robust_delta": float(
                config["twin"].get("predictive_robust_delta", 2.9)
            ),
            "predictive_activation_gain": float(
                config["twin"].get("predictive_activation_gain", 1.0)
            ),
            "predictive_calibrated_gate": bool(
                config["twin"].get("predictive_calibrated_gate", False)
            ),
            "predictive_calibration_epochs": int(
                config["twin"].get("predictive_calibration_epochs", 48)
            ),
            "predictive_excess_gate_start": float(
                config["twin"].get("predictive_excess_gate_start", 0.01)
            ),
            "predictive_excess_gate_full": float(
                config["twin"].get("predictive_excess_gate_full", 0.04)
            ),
            "lambda_correction": float(config["twin"].get("lambda_correction", 0.5)),
            "correction_clip": float(config["twin"].get("correction_clip", 8.0)),
            "correction_clean_gain": float(
                config["twin"].get("correction_clean_gain", 0.25)
            ),
            "correction_attack_gain": float(
                config["twin"].get("correction_attack_gain", 3.0)
            ),
            "correction_gate_start": float(
                config["twin"].get("correction_gate_start", 0.15)
            ),
            "correction_gate_full": float(
                config["twin"].get("correction_gate_full", 0.22)
            ),
            "correction_gate_ewma": float(
                config["twin"].get("correction_gate_ewma", 0.25)
            ),
            "reserve_fraction": float(config["scheduler"]["reserve_fraction"]),
            "fairness_strength": float(config["scheduler"].get("fairness_strength", 1.0)),
            "target_contributors": int(config["scheduler"].get("target_contributors", 0)),
            "constraint_mode": str(config["scheduler"].get("constraint_mode", "reserved")),
        },
        "world": world.metadata,
        "metrics": metrics,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def run_privacy_replay(
    world: SyntheticWorld, config: dict[str, Any], method_name: str = "airproof"
) -> dict[str, Any]:
    """Evaluate the protected release layer without rerunning the graph solver.

    The synthetic world, scheduler and delivery realization remain identical to the
    corresponding end-to-end method. Reference-anchored replay also shares its
    independently public baseline. Other replay methods use the configured constant
    public baseline, not an uncomputed internal predictor history.
    """
    validate_config(config)
    if method_name not in METHODS:
        raise KeyError(f"Unknown method {method_name!r}")
    method = METHODS[method_name]
    if not method.privacy:
        raise ValueError("privacy replay requires a method with the privacy layer enabled")
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    started = time.perf_counter()
    selected, scheduler_metrics = _select(world, config, method)
    steps = int(config["world"]["steps"])
    accepted, arrival_lookup = _deliver(selected, method, steps)
    replay_baselines = (
        public_reference_fields(
            world.reference_observations,
            side=int(config["world"]["grid_side"]),
            steps=steps,
            initial=float(config["world"].get("baseline", 12.0)),
            smoothing=float(config["twin"].get("reference_smoothing", 0.001)),
        ) if method.reference_anchored and not method.final_architecture else None
    )
    if method.final_architecture:
        replay_baselines, _, public_diagnostics = _v4_public_components(world, config)
    privacy_metrics = (_fixed_release_metrics(world, config, method, replay_baselines)
                       if method.final_architecture else _privacy_release_metrics(
        world, accepted, config, enabled=True, public_baselines=replay_baselines
    ))
    privacy_metrics["release_baseline_source"] = (
        "public_regulatory_calibrated_weather" if method.final_architecture
        else "public_regulatory_thin_plate" if method.reference_anchored else "constant_public"
    )
    if method.final_architecture:
        privacy_metrics["public_backbone_calibration"] = public_diagnostics
    privacy_metrics["release_baseline_citizen_independent"] = True
    burn_in = int(config["world"].get("burn_in_steps", 0))
    scoring_selected = [item for item in selected if item.epoch >= burn_in]
    scoring_accepted = [item for item in accepted if item.epoch >= burn_in]
    metrics: dict[str, Any] = {}
    metrics.update(network_metrics(scoring_selected, scoring_accepted, arrival_lookup, steps))
    metrics.update(
        coverage_metrics(scoring_accepted, int(config["world"].get("groups", 4)))
    )
    metrics.update(scheduler_metrics)
    metrics.update(privacy_metrics)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics.update(
        {
            "burn_in_steps": burn_in,
            "scoring_steps": steps - burn_in,
            "runtime_seconds": time.perf_counter() - started,
            "peak_memory_mb": process.memory_info().rss / (1024 * 1024),
            "python_peak_alloc_mb": python_peak / (1024 * 1024),
        }
    )
    cfg_hash = config_hash(config)
    source_digest = source_tree_digest()
    registry_sha = registry_digest()
    run_id = hashlib.sha256(
        f"{__version__}:{source_digest}:{registry_sha}:{cfg_hash}:{world.seed}:"
        f"{method.name}:privacy-replay".encode()
    ).hexdigest()[:20]
    return {
        "run_id": run_id,
        "code_version": __version__,
        "source_tree_sha256": source_digest,
        "registry_sha256": registry_sha,
        "config_hash": cfg_hash,
        "seed": world.seed,
        "method": f"{method.name}_privacy_replay",
        "execution_mode": "privacy-replay",
        "scenario": {
            "agents": int(config["world"]["agents"]),
            "grid_side": int(config["world"]["grid_side"]),
            "steps": steps,
            "participation_skew": float(config["world"].get("participation_skew", 1.0)),
            "availability": float(config["network"].get("availability", 1.0)),
            "outage_median_hours": float(
                config["network"].get("outage_median_hours", 1.0)
            ),
            "outage_p95_hours": float(config["network"].get("outage_p95_hours", 1.0)),
            "attack_fraction": float(config["attack"].get("fraction", 0.0)),
            "attack_kind": str(config["attack"].get("kind", "clean")),
            "epsilon_mean": float(config["privacy"]["epsilon_mean"]),
            "epsilon_count": float(config["privacy"].get("epsilon_count", 0.0)),
            "epsilon_user_max": float(config["privacy"]["epsilon_user_max"]),
            "privacy_release_mode": str(config["privacy"].get("release_mode", "raw")),
            "residual_clip": float(config["privacy"].get("residual_clip", 0.0)),
            "private_eligibility": bool(config["privacy"].get("private_eligibility", False)),
            "k_min": int(config["privacy"]["k_min"]),
            "huber_delta": float(config["twin"]["huber_delta"]),
            "lambda_prior": float(config["twin"]["lambda_prior"]),
            "lambda_spatial": float(config["twin"]["lambda_spatial"]),
            "fixed_lag": int(config["twin"]["fixed_lag"]),
            "reserve_fraction": float(config["scheduler"]["reserve_fraction"]),
            "fairness_strength": float(config["scheduler"].get("fairness_strength", 1.0)),
            "constraint_mode": str(config["scheduler"].get("constraint_mode", "reserved")),
        },
        "world": world.metadata,
        "metrics": metrics,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def run_experiment(config: dict[str, Any], seeds: Iterable[int], methods: Iterable[str]) -> list[dict[str, Any]]:
    validate_config(config)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        world = generate_world(config, int(seed))
        for method in methods:
            results.append(run_method(world, config, method))
    return results


def write_results(results: list[dict[str, Any]], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for item in results:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    flat = [
        {"run_id": item["run_id"], "config_hash": item["config_hash"], "seed": item["seed"], "method": item["method"], **item["scenario"], **item["metrics"]}
        for item in results
    ]
    frame = pd.DataFrame(flat)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    frame.to_parquet(output.with_suffix(".parquet"), index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "artifact_status": "world-level-results",
        "jsonl_sha256": digest,
        "rows": len(results),
        "run_ids": sorted(item["run_id"] for item in results),
        "config_hashes": sorted({item["config_hash"] for item in results}),
        "code_version": __version__,
        "source_tree_sha256": source_tree_digest(),
        "registry_sha256": registry_digest(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def read_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows

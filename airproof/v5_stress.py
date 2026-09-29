"""Frozen one-factor stress transformations; no candidate search or implicit scoring."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import numpy as np
from .config import with_overrides
from .experiment import _v4_public_components
from .field import grid_laplacian
from .meteorology import grid_coordinates
from .metrics import prediction_metrics
from .v5_estimator import EstimatorConfig, enumerate_candidates, estimate_public_field
from .v5_experiment import method_specs, select_arrived
from .v5_transport import generate_transport_trace, simulate_transport

STRESS_SEEDS = (915004, 915005, 915006, 915007)


@dataclass(frozen=True)
class StressCase:
    name: str
    reference_count: int = 16
    reference_delay: int = 0
    reference_bias: float = 0.
    corruption: float = 0.
    attack: str = "clean"
    bandwidth: int = 1024
    local_event: bool = False


def stress_matrix():
    return [StressCase("baseline")] + [StressCase(f"references_{n}", reference_count=n) for n in (8, 4)] + [
        StressCase(f"reference_delay_{d}", reference_delay=d) for d in (6, 24)] + [
        StressCase(f"reference_bias_{b}", reference_bias=b) for b in (4., 8.)] + [
        StressCase(f"{kind}_{fraction:g}", attack=kind, corruption=fraction)
        for kind in ("drift", "hotspot", "inlier") for fraction in (.1, .2, .4)] + [
        StressCase(f"bandwidth_{b}", bandwidth=b) for b in (512, 2048)] + [
        StressCase("local_event_absent_reference", local_event=True)]


def configuration_for_stress(base, case):
    """Generate the SAME clean physical world before immutable measurement transforms."""
    cfg = with_overrides(base, {"attack.kind": "clean", "attack.fraction": 0.,
                               "world.reference_station_count": 16})
    cfg["transport"] = copy.deepcopy(base["transport"])
    cfg["transport"]["capacity_bytes_per_direction"] = case.bandwidth
    return cfg


def transform_world(world, cfg, case):
    """Copy truth only for the registered event; never rewrite input observations."""
    if case not in stress_matrix():
        raise ValueError("unregistered stress case")
    cells = sorted({r.cell for r in world.reference_observations})
    if len(cells) < case.reference_count:
        raise ValueError("insufficient source reference stations")
    kept = set(cells[:case.reference_count])  # nested, outcome-independent geometry subset
    refs = tuple(replace(r, value=r.value + case.reference_bias,
                         direct_arrival=r.epoch + case.reference_delay,
                         relay_arrival=r.epoch + case.reference_delay)
                 for r in world.reference_observations if r.cell in kept)
    steps, count = world.truth.shape
    delta = np.zeros_like(world.truth)
    event_cells = []
    if case.local_event:
        side = int(cfg["world"]["grid_side"])
        xy = grid_coordinates(side)
        distance = np.min(((xy[:, None] - xy[np.array(cells)][None])**2).sum(axis=2), axis=1)
        available = [int(i) for i in np.argsort(-distance, kind="stable") if i not in cells]
        if not available:
            raise ValueError("local event requires cells without reference stations")
        center = available[0]
        ranked = np.argsort(((xy-xy[center])**2).sum(axis=1), kind="stable")
        event_cells = [int(i) for i in ranked if i not in cells][:max(1, count//64)]
        start = max(int(cfg["world"]["burn_in_steps"]), steps//2)
        delta[start:min(steps, start+24), event_cells] = 8.
    agents = int(cfg["world"]["agents"])
    ordering = sorted(range(agents), key=lambda u: hashlib.sha256(f"stress-users:{world.seed}:{u}".encode()).digest())
    corrupted = set(ordering[:round(case.corruption*agents)])
    attack_start = int(cfg["attack"].get("start_epoch", steps//3))
    amplitude = float(cfg["attack"].get("amplitude", 12.))
    observations = []
    for r in world.observations:
        value = r.value + delta[r.epoch, r.cell]
        attacked = r.user_id in corrupted and r.epoch >= attack_start
        if attacked:
            if case.attack == "drift":
                value += amplitude*(r.epoch-attack_start+1)/max(1, steps-attack_start)
            elif case.attack == "hotspot":
                # Fixed spatial group, no private truth or future-error oracle.
                value -= amplitude if r.group == 0 else 0.
            elif case.attack == "inlier":
                value += min(amplitude, 1.25*r.sigma)
        observations.append(replace(r, value=float(value), corrupted=bool(attacked)))
    metadata = copy.deepcopy(world.metadata)
    metadata["v5_stress"] = {"case": asdict(case), "reference_cells": sorted(kept),
        "reference_latest_acquisition_at_t": f"t-{case.reference_delay}",
        "event_cells": event_cells, "event_amplitude": 8. if case.local_event else 0.,
        "event_duration_max_epochs": 24 if case.local_event else 0,
        "corrupted_user_count": len(corrupted), "attack_truth_oracle": False}
    return replace(world, truth=world.truth.copy()+delta, observations=tuple(observations),
                   reference_observations=refs, metadata=metadata), delta > 0


def prepare_stress_public(world, cfg, case):
    """Delay reference-only backbone AND learned-model availability; no truth input."""
    # Existing reference builder requires acquisition-indexed on-time records.
    # Its causal output at s is made available only at s+delay below.
    refs = tuple(replace(r, direct_arrival=r.epoch, relay_arrival=r.epoch)
                 for r in world.reference_observations)
    reference_only = replace(world, truth=np.zeros_like(world.truth), observations=(),
                             reference_observations=refs)
    fields, operators, diagnostics = _v4_public_components(reference_only, cfg)
    delayed = np.full_like(fields, float(cfg["world"].get("baseline", 12.)))
    d = case.reference_delay
    if d < len(fields):
        delayed[d:] = fields[:len(fields)-d] if d else fields
    diagnostics = {**diagnostics, "reference_delay": d, "latest_reference_acquisition": [t-d if t >= d else None for t in range(len(fields))],
        "first_available_selected_model_epoch": int(diagnostics.get("first_selected_model_epoch", cfg["twin"].get("reference_calibration_epochs",48)))+d,
        "reference_bias_applied_to_measurements": case.reference_bias,
        "truth_or_citizens_provided_to_public_builder": False,
        "delay_policy": "hold-delayed-causal-reference-field; current-public-weather-transitions-separate"}
    return delayed, operators, diagnostics


def validate_selected(selected):
    if not isinstance(selected, EstimatorConfig) or asdict(selected) not in [asdict(c) for c in enumerate_candidates()]:
        raise ValueError("explicit registered selected estimator required; no new search")


def score_case(world, public, operators, cfg, selected, event_mask):
    """Root must explicitly authorize scoring after selection; one matched four-method set."""
    validate_selected(selected)
    trace = generate_transport_trace(cfg, world.seed)
    transport = simulate_transport(trace, world.observations, cfg)
    records, scheduler, _ = select_arrived(world, transport.raw_arrivals, cfg, fairness=True)
    arrivals = dict(transport.raw_arrivals)
    arrivals.update({r.nullifier: r.direct_arrival for r in world.reference_observations})
    lag = selected.lag
    tail, last = [], public[-1].copy()
    for _ in range(lag):
        last = last.copy() if operators is None else np.asarray(operators[-1] @ last).ravel()
        tail.append(last.copy())
    extended = np.concatenate([public, np.asarray(tail)]) if lag else public
    extended_ops = None if operators is None else list(operators)+[operators[-1]]*lag
    burn = int(cfg["world"]["burn_in_steps"])
    rows = []
    for spec in method_specs(selected, "severe_clean", factorial=False):
        if spec["estimator"] is None:
            live = reconstructed = public
            diagnostics = {"solver_failure_rate": 0.}
        else:
            result = estimate_public_field(extended, grid_coordinates(int(cfg["world"]["grid_side"])),
                [*records, *world.reference_observations], spec["estimator"], arrival_map=arrivals,
                spatial_laplacian=grid_laplacian(int(cfg["world"]["grid_side"])), transitions=extended_ops)
            live, reconstructed, diagnostics = result.live[:len(public)], result.reconstructed[:len(public)], result.diagnostics
        row = {"seed": world.seed, "case": world.metadata["v5_stress"]["case"]["name"], "method": spec["method"],
            "live": prediction_metrics(world.truth[burn:], live[burn:], world.cell_groups),
            "reconstructed": prediction_metrics(world.truth[burn:], reconstructed[burn:], world.cell_groups),
            "scheduler": scheduler, "transport": transport.metrics, "solver_failure_rate": diagnostics["solver_failure_rate"],
            "support": int(world.truth[burn:].size), "event_support": int(event_mask[burn:].sum()),
            "event_live_rmse": float(np.sqrt(np.mean((live[event_mask]-world.truth[event_mask])**2))) if event_mask.any() else None,
            "clock": "frozen-live-and-final-fixed-lag-reconstruction", "scope": "prespecified-descriptive-OAT-stress-not-selection"}
        rows.append(row)
    return rows

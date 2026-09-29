"""Bounded integration checks; these are not campaign outcome evidence."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airproof.config import load_config, with_overrides
from airproof.continual_release import FixedReleasePlan
from airproof.experiment import _v4_public_components
from airproof.records import Observation
from airproof.simulator import generate_world
from airproof.v5_estimator import EstimatorConfig
from airproof.v5_experiment import (development_specs, evaluate_prepared, method_specs,
                                    release_evidence, reserve_contributions, select_arrived)
from airproof.v5_transport import generate_transport_trace, simulate_transport


def record(user=0, epoch=0, group=0, suffix="", value=10.):
    return Observation(user, epoch, group, group, value, 1., 1., 512,
                       f"{user}:{epoch}:{group}:{suffix}", 999, 888)


def test_registered_method_count_is_32_unique_evaluations_per_world():
    selected = EstimatorConfig()
    cells = ("anchor_clean", "outage_clean", "severe_clean", "severe_drift", "severe_hotspot")
    specs = {cell: method_specs(selected, cell) for cell in cells}
    assert sum(map(len, specs.values())) == 32
    for cell, methods in specs.items():
        assert len({m["method"] for m in methods}) == len(methods)
        assert len(methods) == (10 if cell in ("severe_clean", "severe_hotspot") else 4)
        assert {m["method"] for m in methods[:4]} == {"AP", "SQ", "PUBLIC", "HUBER"}
        sq = next(m["estimator"] for m in methods if m["method"] == "SQ")
        assert not sq.input_clip and not sq.output_cap
        assert sq.objective == selected.objective
        assert sq.regularization_multiplier == selected.regularization_multiplier
    assert len(method_specs(selected, "severe_clean", factorial=False)) == 4
    development = development_specs()
    assert len(development) == 19  # twelve candidates, six matched SQ controls, public
    assert len({m["method"] for m in development}) == 19


def test_reservation_deadline_canonical_aggregation_and_replay_deduplication():
    high = record(group=2)
    low_a, low_b = record(group=1, suffix="a"), record(group=1, suffix="b", value=12.)
    late = record(user=1)
    future = record(user=2, epoch=1)
    items = [high, low_a, low_a, low_b, late, future]
    snapshot = [asdict(r) for r in items]
    result = reserve_contributions(items, {(0, 0): 24, (1, 0): 25, (2, 1): 0}, steps=2)
    assert result == {0: {0: [low_a, low_b]}}
    assert [asdict(r) for r in items] == snapshot
    assert not reserve_contributions(items, {}, steps=2)
    with pytest.raises(ValueError, match="horizon"):
        reserve_contributions([record(epoch=2)], {(0, 2): 2}, steps=2)


def test_release_fixed_schedule_empty_cohorts_matched_noise_and_public_support():
    steps, groups, burn = 48, 4, 8
    public = np.full((steps, groups), 12.)
    truth = public + np.arange(groups)[None, :] + 1.
    world = SimpleNamespace(seed=123, truth=truth, cell_groups=np.arange(groups), observations=())
    cfg = {"world": {"groups": groups, "burn_in_steps": burn},
           "privacy": {"k_min": 20, "epsilon_mean": 8/28, "epsilon_count": 0.,
                       "epsilon_user_max": 8., "private_eligibility": False,
                       "release_deadline_steps": 24, "clip": 50., "residual_clip": 10.}}
    summary, outputs = release_evidence(world, public, {}, cfg)
    schedule = FixedReleasePlan(steps, groups, 20, 8/28, 0, 8).scheduled_epochs
    assert summary["scheduled_epochs"] == list(schedule)
    assert len(schedule) == 28
    assert summary["epsilon_history"] == pytest.approx(8.)
    assert summary["privacy_budget_violations"] == 0
    assert summary["admitted_user_epochs"] == 0
    mask = np.zeros((steps, groups), bool)
    mask[list(schedule)] = True
    np.testing.assert_array_equal(outputs["raw_mask"], mask)
    np.testing.assert_array_equal(outputs["residual_mask"], mask)
    np.testing.assert_allclose(outputs["raw_query"][mask], 0.)
    np.testing.assert_allclose(outputs["residual_query"][mask], 12.)
    raw_noise = outputs["raw"][mask] - outputs["raw_query"][mask]
    residual_noise = outputs["residual"][mask] - outputs["residual_query"][mask]
    np.testing.assert_allclose(raw_noise / 50., residual_noise / 10.)
    scored = mask.copy()
    scored[:burn] = False
    expected_public = np.sqrt(np.mean((public[scored] - truth[scored])**2))
    for method in ("raw", "residual"):
        assert summary[method]["released_group_queries"] == int(scored.sum())
        assert summary[method]["scheduled_group_queries"] == int(scored.sum())
        assert summary[method]["same_support_public_rmse"] == pytest.approx(expected_public)
        assert np.isnan(outputs[method][~mask]).all()


def test_arrival_filter_uses_separate_metadata_and_lag_boundary():
    fresh, late, missing = record(), record(user=1), record(user=2)
    world = SimpleNamespace(observations=(fresh, late, missing))
    cfg = {"world": {"steps": 10, "burn_in_steps": 1, "groups": 1},
           "twin": {"fixed_lag": 6},
           "scheduler": {"budget_bytes_per_epoch": 1024, "target_contributors": 1}}
    selected, metrics, counts = select_arrived(world, {fresh.nullifier: 6, late.nullifier: 7}, cfg, fairness=True)
    assert selected == [fresh]
    assert counts[6, 0] == 1
    assert metrics["retrospective_ingress"] == 1
    assert fresh.direct_arrival == 999 and fresh.relay_arrival == 888
    with pytest.raises(ValueError, match="future-information"):
        select_arrived(world, {fresh.nullifier: -1}, cfg, fairness=True)


@pytest.mark.parametrize("bad", [replace(record(), source_class="reference"),
                                  replace(record(), value=float("nan"))])
def test_reservation_rejects_invalid_citizen_domain(bad):
    with pytest.raises(ValueError):
        reserve_contributions([bad], {(0, 0): 0}, steps=1)


def test_lag_drain_updates_reconstruction_without_changing_live_past():
    steps, lag = 8, 3
    last = record(epoch=steps - 1, value=14.)
    world = SimpleNamespace(seed=193, observations=(last,), reference_observations=(),
                            truth=np.full((steps, 4), 12.), cell_groups=np.zeros(4, dtype=int))
    public = np.full((steps, 4), 10.)
    cfg = {"world": {"steps": steps, "burn_in_steps": 1, "groups": 1, "grid_side": 2},
           "twin": {"fixed_lag": lag},
           "scheduler": {"budget_bytes_per_epoch": 1024, "target_contributors": 1}}
    arrived = {last.nullifier: steps - 1 + lag}
    spec = method_specs(EstimatorConfig(lag=lag), "anchor_clean")[0]
    traces = [{"airproof_deadline": SimpleNamespace(raw_arrivals=arrivals, metrics={})}
              for arrivals in ({}, arrived)]
    outputs = [evaluate_prepared(world, public, None, cfg, transport, spec, strata=False)[1]
               for transport in traces]
    assert outputs[1]["counts"].shape == (steps + lag, 1)
    assert outputs[1]["counts"][-1, 0] == 1
    assert outputs[1]["live"].shape == outputs[1]["reconstructed"].shape == (steps, 4)
    np.testing.assert_array_equal(outputs[0]["live"], outputs[1]["live"])
    assert outputs[1]["reconstructed"][-1, 0] > outputs[0]["reconstructed"][-1, 0]
    assert last.epoch == steps - 1 and last.direct_arrival == 999 and last.relay_arrival == 888


def test_small_48_epoch_32_node_integrated_numerical_path():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml")
    cfg = with_overrides(cfg, {"world.agents": 32, "world.grid_side": 4,
        "world.steps": 48, "world.burn_in_steps": 8, "world.reference_station_count": 4,
        "twin.reference_calibration_epochs": 8, "network.availability": .75,
        "network.outage_median_hours": 2, "network.outage_p95_hours": 6,
        "scheduler.target_contributors": 2, "attack.kind": "clean", "attack.fraction": 0.})
    world = generate_world(cfg, 57931)
    original = tuple(asdict(r) for r in world.observations)
    public, operators, _ = _v4_public_components(world, cfg)
    saved_public = public.copy()
    trace = generate_transport_trace(cfg, 57931)
    transport = {policy: simulate_transport(trace, world.observations, cfg, policy)
                 for policy in ("direct", "airproof_deadline")}
    specs = method_specs(EstimatorConfig(), "anchor_clean")
    for name in ("AP", "PUBLIC"):
        spec = next(s for s in specs if s["method"] == name)
        row, arrays = evaluate_prepared(world, public, operators, cfg, transport, spec)
        assert row["method"] == name
        assert arrays["live"].shape == arrays["reconstructed"].shape == (48, 16)
        assert np.isfinite(arrays["live"]).all()
        assert len(row["strata"]) == 39
        assert row["metrics"]["feasible_floor_violations"] == 0
        assert row["transport"]["max_contact_direction_bytes"] <= 1024
        if name == "PUBLIC":
            np.testing.assert_array_equal(arrays["live"], saved_public)
    summary, outputs = release_evidence(world, public, transport["airproof_deadline"].release_arrivals, cfg)
    assert summary["privacy_budget_violations"] == 0
    assert outputs["residual"].shape == (48, 4)
    assert tuple(asdict(r) for r in world.observations) == original
    np.testing.assert_array_equal(public, saved_public)

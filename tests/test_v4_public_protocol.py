from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from airproof.config import load_config, with_overrides
from airproof.continual_release import (
    FixedReleasePlan, bounded_epoch_query, independent_contributions, release_epoch,
)
from airproof.field import grid_laplacian
from airproof.meteorology import (
    city_graph, grid_coordinates, synthetic_public_meteorology, transition_series, wind_transition,
)
from airproof.records import Observation
from airproof.reference import calibrated_public_reference
from airproof.twin import FixedLagTwin, robust_trajectory_update


def citizen(user=1, epoch=0, group=0, value=12, nullifier=None):
    return Observation(user, epoch, group, group, value, 1, 1, 512,
                       nullifier or f"{user}-{epoch}-{group}", epoch, epoch)


def test_weather_transition_is_directed_constant_preserving_and_nonexpansive():
    xy = grid_coordinates(3)
    adjacency = -grid_laplacian(3)
    adjacency.setdiag(0)
    east = wind_transition(adjacency, xy, wind_u=5, wind_v=0, transport=.4)
    west = wind_transition(adjacency, xy, wind_u=-5, wind_v=0, transport=.4)
    np.testing.assert_allclose(east @ np.ones(9), 1, atol=1e-14)
    assert east[4, 3] > east[4, 5]
    assert west[4, 5] > west[4, 3]
    assert np.min(east.data) >= 0
    assert np.linalg.norm(east.toarray() - np.eye(9)) > .1
    x = np.random.default_rng(9).normal(size=9)
    assert np.max(np.abs(east @ x)) <= np.max(np.abs(x)) + 1e-12
    for kwargs in ({"transport": -1}, {"wind_u": np.nan}, {"humidity": 101}):
        parameters = {"wind_u": 1, "wind_v": 0, **kwargs}
        with pytest.raises(ValueError):
            wind_transition(adjacency, xy, **parameters)


def test_city_graph_is_symmetric_and_handles_small_city():
    xy = np.array([[0., 0], [0, 1], [2, 1], [3, 3]])
    adjacency = city_graph(xy, neighbors=2)
    np.testing.assert_array_equal(adjacency.toarray(), adjacency.T.toarray())
    assert np.all(adjacency.diagonal() == 0)
    assert np.all(adjacency.data > 0)


def test_nonidentity_trajectory_matches_independent_dense_normal_equations():
    cells, window = 4, 3
    operators = tuple(sparse.csr_matrix(np.roll(np.eye(cells), t + 1, axis=1))
                      for t in range(window))
    boundary = np.array([10., 12, 15, 20])
    observations = [[replace(citizen(value=13 + t), cell=t)] for t in range(window)]
    laplacian = grid_laplacian(2)
    temporal, spatial = .7, .2
    estimate, diag = robust_trajectory_update(
        boundary, np.tile(boundary, (window, 1)), observations, laplacian,
        huber=False, delta=4, lambda_temporal=temporal, lambda_spatial=spatial,
        max_irls=4, tolerance=1e-12, transitions=operators,
    )
    # Build a least-squares design independently; avoid copying the block assembly.
    design = np.zeros((window * cells, window * cells))
    targets = np.zeros(window * cells)
    for t in range(window):
        block = slice(t * cells, (t + 1) * cells)
        design[block, block] = np.eye(cells)
        if t:
            design[block, slice((t - 1) * cells, t * cells)] = -operators[t].toarray()
        else:
            targets[block] = operators[t] @ boundary
    normal = temporal * design.T @ design + np.kron(np.eye(window), spatial * laplacian.toarray())
    rhs = temporal * design.T @ targets
    for t, items in enumerate(observations):
        for item in items:
            index = t * cells + item.cell
            normal[index, index] += 1
            rhs[index] += item.value
    np.testing.assert_allclose(estimate.ravel(), np.linalg.solve(normal, rhs), atol=1e-8)
    assert diag.converged


def test_weather_fixed_lag_uses_correct_absolute_epoch_and_keeps_closed_prefix():
    transitions = tuple(sparse.csr_matrix(np.roll(np.eye(4), t % 4, axis=1)) for t in range(6))
    twin = FixedLagTwin(2, 6, 2, False, 4, 1, 0, 4, 1e-10,
                        np.array([10., 15, 20, 25]), transitions=transitions)
    expected = twin.initial_state.copy()
    for epoch in range(5):
        expected = transitions[epoch] @ expected
        twin.ingest_at(epoch, [])
        np.testing.assert_allclose(twin.states[epoch], expected, atol=1e-7)
    closed = twin.states[:3].copy()
    twin.ingest_at(5, [replace(citizen(epoch=4, value=80), cell=1)])
    np.testing.assert_array_equal(twin.states[:3], closed)
    assert twin.states[4, 1] > expected[1]


def test_calibrated_public_reference_does_not_revise_prefix_or_look_ahead():
    side, steps, prefix = 3, 10, 4
    items = [replace(citizen(user=-cell - 1, epoch=t, value=10 + cell + t),
                     cell=cell, source_class="regulatory")
             for t in range(steps) for cell in (0, 2, 6, 8)]
    weather = synthetic_public_meteorology(steps, np.random.default_rng(10))
    adjacency = -grid_laplacian(side)
    adjacency.setdiag(0)
    operators = transition_series(adjacency, grid_coordinates(side), weather)
    args = dict(side=side, steps=steps, calibration_epochs=prefix, transitions=operators)
    original = calibrated_public_reference(items, **args)
    changed = calibrated_public_reference([
        replace(item, value=item.value + 100) if item.epoch >= 7 else item for item in items
    ], **args)
    np.testing.assert_array_equal(original.fields[:7], changed.fields[:7])
    assert original.diagnostics["selected"] == changed.diagnostics["selected"]
    assert original.diagnostics["first_selected_model_epoch"] == prefix
    # Future observations inside a training prefix cannot alter its earlier outputs.
    changed_training = calibrated_public_reference([
        replace(item, value=item.value + 50) if item.epoch == 3 else item for item in items
    ], **args)
    np.testing.assert_array_equal(original.fields[:3], changed_training.fields[:3])
    with pytest.raises(ValueError, match="regulatory"):
        calibrated_public_reference(items + [citizen()], **args)


def test_release_admission_is_user_local_replay_safe_and_independent_of_other_users():
    items = [citizen(1), citizen(1), citizen(1, group=2), citizen(2, value=18),
             replace(citizen(3), relay_arrival=50)]
    args = dict(steps=2, groups=4, relay=True, deadline_steps=24)
    mapped = independent_contributions(items, **args)
    assert len(mapped[0][1]) == 1
    assert mapped[0][1][0].group == 0
    assert 3 not in mapped[0]
    removed = independent_contributions([item for item in items if item.user_id != 1], **args)
    assert removed[0][2] == mapped[0][2]


@pytest.mark.parametrize("residual", [False, True])
def test_full_vector_sensitivity_covers_user_replacement_and_region_change(residual):
    rng = np.random.default_rng(19)
    baselines = np.array([10., 15, 20])
    clip, k = 5., 4
    for size in (0, 1, 3, 4, 5, 20):
        for _ in range(30):
            original = {i: [citizen(i, group=int(rng.integers(3)), value=rng.normal(15, 20))]
                        for i in range(size)}
            changed = {**original, 100: [citizen(100, group=0, value=-1e6)]}
            replacement = {**original, 100: [citizen(100, group=2, value=1e6)]}
            args = dict(groups=3, baselines=baselines, clip=clip, k_min=k, residual=residual)
            empty_query, _ = bounded_epoch_query(original, **args)
            before, before_count = bounded_epoch_query(changed, **args)
            after, after_count = bounded_epoch_query(replacement, **args)
            assert np.abs(empty_query - before).sum() <= 2 * clip / k + 1e-12
            assert np.abs(after - before).sum() <= 4 * clip / k + 1e-12
            assert np.abs(after_count - before_count).sum() <= 2


def test_schedule_and_charges_cannot_depend_on_suppression_or_private_history():
    plan = FixedReleasePlan(10, 3, 20, .5, .25, 3, private_eligibility=True)
    assert plan.scheduled_epochs == (0, 2, 5, 7)
    assert plan.composed_epsilon == 3
    for epoch in range(plan.steps):
        releases = release_epoch({}, epoch=epoch, plan=plan, public_baselines=np.ones(3) * 12,
                                 clip=10, residual=True, rng=np.random.default_rng(epoch))
        assert len(releases) == 3
        assert all("distinct_users" not in item.protected_payload() for item in releases)
    assert plan.composed_epsilon == 3
    assert FixedReleasePlan(10, 3, 20, 1, 0, 0).scheduled_epochs == ()


def test_default_fixed_schedule_emits_empty_cohorts_without_private_count_gate():
    plan = FixedReleasePlan(2, 3, 20, 1, 0, 2)
    releases = release_epoch({}, epoch=0, plan=plan, public_baselines=np.ones(3) * 12,
                             clip=10, residual=True, rng=np.random.default_rng(3))
    assert all(item.released and item.value is not None for item in releases)
    assert all(item.sensitivity == 2 for item in releases)


def test_final_v4_end_to_end_and_replay_use_identical_public_baseline_and_fixed_release():
    from airproof.experiment import run_method, run_privacy_replay
    from airproof.simulator import generate_world

    cfg = with_overrides(load_config(Path(__file__).resolve().parents[1]
                                    / "configs/v4/public_reference_validation.yaml"), {
        "world.agents": 30, "world.grid_side": 4, "world.steps": 10, "world.burn_in_steps": 4,
        "world.reference_station_count": 4, "twin.reference_calibration_epochs": 4,
        "privacy.k_min": 2, "twin.max_irls": 4, "twin.fixed_lag": 2,
    })
    world = generate_world(cfg, 7200)
    direct = run_method(world, cfg, "airproof_v4")["metrics"]
    replay = run_privacy_replay(world, cfg, "airproof_v4")["metrics"]
    for metric in ("release_count", "release_rmse", "max_composed_user_epsilon",
                   "privacy_vector_sensitivity", "public_backbone_calibration"):
        assert direct[metric] == replay[metric]
    assert direct["privacy_raw_scheduler_independent"]
    assert direct["privacy_budget_violation_count"] == 0
    assert direct["release_baseline_citizen_independent"]
    assert direct["predictive_adaptive_delta"] is False
    assert direct["solver_failure_rate"] == 0
    assert direct["live_rmse"] >= 0
    assert direct["prediction_clock"] == "bounded-fixed-lag-reconstruction-at-horizon-end"
    assert direct["live_prediction_clock"] == "frozen-at-own-epoch-before-future-arrivals"
    assert direct["intersectional_audit"]["definition"]["mask_sha256"]


def test_arrival_allocation_never_uses_future_delivery_and_counts_each_user_once():
    from airproof.scheduler import select_evidence

    one = citizen(1, epoch=1, value=12)
    newer_same_user = citizen(1, epoch=2, value=13)
    another = citizen(2, epoch=2, value=14)
    args = dict(budget_bytes=1024, reserve_fraction=.25, targets={0: 2},
                constraint_mode="hard_if_feasible", allocation_epoch=3)
    selected = select_evidence([one, newer_same_user, another], **args)
    assert selected.counts[0] == 2
    assert newer_same_user in selected.selected and one not in selected.selected
    # Candidate set is supplied by the arrival event loop; irrelevant future
    # delivery fields cannot change the utility once a record has arrived.
    changed = select_evidence([replace(item, relay_arrival=999, direct_arrival=998)
                                for item in (one, newer_same_user, another)], **args)
    assert [(v.user_id, v.epoch) for v in selected.selected] == [(v.user_id, v.epoch) for v in changed.selected]
    with pytest.raises(ValueError, match="future-acquired"):
        select_evidence([citizen(epoch=4)], **args)


def test_v4_selection_is_causal_at_the_arrival_boundary():
    from airproof.experiment import _select, METHODS
    from airproof.simulator import generate_world

    cfg = with_overrides(load_config(Path(__file__).resolve().parents[1]
                                    / "configs/v4/public_reference_validation.yaml"), {
        "world.agents": 5, "world.grid_side": 4, "world.steps": 8,
        "world.burn_in_steps": 2, "twin.fixed_lag": 2,
    })
    world = generate_world(cfg, 7201)
    events = (replace(citizen(1, epoch=0), relay_arrival=0),
              replace(citizen(2, epoch=0), relay_arrival=4),  # retrospective, not current floor
              replace(citizen(3, epoch=2), relay_arrival=3),
              replace(citizen(4, epoch=3), relay_arrival=7))
    world = replace(world, observations=events)
    selected, diagnostics = _select(world, cfg, METHODS["airproof_v4"])
    assert {item.user_id for item in selected} == {1, 3}
    assert diagnostics["scheduler_retrospective_ingress_records"] == 2
    # Mutating an event that has not arrived cannot displace a past accepted record.
    changed, _ = _select(replace(world, observations=events[:3] + (
        replace(events[3], value=1e6, quality=0.),)), cfg, METHODS["airproof_v4"])
    assert selected == changed

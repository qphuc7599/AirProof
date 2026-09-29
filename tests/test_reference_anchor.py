from dataclasses import replace

import numpy as np
import pytest

from airproof.records import Observation
from airproof.reference import public_reference_fields
from airproof.twin import FixedLagTwin


def reference(epoch=0, cell=0, value=10.0):
    return Observation(
        -cell - 1, epoch, cell, 0, value, 0.5, 1.0, 0,
        f"ref-{cell}-{epoch}", epoch, epoch, source_class="regulatory",
    )


def test_reference_builder_is_causal_and_carries_only_public_history():
    prefix = [reference(1, cell, 15.0) for cell in (0, 3, 12, 15)]
    future = [reference(3, cell, 90.0) for cell in (0, 3, 12, 15)]
    first = public_reference_fields(prefix, side=4, steps=4)
    second = public_reference_fields(prefix + future, side=4, steps=4)
    np.testing.assert_array_equal(first[:3], second[:3])
    np.testing.assert_allclose(first[0], 12.0)
    np.testing.assert_allclose(first[1:3], 15.0)
    np.testing.assert_allclose(second[3], 90.0)


def test_reference_builder_rejects_nonpublic_and_future_arrival_inputs():
    with pytest.raises(ValueError, match="regulatory"):
        public_reference_fields([replace(reference(), source_class="citizen")], side=2, steps=2)
    with pytest.raises(ValueError, match="on-time"):
        public_reference_fields([replace(reference(), direct_arrival=1)], side=2, steps=2)


def test_reference_builder_handles_sparse_collinear_and_empty_references():
    np.testing.assert_allclose(public_reference_fields([], side=2, steps=2), 12.0)
    records = [reference(0, cell, 10.0) for cell in (0, 1, 2)]
    np.testing.assert_allclose(public_reference_fields(records, side=4, steps=2), 10.0)


def anchored_twin(steps=40):
    return FixedLagTwin(
        2, steps, 2, True, 1.345, 0.5, 0.2, 4, 1e-9, np.full(4, 10.0),
        predictive_residual=True, predictive_adaptive_delta=True, correction_clip=2.0,
    )


def test_anchor_is_independent_of_entire_citizen_history_and_uniformly_bounded():
    clean, poisoned = anchored_twin(), anchored_twin()
    for epoch in range(40):
        anchor = np.full((min(epoch, 2) + 1, 4), 10.0)
        base = replace(reference(epoch, 0), user_id=1, source_class="citizen", sigma=1.0)
        clean.ingest_at(epoch, [base], external_predictor=anchor)
        poisoned.ingest_at(epoch, [replace(base, value=1e6)], external_predictor=anchor)
        np.testing.assert_array_equal(clean.release_baselines, poisoned.release_baselines)
        assert np.max(np.abs(poisoned.states[:epoch + 1] - 10.0)) <= 2.0 + 1e-8
        assert np.max(np.abs(poisoned.states - clean.states)) <= 4.0 + 1e-8


def test_late_update_preserves_closed_prefix_and_public_release_history():
    twin = anchored_twin(6)
    for epoch in range(5):
        twin.ingest_at(epoch, [], external_predictor=np.full((min(epoch, 2) + 1, 4), 10.0))
    before, public_before = twin.states.copy(), twin.release_baselines.copy()
    late = replace(reference(4, 0, 50.0), user_id=1, source_class="citizen")
    twin.ingest_at(5, [late], external_predictor=np.full((3, 4), 10.0))
    np.testing.assert_array_equal(twin.states[:3], before[:3])
    np.testing.assert_array_equal(twin.release_baselines[:5], public_before[:5])
    assert twin.states[4, 0] > 10.0


def test_invalid_external_anchor_does_not_mutate_state():
    twin = anchored_twin()
    before = twin.states.copy()
    for invalid in (np.ones((4,)), np.full((1, 4), np.nan), -np.ones((1, 4))):
        with pytest.raises(ValueError, match="external predictor"):
            twin.ingest_at(0, [], external_predictor=invalid)
        np.testing.assert_array_equal(twin.states, before)


def test_single_track_anchor_does_not_activate_an_empirical_tail_gate():
    twin = anchored_twin(6)
    twin.predictive_adaptive_delta = False
    for epoch in range(6):
        observation = replace(reference(epoch, 0, 1e6), user_id=1, source_class="citizen")
        twin.ingest_at(epoch, [observation],
                       external_predictor=np.full((min(epoch, 2) + 1, 4), 10.0))
    np.testing.assert_array_equal(twin.predictive_activation_by_epoch, 0.0)
    assert np.max(np.abs(twin.states - 10.0)) <= twin.correction_clip + 1e-8


def test_public_anchor_release_matches_its_privacy_replay():
    from pathlib import Path

    from airproof.config import load_config, with_overrides
    from airproof.experiment import run_method, run_privacy_replay
    from airproof.simulator import generate_world

    config = with_overrides(load_config(
        Path(__file__).resolve().parents[1] / "configs/v3/diagnostic.yaml"
    ), {"world.agents": 40, "world.grid_side": 4, "world.steps": 8,
        "world.burn_in_steps": 2, "privacy.release_mode": "residual", "privacy.k_min": 2})
    world = generate_world(config, 7100)
    direct = run_method(world, config, "airproof_reference_anchored")["metrics"]
    replay = run_privacy_replay(world, config, "airproof_reference_anchored")["metrics"]
    assert direct["release_baseline_citizen_independent"]
    assert replay["release_baseline_citizen_independent"]
    for metric in ("release_count", "suppressed_release_count", "release_rmse",
                   "max_composed_user_epsilon"):
        assert direct[metric] == replay[metric]

from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import yaml
from airproof.config import with_overrides
from airproof.simulator import generate_world
from airproof.v5_stress import (STRESS_SEEDS, configuration_for_stress, prepare_stress_public,
                               stress_matrix, transform_world, validate_selected)


@pytest.fixture(scope="module")
def prepared():
    root = Path(__file__).resolve().parents[1]
    protocol = yaml.safe_load((root/"configs/v5/protocol.yaml").read_text())
    cfg = yaml.safe_load((root/protocol["historical_configuration"]).read_text())
    cfg["transport"] = protocol["transport"]
    cfg = with_overrides(cfg, {"world.agents": 12, "world.grid_side": 8, "world.steps": 48,
        "world.burn_in_steps": 8, "twin.reference_calibration_epochs": 8, "attack.start_epoch": 16})
    cfg = configuration_for_stress(cfg, stress_matrix()[0])
    return cfg, generate_world(cfg, 910009)


def test_frozen_oat_matrix_and_selection_gate():
    matrix = stress_matrix()
    assert len(matrix) == 19 and len({c.name for c in matrix}) == 19
    assert STRESS_SEEDS == (915004,915005,915006,915007)
    with pytest.raises(ValueError): validate_selected(None)


def test_transforms_preserve_original_and_reference_bias_not_truth(prepared):
    cfg, world = prepared
    before = repr(world.observations), repr(world.reference_observations), world.truth.copy()
    bias = next(c for c in stress_matrix() if c.reference_bias == 8)
    changed, _ = transform_world(world, cfg, bias)
    assert np.array_equal(changed.truth, world.truth)
    assert changed.observations == world.observations
    assert all(b.value == a.value+8 for a,b in zip(world.reference_observations, changed.reference_observations))
    assert before[0] == repr(world.observations) and before[1] == repr(world.reference_observations)
    assert np.array_equal(before[2], world.truth)


def test_event_changes_truth_citizens_not_regulatory(prepared):
    cfg, world = prepared
    case = stress_matrix()[-1]
    changed, mask = transform_world(world, cfg, case)
    assert mask.any() and changed.reference_observations == world.reference_observations
    assert not mask[:, [r.cell for r in world.reference_observations]].any()
    np.testing.assert_allclose((changed.truth-world.truth)[mask], 8, rtol=0, atol=1e-12)
    for old,new in zip(world.observations, changed.observations):
        assert new.value == old.value + (8 if mask[old.epoch,old.cell] else 0)


def test_delay_preserves_acquisition_and_no_future_reference_leakage(prepared):
    cfg, world = prepared
    case = next(c for c in stress_matrix() if c.reference_delay == 6)
    changed, _ = transform_world(world, cfg, case)
    assert all(r.direct_arrival == r.epoch+6 for r in changed.reference_observations)
    first, _, diagnostic = prepare_stress_public(changed, cfg, case)
    future = replace(changed, reference_observations=tuple(replace(r, value=r.value+1000) if r.epoch >= 28 else r for r in changed.reference_observations))
    second, _, _ = prepare_stress_public(future, cfg, case)
    np.testing.assert_array_equal(first[:34], second[:34])
    assert diagnostic["first_available_selected_model_epoch"] == 14
    assert np.all(first[:6] == cfg["world"]["baseline"])
    no_truth = replace(changed, truth=changed.truth+999, observations=())
    third, _, _ = prepare_stress_public(no_truth, cfg, case)
    np.testing.assert_array_equal(first, third)


def test_bandwidth_transform_does_not_mutate_config(prepared):
    cfg, _ = prepared
    result = configuration_for_stress(cfg, next(c for c in stress_matrix() if c.bandwidth == 512))
    assert result["transport"]["capacity_bytes_per_direction"] == 512
    assert cfg["transport"]["capacity_bytes_per_direction"] == 1024

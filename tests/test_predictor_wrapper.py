from dataclasses import replace

import numpy as np

from airproof.predictor_wrapper import synthetic_citizen_replay, wrap_public_predictions


def test_wrapper_bounds_correction_and_never_revises_live_past():
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .3]])
    truth = np.tile(np.arange(5.) + 20, (18, 1))
    prediction = truth + 1
    observations = synthetic_citizen_replay(truth, seed=14, reports_per_station=2)
    a, diagnostics = wrap_public_predictions(prediction, xy, observations, fixed_lag=3)
    changed = [replace(item, value=item.value + 10000) if item.epoch >= 12 else item
               for item in observations]
    b, _ = wrap_public_predictions(prediction, xy, changed, fixed_lag=3)
    np.testing.assert_array_equal(a[:12], b[:12])
    assert np.max(np.abs(b - prediction)) <= 8. + 1e-9
    assert diagnostics["solver_failure_rate"] == 0
    assert diagnostics["padded_isolated_vertices"] == 4
    assert diagnostics["public_predictor_updated_from_citizen"] is False
    replay, _ = wrap_public_predictions(prediction, xy, observations * 2, fixed_lag=3)
    np.testing.assert_array_equal(a, replay)


def test_replay_attack_does_not_change_presence_delays_or_pre_attack_values():
    truth = np.full((20, 5), 20.)
    clean = synthetic_citizen_replay(truth, seed=5)
    attack = synthetic_citizen_replay(truth, seed=5, kind="adversarial_drift")
    assert len(clean) == len(attack)
    assert any(item.corrupted for item in attack)
    for a, b in zip(clean, attack, strict=True):
        assert (a.user_id, a.epoch, a.relay_arrival, a.nullifier) == (b.user_id, b.epoch, b.relay_arrival, b.nullifier)
        if not b.corrupted:
            assert a.value == b.value

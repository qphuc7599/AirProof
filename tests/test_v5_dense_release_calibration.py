from types import SimpleNamespace
import numpy as np
from airproof.records import Observation
from airproof.continual_release import bounded_epoch_query
from airproof.v5_experiment import release_evidence, reserve_contributions


def test_dense_calibration_preserves_default_arrays_and_actual_hourly_query():
    steps = 60
    public = np.full((steps, 2), 20.)
    observations = [Observation(0, epoch, 0, 0, 100., 1., 1., 512, str(epoch), epoch, epoch)
                    for epoch in (0, 1, 23, 24, 47, 59)]
    # One late arrival must be excluded by exactly the existing reservation rule.
    arrivals = {(0, r.epoch): r.epoch for r in observations}
    arrivals[0, 23] = 48
    world = SimpleNamespace(truth=public+1, cell_groups=np.array([0, 1]),
                            observations=observations, seed=912000)
    cfg = {"world": {"groups": 2, "burn_in_steps": 2},
           "privacy": {"k_min": 20, "epsilon_mean": 8/28, "epsilon_count": 0,
                       "epsilon_user_max": 8, "clip": 50, "residual_clip": 10,
                       "release_deadline_steps": 24}}
    default_summary, default_arrays = release_evidence(world, public, arrivals, cfg)
    explicit_summary, explicit_arrays = release_evidence(world, public, arrivals, cfg, dense_calibration=False)
    dense_summary, dense_arrays = release_evidence(world, public, arrivals, cfg, dense_calibration=True)
    assert default_summary == explicit_summary
    assert set(default_arrays) == {"raw", "raw_query", "raw_mask", "residual", "residual_query", "residual_mask", "baseline", "truth"}
    for key in default_arrays:
        np.testing.assert_array_equal(default_arrays[key], explicit_arrays[key])
        np.testing.assert_array_equal(default_arrays[key], dense_arrays[key])
    assert all(dense_summary[key] == value for key, value in default_summary.items())
    assert dense_summary["dense_query_calibration_only"]
    dense = dense_arrays["residual_query_dense"]
    assert dense.shape == (steps, 2) and np.isfinite(dense).all()
    contributions = reserve_contributions(observations, arrivals, steps=steps, deadline=24)
    for epoch in range(steps):
        expected, _ = bounded_epoch_query(contributions.get(epoch, {}), groups=2,
                                          baselines=public[epoch], clip=10, k_min=20, residual=True)
        np.testing.assert_array_equal(dense[epoch], expected)
    assert dense[1, 0] == 20.5
    assert dense[23, 0] == 20.  # Deadline exclusion, not an interpolated neighbor.
    np.testing.assert_array_equal(dense[:, 1], public[:, 1])  # Empty groups are finite.

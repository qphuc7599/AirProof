from dataclasses import replace

import numpy as np
import pytest
import yaml

from airproof.continual_release import FixedReleasePlan, independent_contributions, release_epoch
from airproof.privacy_grid import (evaluate_setting, grid_settings, plan_for_setting,
                                  prepare_statistics, replay_release)
from airproof.records import Observation


def inputs():
    records = [Observation(user, epoch, group, group, float(value), 2., 1., 512,
        f"{user}-{epoch}-{group}-{value}", epoch, epoch + 1)
        for user, epoch, group, value in [(1, 0, 0, 12), (1, 0, 0, 16), (1, 0, 1, 10000),
            (2, 0, 1, 10), (3, 0, 2, 100), (4, 1, 0, -1000), (5, 1, 2, 20),
            (1, 3, 1, 15), (2, 4, 1, 16), (3, 5, 2, 17), (4, 6, 0, 18)]]
    records += [records[0], replace(records[0], user_id=99, relay_arrival=100)]
    baseline = np.arange(21).reshape(7, 3) / 2 + 10
    return records, baseline


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_cached_statistics_match_full_production_mechanism(private, residual):
    records, baseline = inputs()
    stats = prepare_statistics(records, public_baselines=baseline, groups=3, deadline_steps=2)
    contributions = independent_contributions(records, steps=7, groups=3, relay=True, deadline_steps=2)
    plan = FixedReleasePlan(7, 3, 2, .5, .25, 3., private)
    values, emitted = replay_release(stats, plan, residual=residual, rng=np.random.default_rng(50))
    rng = np.random.default_rng(50)
    for epoch in range(7):
        expected = release_epoch(contributions.get(epoch, {}), epoch=epoch, plan=plan,
            public_baselines=baseline[epoch], clip=10 if residual else 50,
            residual=residual, rng=rng)
        np.testing.assert_array_equal(emitted[epoch], [item.released for item in expected])
        np.testing.assert_allclose(values[epoch], [np.nan if item.value is None else item.value for item in expected],
                                   rtol=1e-13, atol=1e-13, equal_nan=True)


def test_all_216_settings_have_exact_registered_schedule_and_valid_accounting():
    from pathlib import Path
    protocol = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/v4/privacy_interactions.yaml").read_text())
    settings = grid_settings(protocol)
    assert len(settings) == 216
    for setting in settings:
        plan = plan_for_setting(setting, steps=672, groups=4, false_release_probability=.01)
        assert len(plan.scheduled_epochs) == setting["scheduled_epochs"]
        assert plan.composed_epsilon <= setting["epsilon_user_max"] + 1e-10
        assert len(set(plan.scheduled_epochs)) == len(plan.scheduled_epochs)


def test_raw_and_residual_share_count_gate_support_and_empty_results_are_not_zero():
    records, baseline = inputs()
    stats = prepare_statistics(records, public_baselines=baseline, groups=3, deadline_steps=2)
    setting = {"epsilon_user_max": 1., "scheduled_epochs": 7, "k_min": 1000,
               "private_eligibility": True, "count_budget_fraction": .2}
    results = [evaluate_setting(stats, baseline, {**setting, "release_mode": mode}, seed=71,
                                burn_in=1, deadline_steps=2) for mode in ("raw", "residual")]
    assert results[0]["release_count"] == results[1]["release_count"]
    assert results[0]["release_rmse"] is None and results[1]["release_rmse"] is None
    assert results[0]["privacy_budget_violation_count"] == 0
    assert results[0]["release_clock"] == "acquisition_epoch_plus_fixed_collection_deadline"

from types import SimpleNamespace

import numpy as np

from airproof.records import Observation
from airproof.v6_fairness_validation import allocate_shared_arrivals, outcome_strata_39


def record(user, epoch, cell, group, size=1):
    return Observation(user, epoch, cell, group, 1., 1., 1., size,
                       f"{user}-{epoch}-{group}", None, None)


def config():
    return {"world": {"steps": 4, "burn_in_steps": 0, "groups": 2},
            "twin": {"fixed_lag": 1},
            "scheduler": {"budget_bytes_per_epoch": 2, "target_contributors": 1}}


def test_three_allocators_share_arrivals_and_respect_budget():
    observations = [record(0, 0, 0, 0), record(1, 0, 1, 1),
                    record(2, 1, 2, 0), record(3, 1, 3, 1)]
    world = SimpleNamespace(observations=observations)
    arrivals = {item.nullifier: item.epoch for item in observations}
    for mode in ("v6_minimax", "utility_only", "v4_fallback"):
        result = allocate_shared_arrivals(world, arrivals, config(), mode, window=2)
        assert result.metrics["byte_violations"] == 0
        assert result.metrics["max_spent_bytes"] <= 2
        assert result.metrics["selected_records"] == 4
        assert result.counts.shape == (5, 2)


def test_opportunity_aware_starvation_and_distinct_debt_are_separate():
    observations = [record(0, epoch, 0, 0, 2) for epoch in range(4)]
    observations += [record(2, epoch, 2, 0, 3) for epoch in range(4)]
    observations += [record(1, epoch, 1, 1, 1) for epoch in range(4)]
    world = SimpleNamespace(observations=observations)
    arrivals = {item.nullifier: item.epoch for item in observations}
    result = allocate_shared_arrivals(world, arrivals, config(), "utility_only", window=2)
    assert result.metrics["worst_available_but_unserved_streak"] >= 1
    assert result.metrics["distinct_user_debt_by_group"] == {0: 1, 1: 0}


def test_all_39_strata_are_retained_including_empty_masks():
    side = 4; cells = side*side; steps = 5; burn = 1
    observations = [record(cell, 0, cell, cell % 4) for cell in range(cells)]
    world = SimpleNamespace(observations=observations,
        truth=np.ones((steps, cells)), cell_groups=np.arange(cells) % 4)
    public = np.ones_like(world.truth); predictions = np.ones((steps-burn, cells))
    strata = outcome_strata_39(world, public, {x.nullifier: 0 for x in observations},
                               predictions, predictions, observations, burn)
    assert len(strata) == 39
    assert any(not item["supported"] for item in strata.values())
    assert all("selected_distinct_contributors" in item for item in strata.values())

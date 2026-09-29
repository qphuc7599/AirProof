"""Constructive counterexamples: unsafe swaps can break cross-layer invariants."""
from dataclasses import replace

import numpy as np

from airproof.field import grid_laplacian
from airproof.records import Observation
from airproof.scheduler import select_evidence
from airproof.twin import FixedLagTwin, robust_state_update


def record(user=0, group=0, epoch=0, value=10.):
    return Observation(user, epoch, 0, group, value, 1., 1., 100,
                       f"{user}-{group}-{epoch}", epoch, epoch)


def test_assimilating_before_dedup_changes_numerical_influence():
    kwargs = dict(huber=False, delta=4, lambda_prior=1, lambda_spatial=0,
                  max_irls=4, tolerance=1e-12)
    correct, _ = robust_state_update(np.zeros(4), [record()], grid_laplacian(2), **kwargs)
    wrong, _ = robust_state_update(np.zeros(4), [record(), record()], grid_laplacian(2), **kwargs)
    assert np.isclose(correct[0], 5)
    assert np.isclose(wrong[0], 20 / 3)
    assert not np.array_equal(correct, wrong)


def test_unaccounted_private_baseline_can_exceed_the_claimed_privacy_bound():
    # Empty current cohort; only a past private reading changes the reused center.
    # At output y=0, the ratio of Laplace densities is exp(distance/scale).
    epsilon, clip, k_min = 1., 10., 1
    scale = 4 * clip / (k_min * epsilon)
    private_history_centers = (0., 1000.)
    log_density_ratio = abs(private_history_centers[1] - private_history_centers[0]) / scale
    assert log_density_ratio == 25
    assert log_density_ratio > epsilon
    # Merely saying 'computed before the current batch' cannot remove this loss.


def test_utility_before_floor_reservation_can_destroy_global_feasibility():
    low = [replace(record(user=i, group=0), sigma=10., quality=.1) for i in range(2)]
    high = [record(user=i, group=1) for i in range(2, 5)]
    candidates = low + high
    kwargs = dict(reserve_fraction=.25, targets={0: 2, 1: 1}, constraint_mode="hard_if_feasible")
    correct = select_evidence(candidates, budget_bytes=300, fairness=True, **kwargs)
    assert correct.globally_feasible and correct.constraint_satisfied
    utility_first = select_evidence(candidates, budget_bytes=200, fairness=False, **kwargs)
    assert len(utility_first.selected) == 2
    assert all(item.group == 1 for item in utility_first.selected)
    remaining_budget = 300 - utility_first.spent_bytes
    assert remaining_budget < sum(item.size_bytes for item in low)


def test_arrival_timestamp_relabeling_injects_old_evidence_into_current_state():
    def twin():
        return FixedLagTwin(2, 4, 1, False, 4, 1, 0, 4, 1e-10, np.full(4, 10.))

    correct, wrong = twin(), twin()
    for epoch in range(3):
        correct.ingest_at(epoch, [])
        wrong.ingest_at(epoch, [])
    late = replace(record(epoch=0, value=20), direct_arrival=3, relay_arrival=3)
    correct.ingest_at(3, [late])
    wrong.ingest_at(3, [replace(late, epoch=3)])
    assert len(correct.retrospective_records) == 1
    np.testing.assert_allclose(correct.states, 10)
    assert wrong.states[3, 0] > 10

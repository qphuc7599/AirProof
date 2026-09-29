from airproof.records import Observation
from airproof.scheduler import policy_weight_targets, population_targets, select_evidence


def record(user, group, size=100):
    return Observation(user, 0, group, group, 10.0, 1.0, 1.0, size, f"n-{user}-{group}", 0, 0)


def test_two_group_three_slot_recomputes_deficit_each_step():
    result = select_evidence(
        [record(0, 0), record(1, 0), record(2, 0), record(3, 1), record(4, 1), record(5, 1)],
        budget_bytes=300,
        reserve_fraction=1.0,
        targets={0: 10, 1: 10},
    )
    assert sum(result.counts.values()) == 3
    assert abs(result.counts[0] - result.counts[1]) <= 1


def test_user_group_epoch_contribution_is_bounded():
    duplicated = [record(1, 0), record(1, 0), record(2, 0)]
    result = select_evidence(duplicated, budget_bytes=1000, reserve_fraction=0, targets={0: 1})
    assert len(result.selected) == 2


def test_infeasibility_is_reported():
    result = select_evidence([record(1, 0)], budget_bytes=100, reserve_fraction=1, targets={0: 1, 1: 1})
    assert result.infeasible_groups == (1,)
    assert result.opportunity_shortfall_groups == (1,)


def test_population_and_policy_targets_have_distinct_semantics():
    assert population_targets({0: 100, 1: 200}, 0.01) == {0: 1, 1: 2}
    assert policy_weight_targets({0: 1.0, 1: 3.0}, 8) == {0: 2, 1: 6}


def test_hard_constraint_satisfies_every_globally_feasible_floor():
    candidates = [
        record(0, 0),
        record(1, 0),
        record(2, 1),
        record(3, 1),
        record(4, 0),
    ]
    result = select_evidence(
        candidates,
        budget_bytes=400,
        reserve_fraction=0.1,
        targets={0: 2, 1: 2},
        fairness_strength=1.0,
        constraint_mode="hard_if_feasible",
    )
    assert result.globally_feasible
    assert result.constraint_satisfied
    assert result.counts[0] >= 2
    assert result.counts[1] >= 2


def test_zero_fairness_strength_matches_unconstrained_selection():
    candidates = [record(0, 0), record(1, 0), record(2, 1), record(3, 1)]
    zero = select_evidence(
        candidates,
        budget_bytes=200,
        reserve_fraction=1.0,
        targets={0: 2, 1: 2},
        fairness_strength=0.0,
        constraint_mode="hard_if_feasible",
    )
    unconstrained = select_evidence(
        candidates,
        budget_bytes=200,
        reserve_fraction=1.0,
        targets={0: 2, 1: 2},
        fairness=False,
    )
    assert [item.nullifier for item in zero.selected] == [
        item.nullifier for item in unconstrained.selected
    ]

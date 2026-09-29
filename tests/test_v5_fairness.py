from dataclasses import replace
from fractions import Fraction
from itertools import product
import random

import pytest

from airproof.records import Observation
from airproof.v5_fairness import causal_representatives, select_minimax_evidence


def record(user, group, size=100, epoch=0, quality=1.):
    return Observation(user, epoch, group, group, 10., 1., quality, size,
                       f"{user}-{group}-{epoch}", None, None)


def oracle(pool, targets, budget):
    available = {g: min(q, sum(x.group == g for x in pool)) for g, q in targets.items()}
    best = Fraction(1)
    for mask in product((False, True), repeat=len(pool)):
        selected = [x for x, yes in zip(pool, mask) if yes]
        if sum(x.size_bytes for x in selected) <= budget:
            score = max((Fraction(max(0, available[g] - sum(x.group == g for x in selected)), q)
                         for g, q in targets.items()), default=Fraction(0))
            best = min(best, score)
    return best


def test_exhaustive_subset_oracle_random_heterogeneous_instances():
    rng = random.Random(56931)
    for _ in range(300):
        targets = {g: rng.randint(1, 5) for g in range(rng.randint(1, 4))}
        pool = [record(i, rng.choice(list(targets)), rng.randint(0, 9)) for i in range(rng.randint(0, 9))]
        budget = rng.randint(0, 30)
        result = select_minimax_evidence(pool, budget_bytes=budget, targets=targets, allocation_epoch=0)
        expected = oracle(pool, targets, budget)
        assert Fraction(result.optimum_numerator, result.optimum_denominator) == expected
        assert not result.violations
        assert result.spent_bytes <= budget
        for g in targets:
            assert result.deficits[g] == pytest.approx(result.unavoidable_deficits[g] + result.avoidable_deficits[g])


def test_floor_certificate_free_candidates_and_missing_groups():
    result = select_minimax_evidence([record(0, 0, 0), record(1, 0, 0)],
                                    budget_bytes=0, targets={0: 1, 1: 30}, allocation_epoch=0)
    assert len(result.selected) == 2
    assert result.opportunity_shortfall_groups == (1,)
    assert result.unavoidable_deficits[1] == 1
    assert result.avoidable_deficits[1] == 0
    assert result.optimum_numerator == 0
    assert not result.globally_feasible


def test_causal_representative_ties_are_permutation_invariant():
    old = record(1, 0, 2, epoch=0)
    new = record(1, 0, 9, epoch=1)
    tied = replace(new, nullifier="0", direct_arrival=10000)
    pool = [old, new, tied]
    assert causal_representatives(pool, 1) == causal_representatives(reversed(pool), 1) == (tied,)
    with pytest.raises(ValueError, match="future"):
        causal_representatives(pool, 0)


def test_no_targets_and_no_candidates_and_untargeted_fill():
    result = select_minimax_evidence([], budget_bytes=0, targets={}, allocation_epoch=0)
    assert result.globally_feasible and not result.violations
    result = select_minimax_evidence([record(0, 9, 3)], budget_bytes=3, targets={}, allocation_epoch=0)
    assert result.counts == {9: 1}


def test_feasible_targets_retained_after_utility_fill():
    pool = [record(i, i % 2, i + 1) for i in range(6)]
    result = select_minimax_evidence(pool, budget_bytes=10, targets={0: 2, 1: 2}, allocation_epoch=0)
    assert result.globally_feasible and result.constraint_satisfied
    assert result.certificate_bytes == 10
    assert result.certificate_counts == {0: 2, 1: 2}


def test_invalid_inputs():
    for budget, targets, pool in [(-1, {0: 1}, []), (1, {0: 0}, []), (1, {0: 1}, [record(0, 0, -1)])]:
        with pytest.raises(ValueError):
            select_minimax_evidence(pool, budget_bytes=budget, targets=targets, allocation_epoch=0)


def test_no_fairness_utility_per_byte_and_zero_cost():
    pool = [record(0, 0, 3), record(1, 1, 1), record(2, 1, 0)]
    result = select_minimax_evidence(pool, budget_bytes=1, targets={0: 1, 1: 1}, allocation_epoch=0, fairness=False)
    assert {x.user_id for x in result.selected} == {1, 2}
    assert not result.constraint_satisfied
    assert result.certificate_bytes == 0

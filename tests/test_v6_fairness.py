from airproof.records import Observation
from airproof.v6_fairness import CumulativeServiceState, select_cumulative_evidence
from dataclasses import replace
from fractions import Fraction
from scripts.verify_v6_fairness import audit, oracle
from airproof.v6_fairness import causal_representatives


def r(user, group=0):
    return Observation(user,0,user,group,1.,1.,1.,512,str(user),None,None)


def test_cumulative_preference_preserves_floor_and_budget():
    state=CumulativeServiceState()
    first=state.allocate([r(0),r(1)],budget_bytes=512,targets={0:1},allocation_epoch=0)
    second=state.allocate([r(0),r(1)],budget_bytes=512,targets={0:1},allocation_epoch=1)
    assert first.selected[0].user_id!=second.selected[0].user_id
    assert second.constraint_satisfied and not second.violations
    assert state.distinct_debt({0:3})=={0:1}
    assert state.epoch_debt=={0:0}


def test_nf_does_not_use_cumulative_preference():
    result=select_cumulative_evidence([r(0),r(1)],budget_bytes=512,targets={0:1},allocation_epoch=0,
        fairness=False,served_users=frozenset({(0,0)}))
    assert result.selected[0].user_id==0


def test_exhaustive_small_partition_oracle_and_decomposition():
    report = audit()
    assert report['violations'] == 0
    assert report['comparisons']['v6']['worse_than_oracle'] == 0
    assert report['zero_budget_instances'] > 0
    assert report['comparisons']['utility_only']['worse_than_oracle'] > 0
    assert report['comparisons']['v4_fallback']['worse_than_oracle'] > 0


def test_q30_feasible_floor_and_missing_opportunity_zero_budget():
    pool = [r(i, i//30) for i in range(60)]
    result = select_cumulative_evidence(pool, budget_bytes=60*512,
        targets={0:30, 1:30}, allocation_epoch=0)
    assert result.constraint_satisfied and result.certificate_counts == {0:30, 1:30}
    assert result.certificate_bytes == 60*512 and not result.violations
    result = select_cumulative_evidence([replace(r(0), size_bytes=0)], budget_bytes=0,
        targets={0:30, 1:30}, allocation_epoch=0)
    assert result.unavoidable_deficits == {0:29/30, 1:1.}
    assert result.avoidable_deficits == {0:0., 1:0.}
    assert result.opportunity_shortfall_groups == (0,1)
    assert result.budget_infeasible_groups == ()


def test_representative_restriction_can_discard_cheaper_feasible_record():
    cheap = replace(r(0), size_bytes=1, quality=0.)
    costly = replace(r(0), size_bytes=3, quality=1., nullifier='costly')
    pool = [cheap, costly]
    representatives = causal_representatives(pool, 0)
    assert representatives == (costly,)
    result = select_cumulative_evidence(pool, budget_bytes=1, targets={0:1}, allocation_epoch=0)
    assert Fraction(result.optimum_numerator, result.optimum_denominator) == 1
    assert oracle(representatives, {0:1}, 1) == 1
    assert cheap.size_bytes <= 1  # raw alternatives admit the floor, representative pool does not


def test_cumulative_preference_does_not_guarantee_no_starvation():
    state = CumulativeServiceState()
    pool = [replace(r(0), size_bytes=1), replace(r(1), size_bytes=2)]
    for epoch in range(4):
        result = state.allocate(pool, budget_bytes=2, targets={0:1}, allocation_epoch=epoch)
        assert [x.user_id for x in result.selected] == [0]
        assert result.constraint_satisfied
    assert state.distinct_debt({0:2}) == {0:1}


def test_empty_targets_and_untargeted_fill():
    result = select_cumulative_evidence([], budget_bytes=0, targets={}, allocation_epoch=0)
    assert result.globally_feasible and not result.violations
    result = select_cumulative_evidence([r(0,9)], budget_bytes=512, targets={}, allocation_epoch=0)
    assert result.counts == {9:1} and result.optimum_numerator == 0

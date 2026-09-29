import json
from pathlib import Path

import pytest

from airproof.v6_opportunity_procurement import (
    POLICIES, ProcurementOpportunity, procure_opportunities)


ROOT=Path(__file__).resolve().parents[1]


def opportunity(agent,group,uncertainty=1.,contact=.5,invite=8,energy=1):
    return ProcurementOpportunity(agent,7,group,group,invite,energy,uncertainty,2.,.85,contact)


def test_registered_family_is_finite_and_causal_ranking_targets_debt_and_uncertainty():
    assert len(POLICIES)==3
    pool=[opportunity(0,0,.2),opportunity(1,1,2.),opportunity(2,0,.1)]
    result=procure_opportunities(pool,policy=POLICIES[1],group_debt={0:10,1:0},
        decision_epoch=7,invitation_budget_bytes=8,sensing_energy_budget_units=1)
    assert [item.agent_id for item in result.invited]==[0]
    assert result.invitation_bytes==8 and result.sensing_energy_units==1


def test_representative_utility_uses_advertised_precision_and_contact_history():
    low=opportunity(0,0,1.,.1)
    high=ProcurementOpportunity(1,7,0,0,8,1,1.,1.,.85,.9)
    result=procure_opportunities([low,high],policy=POLICIES[2],group_debt={0:1},
        decision_epoch=7,invitation_budget_bytes=8,sensing_energy_budget_units=1)
    assert result.invited==(high,)


def test_no_free_sensing_and_no_future_or_duplicate_opportunities():
    with pytest.raises(ValueError,match='positive cost'):
        procure_opportunities([opportunity(0,0,energy=0)],policy=POLICIES[0],group_debt={},
            decision_epoch=7,invitation_budget_bytes=8,sensing_energy_budget_units=1)
    with pytest.raises(ValueError,match='current-epoch'):
        procure_opportunities([ProcurementOpportunity(0,8,0,0,8,1,1.,2.,.85,.5)],
            policy=POLICIES[0],group_debt={},decision_epoch=7,
            invitation_budget_bytes=8,sensing_energy_budget_units=1)
    with pytest.raises(ValueError,match='at most one'):
        procure_opportunities([opportunity(0,0),opportunity(0,1)],policy=POLICIES[0],
            group_debt={},decision_epoch=7,invitation_budget_bytes=16,
            sensing_energy_budget_units=2)


def test_both_budgets_bind():
    pool=[opportunity(0,0),opportunity(1,0)]
    result=procure_opportunities(pool,policy=POLICIES[0],group_debt={0:1},
        decision_epoch=7,invitation_budget_bytes=16,sensing_energy_budget_units=1)
    assert len(result.invited)==1


def test_prerequisite_audit_blocks_world_generation_and_retains_ceiling():
    path=ROOT/'reports/v6/fairness_opportunity_procurement_design/prerequisite_audit.json'
    audit=json.loads(path.read_text())
    assert audit['status']=='blocked'
    assert audit['required_mean_additional_origins']==42.75
    assert audit['eligible_dormant_origins_available'] is None
    assert audit['can_supply_required_origins'] is None
    assert audit['provisional_seeds_opened'] is False
    assert audit['confirmation_6204000_opened'] is False
    assert not audit['official_seed_collision']

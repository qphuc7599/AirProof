"""Causal invitation ranking for a future opt-in sensing opportunity interface.

This module selects invitations only.  It never constructs an observation or
interprets an invitation as consent, sensing, delivery, or service.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


POLICIES = (
    "group_debt",
    "group_debt_public_uncertainty",
    "group_debt_uncertainty_representative_utility",
)


@dataclass(frozen=True, slots=True)
class ProcurementOpportunity:
    agent_id: int
    epoch: int
    group: int
    cell: int
    invitation_bytes: int
    sensing_energy_units: int
    public_uncertainty: float
    advertised_sigma: float
    advertised_quality: float
    ontime_contact_probability: float

    def validate(self, decision_epoch: int) -> None:
        if min(self.agent_id, self.epoch, self.group, self.cell) < 0:
            raise ValueError("opportunity identifiers must be nonnegative")
        if self.epoch != decision_epoch:
            raise ValueError("only current-epoch eligible opportunities may be ranked")
        if self.invitation_bytes <= 0 or self.sensing_energy_units <= 0:
            raise ValueError("an invitation and a sensing action must have positive cost")
        values=(self.public_uncertainty,self.advertised_sigma,self.advertised_quality,
                self.ontime_contact_probability)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("opportunity covariates must be finite")
        if self.public_uncertainty < 0 or self.advertised_sigma <= 0:
            raise ValueError("uncertainty must be nonnegative and sigma positive")
        if not 0 <= self.advertised_quality <= 1 or not 0 <= self.ontime_contact_probability <= 1:
            raise ValueError("quality and contact probability must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class ProcurementDecision:
    policy: str
    invited: tuple[ProcurementOpportunity, ...]
    invitation_bytes: int
    sensing_energy_units: int
    score_by_agent: dict[int, float]


def _normalise(values: list[float]) -> list[float]:
    maximum=max(values,default=0.)
    return [0. if maximum<=0 else value/maximum for value in values]


def procure_opportunities(
    opportunities: Iterable[ProcurementOpportunity], *, policy: str,
    group_debt: Mapping[int, int], decision_epoch: int,
    invitation_budget_bytes: int, sensing_energy_budget_units: int,
) -> ProcurementDecision:
    """Allocate fixed invitation and sensing budgets using causal public fields.

    ``group_debt`` must be collector history through ``decision_epoch - 1``;
    ``public_uncertainty`` must be produced without current/future citizen truth;
    and ``ontime_contact_probability`` must use contact history strictly before the
    decision.  The caller owns and must enforce those provenance conditions.
    """
    if policy not in POLICIES:
        raise ValueError("unregistered opportunity-procurement policy")
    if invitation_budget_bytes < 0 or sensing_energy_budget_units < 0:
        raise ValueError("budgets must be nonnegative")
    pool=list(opportunities)
    if len({item.agent_id for item in pool}) != len(pool):
        raise ValueError("at most one current invitation opportunity per agent")
    for item in pool:item.validate(decision_epoch)
    if any(not isinstance(value,int) or value<0 for value in group_debt.values()):
        raise ValueError("group debt must contain nonnegative integers")

    debt=_normalise([float(group_debt.get(item.group,0)) for item in pool])
    uncertainty=_normalise([item.public_uncertainty for item in pool])
    # Mirrors the existing causal representative utility components without a
    # measurement value. Contact history supplies expected timely freshness.
    representative=[.45/(item.advertised_sigma**2)+.35*item.ontime_contact_probability
                    +.20*item.advertised_quality for item in pool]
    representative=_normalise(representative)
    scores=[]
    for index,item in enumerate(pool):
        score=debt[index]
        if policy != "group_debt":score+=uncertainty[index]
        if policy == "group_debt_uncertainty_representative_utility":
            score+=representative[index]
        scores.append(score)
    ranked=sorted(zip(pool,scores),key=lambda pair:(-pair[1],pair[0].invitation_bytes,
        pair[0].sensing_energy_units,pair[0].agent_id,pair[0].cell))
    invited=[];used_bytes=used_energy=0
    for item,score in ranked:
        if (used_bytes+item.invitation_bytes<=invitation_budget_bytes and
                used_energy+item.sensing_energy_units<=sensing_energy_budget_units):
            invited.append(item);used_bytes+=item.invitation_bytes
            used_energy+=item.sensing_energy_units
    return ProcurementDecision(policy,tuple(invited),used_bytes,used_energy,
        {item.agent_id:score for item,score in zip(pool,scores)})

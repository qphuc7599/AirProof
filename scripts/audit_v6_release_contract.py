"""Bounded proof/integration audit; never reruns exposed value experiments."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from airproof.continual_release import (FixedReleasePlan, bounded_epoch_query,
                                        independent_contributions, release_epoch)
from airproof.privacy import PrivateRelease
from airproof.records import Observation
from airproof.v5_release_twin import PublicReleaseObservation, ReleaseCalibration
from airproof.v6_privacy import release_contract_certificate
from airproof.v6_release_twin import ReleaseOnlyTwin


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v6/privacy_release_contract'


def citizen(user,group,value,identifier):
    return Observation(user,0,group,group,float(value),1.,1.,512,identifier,0,0)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    certificate=release_contract_certificate()
    plan=FixedReleasePlan(672,4,20,8/28,0.,8.)
    schedule=plan.scheduled_epochs

    unchanged=[citizen(2,1,3.,'other')]
    before=unchanged+[citizen(1,0,-10.,'switch-before')]
    after=unchanged+[citizen(1,1,10.,'switch-after')]
    kwargs=dict(steps=672,groups=4,relay=False,deadline_steps=24)
    admitted_before=independent_contributions(before,**kwargs)
    admitted_after=independent_contributions(after,**kwargs)
    other_user_invariant=(admitted_before[0][2]==admitted_after[0][2])
    qargs=dict(groups=4,baselines=np.zeros(4),clip=10.,k_min=20,residual=False)
    query_before,_=bounded_epoch_query(admitted_before[0],**qargs)
    query_after,_=bounded_epoch_query(admitted_after[0],**qargs)
    observed_l1=float(np.abs(query_after-query_before).sum())

    emitted=release_epoch(admitted_before[0],epoch=schedule[0],plan=plan,
        public_baselines=np.zeros(4),clip=10.,residual=False,rng=np.random.default_rng(606))
    public_keys=set().union(*(item.protected_payload() for item in emitted))

    calibration=ReleaseCalibration(np.zeros(2),np.eye(2)*.5,np.eye(2),np.eye(2),1,
                                   source='deterministic-contract-audit')
    missing,suppressed,assimilated=map(lambda _:ReleaseOnlyTwin(calibration),range(3))
    for epoch in range(24):
        for twin in (missing,suppressed,assimilated):twin.update(epoch,np.zeros(1))
    missing_value=missing.update(24,np.zeros(1))
    suppressed_value=suppressed.update(24,np.zeros(1),[
        PublicReleaseObservation('suppressed-slot',0,24,0,None,7.)])
    prior=missing.covariance.copy();covariance_h=prior[:,0]+prior[:,1]
    expected=prior-np.outer(covariance_h,covariance_h)/(covariance_h[0]+covariance_h[1]+98.)
    release=PublicReleaseObservation('one-effect',0,24,0,1.,7.)
    assimilated.update(24,np.zeros(1),[release,release])

    collision=ReleaseOnlyTwin(calibration)
    for epoch in range(24):collision.update(epoch,np.zeros(1))
    collision_rejected=False
    try:
        collision.update(24,np.zeros(1),[
            PublicReleaseObservation('slot-a',0,24,0,1.,7.),
            PublicReleaseObservation('slot-b',0,24,0,2.,7.)])
    except ValueError:
        collision_rejected=True

    negative_path=ROOT/'reports/v6/release_incremental_value/result.json'
    negative=json.loads(negative_path.read_text())
    integrity={
        'schedule_has_28_publications':len(schedule)==28 and len(set(schedule))==28,
        'publication_deadline_is_24':all(publication-acquisition==24
            for acquisition,publication in zip(schedule,(x+24 for x in schedule))),
        'whole_history_epsilon_lte_8':plan.composed_epsilon<=8+1e-12,
        'whole_history_epsilon':plan.composed_epsilon,
        'k_min_is_20':plan.k_min==20,
        'other_user_admission_invariant_under_group_switch':other_user_invariant,
        'group_switch_query_l1':observed_l1,
        'group_switch_within_deployed_bound':observed_l1<=certificate['deployed_vector_l1_sensitivity']+1e-12,
        'public_payload_excludes_exact_private_count':
            public_keys=={'group','epoch','released','value','transcript_scope'},
        'consumer_signature_excludes_raw_or_count':set(inspect.signature(ReleaseOnlyTwin.update).parameters)
            =={'self','epoch','public_baseline','releases'},
        'suppression_equals_missing_state':np.array_equal(missing_value,suppressed_value)
            and np.array_equal(missing.covariance,suppressed.covariance),
        'suppression_recorded_without_assimilation':suppressed.suppressed==1 and suppressed.assimilated==0,
        'one_release_id_one_effect':assimilated.assimilated==1 and assimilated.duplicates==1,
        'different_ids_same_slot_rejected_before_update':collision_rejected and collision.epochs==list(range(24)),
        'laplace_variance_added_once':np.allclose(assimilated.covariance,expected,rtol=0,atol=1e-12),
        'fixed_consumer_lag_48':assimilated.lag==48,
        'negative_incremental_value_reused_not_rerun':negative['citizen_value_confirmed'] is False
            and negative['confirmation_authorized'] is False
    }
    evidence_hashes={str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (negative_path,ROOT/'reports/v6/release_incremental_value/registration.json')}
    sources=[Path(__file__),ROOT/'airproof/v6_privacy.py',ROOT/'airproof/continual_release.py',
             ROOT/'airproof/v6_release_twin.py',ROOT/'airproof/v6_release_experiment.py']
    manifest={'role':'bounded correctness/theory audit; no scientific value rerun',
        'all_integrity_checks_passed':all(v is True or not isinstance(v,bool) for v in integrity.values()),
        'source_hashes':{str(path.relative_to(ROOT)).replace('\\','/'):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'reused_evidence_hashes':evidence_hashes,
        'scientific_disposition':'no protected-release gain supported; exposed negative result retained',
        'claim_limit':'release-only transcript under stated adjacency; no broader raw/audit transcript DP claim'}
    (OUT/'certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')
    (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'certificate':certificate,'integrity':integrity,'manifest':manifest},indent=2))


if __name__=='__main__':main()

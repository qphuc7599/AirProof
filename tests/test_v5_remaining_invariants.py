"""Additional bounded properties; no campaign or frozen source modification."""
from dataclasses import replace
import random
import numpy as np

from airproof.records import Observation
from airproof.v5_transport import TransportTrace,simulate_transport
from airproof.v5_experiment import reserve_contributions
from airproof.v5_estimator import EstimatorConfig,estimate_public_field


def record(user,epoch,group,identifier):
    return Observation(user,epoch,group,group,12.,1.,1.,512,identifier,999,888)


def test_whole_history_group_replacement_and_flood_preserve_other_admission():
    rng=random.Random(43291)
    for trial in range(20):
        gateways=tuple(tuple(n for n in range(5) if rng.random()<.35) for _ in range(9))
        peers=tuple(((0,1),(1,2),(2,3),(3,4)) for _ in range(9))
        trace=TransportTrace(trial,5,6,3,(0,1,2,3,0),gateways,peers,'bounded-history')
        ordinary=[record(u,e,(u+e)%4,f'{u}:{e}') for u in range(5) for e in range(6)]
        fixed=[r for r in ordinary if r.user_id!=0]
        replacement=[record(0,e,(e+3)%4,f'replaced:{e}:{j}') for e in range(6) for j in range(8)]
        cfg={'transport':{'raw_buffer_bytes':1024,'release_buffer_bytes':512,'ttl_epochs':2}}
        baseline=None
        for pool in (ordinary,fixed+replacement,fixed):
            result=simulate_transport(trace,pool,cfg)
            other_arrivals={k:v for k,v in result.release_arrivals.items() if k[0]!=0}
            admitted=reserve_contributions(pool,result.release_arrivals,steps=6,deadline=2)
            other_records={e:{u:records for u,records in users.items() if u!=0}
                           for e,users in admitted.items() if any(u!=0 for u in users)}
            current=(other_arrivals,other_records)
            if baseline is None:baseline=current
            else:assert current==baseline


def test_disabled_input_clip_and_output_cap_ignore_their_numeric_limits():
    public=np.full((3,1),10.)
    observations=[record(0,e,0,str(e)) for e in range(3)]
    observations=[replace(r,value=100.) for r in observations]
    arrivals={r.nullifier:r.epoch for r in observations}
    unrestricted=EstimatorConfig(lag=0,input_clip=False,output_cap=False,delta=.01,cap=.01,tolerance=1e-11)
    run=lambda cfg:estimate_public_field(public,np.zeros((1,2)),observations,cfg,arrival_map=arrivals)
    a=run(unrestricted)
    b=run(replace(unrestricted,delta=1000.,cap=1000.))
    np.testing.assert_array_equal(a.live,b.live)
    np.testing.assert_array_equal(a.reconstructed,b.reconstructed)
    assert float(np.max(a.live-public))>8.
    assert a.diagnostics['input_clip_count']==0
    clipped=run(replace(unrestricted,input_clip=True,delta=4.))
    capped=run(replace(unrestricted,output_cap=True,cap=8.))
    assert np.max(clipped.live-public)<=4.
    assert np.max(capped.live-public)<=8.
    assert not np.array_equal(a.live,clipped.live)
    assert not np.array_equal(a.live,capped.live)

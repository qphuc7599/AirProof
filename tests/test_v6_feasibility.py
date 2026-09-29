from dataclasses import replace
from airproof.v5_transport import TransportTrace
from airproof.records import Observation
from airproof.v6_feasibility import timely_reachability_upper_bound,cap_rmse_floor


def test_no_same_slot_multihop_in_relaxed_bound():
    trace=TransportTrace(1,3,1,2,(0,0,0),((),(2,),()),(((0,1),(1,2)),(),()),'test')
    r=Observation(0,0,0,0,1.,1.,1.,512,'a',None,None)
    assert timely_reachability_upper_bound(trace,[r],1)['timely_records_upper']==0
    trace=replace(trace,peer_contacts=(((0,1),),((1,2),),()),gateway_contacts=((),(),(2,)))
    assert timely_reachability_upper_bound(trace,[r],2)['timely_records_upper']==1


def test_cap_bound_sharp_projection():
    assert cap_rmse_floor([20.],[10.],8.)==2.

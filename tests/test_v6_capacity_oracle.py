import pytest
from airproof.records import Observation
from airproof.v5_transport import TransportTrace
from airproof.v6_capacity_oracle import coupled_service_bound


def r(user=0,epoch=0,group=0):
    return Observation(user,epoch,0,group,1.,1.,1.,512,f'{user}:{epoch}:{group}',None,None)


def trace(nodes=2,steps=1,gateways=None,peers=None):
    return TransportTrace(1,nodes,steps,0,(0,)*nodes,
        tuple(gateways or [()]*steps),tuple(peers or [()]*steps),'fixture')


def test_shared_contact_and_processing_are_coupled():
    t=trace(gateways=[(0,1)])
    assert coupled_service_bound(t,[r(0),r(1)],{},processing_budget_bytes=512)['objective']==pytest.approx(1)
    assert coupled_service_bound(t,[r(0),r(1)],{},processing_budget_bytes=1024)['objective']==pytest.approx(2)
    t=trace(nodes=1)
    assert coupled_service_bound(t,[r()],{})['objective']==0


def test_gateway_512_capacity_cannot_carry_fractional_packet():
    result=coupled_service_bound(trace(gateways=[(0,)]),[r()],
        {'transport':{'capacity_bytes_per_direction':512}})
    assert result['objective']==0


def test_cumulative_identity_and_representative_processing():
    t=trace(nodes=1,steps=2,gateways=[(0,),(0,)])
    out=coupled_service_bound(t,[r(epoch=0),r(epoch=1)],{})
    assert out['objective']==pytest.approx(1)
    assert out['upper_bound']==pytest.approx(1)


def test_peer_next_slot_and_lag():
    t=trace(steps=2,gateways=[(1,),()],peers=[((0,1),),()])
    assert coupled_service_bound(t,[r()],{})['objective']==0
    t=trace(steps=2,gateways=[(),(1,)],peers=[((0,1),),()])
    assert coupled_service_bound(t,[r()],{})['objective']==pytest.approx(1)
    assert coupled_service_bound(t,[r()],{},useful_lag=0)['objective']==0


def test_buffer_and_deferred_processing():
    t=trace(nodes=1,steps=2,gateways=[(0,),()])
    out=coupled_service_bound(t,[r()],{},processing_budget_bytes={0:0,1:512})
    assert out['objective']==pytest.approx(1)
    assert coupled_service_bound(t,[r()],{'transport':{'raw_buffer_bytes':0}})['objective']==0
    assert coupled_service_bound(t,[r()],{},max_variables=1)['status']=='resource_limit'


def test_fractional_bound_and_dual_direction():
    t=trace(gateways=[(0,1)])
    out=coupled_service_bound(t,[r(0),r(1)],{},processing_budget_bytes=768)
    assert out['objective']==pytest.approx(1.5)
    assert out['upper_bound']>=1.5-1e-10
    assert out['max_capacity_violation']<1e-8
    assert out['max_equality_residual']<1e-8


@pytest.mark.parametrize('contact_mask',range(16))
def test_exhaustive_two_user_two_epoch_direct_fixture(contact_mask):
    # Exhaust all binary selected actions; no peers, one acquired record per user.
    from itertools import product
    gateways=[tuple(u for u in range(2) if contact_mask & (1<<(2*t+u))) for t in range(2)]
    t=trace(steps=2,gateways=gateways)
    integer_optimum=0
    for actions in product((0,1),repeat=4):
        if any(actions[2*epoch+u] and not any(u in gateways[earlier] for earlier in range(epoch+1))
               for epoch in range(2) for u in range(2)):continue
        if any(sum(actions[2*epoch:2*epoch+2])>1 for epoch in range(2)):continue
        if any(actions[u]+actions[2+u]>1 for u in range(2)):continue
        integer_optimum=max(integer_optimum,sum(actions))
    out=coupled_service_bound(t,[r(0),r(1)],{},processing_budget_bytes=512)
    assert out['upper_bound']>=integer_optimum-1e-8
    assert out['objective']==pytest.approx(integer_optimum)


@pytest.mark.parametrize('policy',['airproof_deadline','deadline6','combined','binary_spray_wait','epidemic_cap'])
def test_bound_dominates_executable_small_policy(policy):
    from airproof.v6_transport import generate_transport_trace,simulate_transport
    cfg={'world':{'agents':4,'steps':5,'grid_side':2,'groups':2},
         'scale':{'drain_epochs':3},'network':{'availability':.6,'contact_probability':1.,
         'outage_median_hours':1,'outage_p95_hours':2},
         'transport':{'raw_buffer_bytes':1024}}
    t=generate_transport_trace(cfg,6006001)
    records=[r(user=u,epoch=e,group=u%2) for u in range(4) for e in range(5)]
    executed=simulate_transport(t,records,cfg,policy)
    bound=coupled_service_bound(t,records,cfg)
    assert bound['upper_bound']>=executed.metrics['collector_selected_unique']-1e-8

from dataclasses import asdict, replace

import pytest
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airproof.records import Observation
from airproof.v6_transport import (TransportTrace, decode_packet, encode_packet,
                                  generate_transport_trace, simulate_transport)


def record(user=0, epoch=0, group=0, suffix=""):
    return Observation(user, epoch, 0, group, 10., 1., 1., 512,
                       f"{user}:{epoch}:{group}:{suffix}", 999, 888)


def trace(nodes=2, acquisition=1, drain=2, gateways=None, peers=None):
    horizon = acquisition + drain
    return TransportTrace(1, nodes, acquisition, drain, tuple(i % 2 for i in range(nodes)),
                          tuple(gateways or [()] * horizon), tuple(peers or [()] * horizon), "manual")



from airproof.v6_transport import relaxed_feasibility_bound, necessary_delivery_fraction


def test_lag_and_expiry_keep_denominator():
    r=record()
    t=trace(drain=8, gateways=[()]*8+[(0,)])
    result=simulate_transport(t,[r],{})
    assert result.raw_arrivals[r.nullifier]==8
    assert result.metrics["raw_late_delivered"]==1
    assert result.metrics["raw_generated"]==1
    assert result.metrics["raw_timely_delivered"]==0


def test_replacement_keeps_fresh_evidence_and_release_independent():
    pool=[record(epoch=0),record(epoch=7)]
    t=trace(acquisition=8,drain=0,gateways=[()]*7+[(0,)])
    cfg={"transport":{"raw_buffer_bytes":512}}
    newer=simulate_transport(t,pool,cfg,"replacement")
    older=simulate_transport(t,pool,cfg,"deadline6")
    assert pool[1].nullifier in newer.raw_arrivals
    assert pool[0].nullifier in older.raw_arrivals
    assert newer.release_arrivals==older.release_arrivals
    assert newer.metrics["raw_replacements"]==1
    assert newer.metrics["token_violations"]==0


def test_feedback_is_selected_delayed_and_charged():
    r=record()
    t=trace(drain=1,gateways=[(0,),(0,)])
    yes=simulate_transport(t,[r],{},"unique_feedback")
    no=simulate_transport(t,[r],{},"unique_feedback",collector_selector=lambda records, epoch: ())
    assert yes.metrics["feedback_bytes"]==16
    assert no.metrics["feedback_bytes"]==0
    assert yes.metrics["control_bytes"]-no.metrics["control_bytes"]==16
    assert yes.metrics["local_decisions"][0]["epoch"]==1
    assert yes.metrics["max_control_direction_bytes"]<=128


def test_payload_and_claimed_arrival_not_routing_oracles():
    t=trace(drain=2,gateways=[(),(),(1,)],peers=[((0,1),),(),()])
    original=record()
    changed=replace(original,value=1e20,sigma=1e20,quality=1e20,corrupted=True,direct_arrival=0)
    a=simulate_transport(t,[original],{})
    b=simulate_transport(t,[changed],{})
    assert a==b
    assert a.metrics["token_violations"]==0


def test_collector_rejection_preserves_physical_ack_and_excludes_allocation():
    r=record()
    selected_inputs=[]
    t=trace(drain=0,gateways=[(0,)])
    result=simulate_transport(t,[r],{},raw_acceptor=lambda record, epoch: False,
        collector_selector=lambda arrivals, epoch: selected_inputs.extend(arrivals) or ())
    assert result.raw_arrivals=={r.nullifier:0}
    assert result.metrics["raw_gateway_transmissions"]==1
    assert result.metrics["raw_payload_bytes"]==512
    assert result.metrics["raw_remaining_copies"]==0
    assert result.metrics["token_violations"]==0
    assert selected_inputs==[]


def test_bound_capacity_and_resource_limit():
    t=trace(drain=0,gateways=[(0,)])
    assert relaxed_feasibility_bound(t,[record()],{})["upper_bound"]==1
    assert relaxed_feasibility_bound(t,[record()],{"transport":{"capacity_bytes_per_direction":512}})["upper_bound"]==0
    t=trace(drain=3)
    assert relaxed_feasibility_bound(t,[record()],{},max_states=1)["status"]=="resource_limit"
    assert necessary_delivery_fraction(24)==pytest.approx(.15)


@pytest.mark.parametrize("policy", [
    "deadline6", "replacement", "unique_feedback", "marginal_copy", "combined",
    "combined_no_service_priority", "combined_group_fair",
])
def test_all_v6_policies_finite_on_seeded_trace(policy):
    config={"world":{"agents":8,"steps":12,"grid_side":2,"groups":2},
            "scale":{"drain_epochs":8},"network":{"availability":.7,"contact_probability":1.},
            "transport":{"raw_buffer_bytes":1024}}
    t=generate_transport_trace(config,4)
    pool=[record(user=u,epoch=e,group=u%2) for u in range(8) for e in range(12)]
    result=simulate_transport(t,pool,config,policy)
    m=result.metrics
    assert m["token_violations"]==0
    assert m["max_live_copies"]<=4
    assert m["max_raw_buffer_bytes"]<=1024
    assert m["max_contact_direction_bytes"]<=1024
    assert m["max_control_direction_bytes"]<=128
    assert m["raw_generated"]==m["raw_delivered"]+m["raw_undelivered"]
    assert m["raw_delivered"]==m["raw_timely_delivered"]+m["raw_late_delivered"]

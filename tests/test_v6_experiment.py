from airproof.records import Observation
from airproof.v6_transport import TransportTrace
from airproof.v6_experiment import integrated_transport


def test_backlog_selection_and_causal_receipts():
    records=[Observation(i,0,0,0,10.,1.,1.,512,str(i),None,None) for i in range(2)]
    trace=TransportTrace(1,2,1,3,(0,0),((0,1),(0,1),(0,1),(0,1)),((),(),(),()),'fixture')
    cfg={'world':{'groups':1},'scheduler':{'budget_bytes_per_epoch':512,'target_contributors':30}}
    result,c=integrated_transport(trace,records,cfg)
    assert len(c.selected)==2
    assert sorted(c.selection_times.values())==[0,1]
    assert len(c.returns.delivered)==4
    assert result.metrics["audit_verified_return_pairs"]==2
    assert result.metrics["audit_return_control_bytes"]==6*16
    assert result.metrics["total_wire_bytes"]==sum(result.metrics[k] for k in ("control_bytes","raw_payload_bytes","release_payload_bytes","receipt_payload_bytes"))
    assert min(c.returns.delivered.values())>0
    assert result.metrics['max_contact_direction_bytes']<=1024
    assert result.metrics['receipt_payload_bytes']==c.returns.transmitted_bytes
    assert all(r.direct_arrival is None for r in records)


def test_audit_capacity_rejection_cannot_enter_twin_but_upload_is_charged():
    records=[Observation(0,0,0,0,10.,1.,1.,512,'rejected',None,None)]
    trace=TransportTrace(1,1,1,0,(0,),((0,),),((),),'reject')
    result,c=integrated_transport(trace,records,{'audit':{'collector_return_buffer_bytes':0}})
    assert result.metrics['raw_delivered']==1
    assert result.metrics['raw_payload_bytes']==512
    assert result.metrics['audit_admission_rejected']==1
    assert not c.selected and not c.accepted and not c.audit.acceptance_ids
    assert not c.returns.dropped
    assert result.metrics['raw_remaining_copies']==0
    assert result.metrics['token_violations']==0


def test_declared_return_frame_budget_is_an_admission_gate():
    records=[Observation(0,0,0,0,10.,1.,1.,512,'frame-rejected',None,None)]
    trace=TransportTrace(1,1,1,0,(0,),((0,),),((),),'frame-reject')
    result,c=integrated_transport(trace,records,{'audit':{'max_return_frames':0}})
    assert result.metrics['raw_delivered']==1
    assert result.metrics['raw_payload_bytes']==512
    assert result.metrics['audit_admission_rejected']==1
    assert result.metrics['receipt_issued']==0
    assert not c.selected


def test_audit_useful_deadline_is_inclusive_and_stale_upload_stays_charged():
    records=[Observation(i,0,0,0,10.,1.,1.,512,f'deadline-{i}',None,None) for i in range(2)]
    trace=TransportTrace(1,2,1,2,(0,0),((),(0,),(1,)),((),(),()),'audit-deadline')
    result,c=integrated_transport(trace,records,{'world':{'groups':1},
        'transport':{'useful_lag_epochs':1}})
    assert result.metrics['raw_delivered']==2
    assert result.metrics['raw_payload_bytes']==1024
    assert result.metrics['audit_attempted']==2
    assert result.metrics['audit_stale_rejected']==1
    assert result.metrics['audit_capacity_rejected']==0
    assert result.metrics['audit_admission_rejected']==1
    assert c.accepted=={'deadline-0'}
    assert [r.nullifier for r in c.selected]==['deadline-0']
    assert c.stale_rejected[0]['arrival_epoch']==2
    assert result.metrics['raw_remaining_copies']==0


def test_absent_origin_contacts_keep_failed_return_visible():
    records=[Observation(0,0,0,0,10.,1.,1.,512,'stranded',None,None)]
    trace=TransportTrace(1,1,1,25,(0,),((0,),)+((),)*25,((),)*26,'stranded')
    result,c=integrated_transport(trace,records,{})
    assert len(c.selected)==1  # signed inclusion already exists at collector
    assert result.metrics['audit_terminal_issued']==1
    assert result.metrics['audit_verified_return_pairs']==0
    assert result.metrics['audit_unreturned_pairs']==1
    assert result.metrics['receipt_expired']==2
    assert result.metrics['receipt_pending']==0
    assert result.metrics['receipt_payload_bytes']==0
    assert result.metrics['audit_collector_log']['resolved_on_time']==1

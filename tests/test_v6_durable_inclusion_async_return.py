import json
from dataclasses import replace

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airproof.audit import SignedTreeHead, verify_tree_head
from airproof.records import Observation
from airproof.v6_audit_return import (DurableInclusionAsyncReturn,
    durable_record_storage_bound,verify_returned_bundle)
from airproof.v6_experiment import IntegratedCollector,integrated_transport
from airproof.v6_resources import ContactBudget
from airproof.v6_transport import TransportTrace


def _config(**audit):
    fixed={'return_mode':'durable_inclusion_async_return_v1','max_records':8,
           'max_content_bytes':2048,'durable_log_capacity_bytes':65536,
           'collector_return_buffer_bytes':18432,'inclusion_deadline_epochs':24}
    fixed.update(audit);return {'world':{'groups':1},'audit':fixed}


def test_durable_storage_is_reserved_before_signed_inclusion_and_exact_boundary_accepts():
    content=b'bounded-content';bound=durable_record_storage_bound(len(content),max_records=8)
    rejected=DurableInclusionAsyncReturn(Ed25519PrivateKey.generate(),max_records=8,
        max_content_bytes=2048,durable_log_capacity_bytes=bound-1)
    assert rejected.accept(content,0,0) is None
    assert rejected.tree.size==0 and rejected.log.audit(now=0)['receipts']==0
    engine=DurableInclusionAsyncReturn(Ed25519PrivateKey.generate(),max_records=8,
        max_content_bytes=2048,durable_log_capacity_bytes=bound)
    receipt=engine.accept(content,0,0)
    assert receipt is not None and engine.tree.size==1
    assert engine.log.audit(now=0)['resolved_on_time']==1
    assert 0<engine.storage_bytes<=bound and engine.storage_reserved_bytes==0


def test_missing_reverse_contact_does_not_block_allocation_or_claim_origin_verification():
    record=Observation(0,0,0,0,10.,1.,1.,512,'durable-stranded',None,None)
    trace=TransportTrace(1,1,1,2,(0,),((0,),(),()),((),(),()),'durable-stranded')
    result,c=integrated_transport(trace,[record],_config())
    assert [r.nullifier for r in c.selected]==['durable-stranded']
    assert result.metrics['audit_collector_log']['resolved_on_time']==1
    assert result.metrics['audit_verified_return_pairs']==0
    assert result.metrics['audit_externally_checkpointed']==0
    assert result.metrics['audit_publicly_checkpointed']==0
    assert result.metrics['audit_durable_storage_bytes']>0


def test_receipt_can_arrive_without_terminal_and_omission_remains_visible():
    collector=IntegratedCollector(_config(),1)
    record=Observation(0,0,0,0,10.,1.,1.,512,'omission',None,None)
    assert collector.accept(record,0)
    collector.return_receipt(0,1,ContactBudget())
    completed=[json.loads(x) for x in collector.receivers[0].completed.values()]
    assert [x['kind'] for x in completed]==['acceptance']
    assert collector.returned_pairs()==[] and collector.externally_checkpointed()==set()


def test_async_return_preserves_content_binding_and_detects_forged_or_rollback_heads():
    key=Ed25519PrivateKey.generate();engine=DurableInclusionAsyncReturn(key,max_records=8,
        max_content_bytes=2048,durable_log_capacity_bytes=65536)
    receipts=[engine.accept(x,0,i) for i,x in enumerate((b'first',b'second'))]
    first=engine.durable_records[receipts[0].receipt_id]
    assert verify_returned_bundle(first['receipt'],first['witness'],first['checkpoint'],b'first',key.public_key())
    assert not verify_returned_bundle(first['receipt'],first['witness'],first['checkpoint'],b'mutated',key.public_key())
    forged=replace(first['checkpoint'],root_hex='0'*64)
    assert not verify_tree_head(forged,key.public_key())
    second=engine.durable_records[receipts[1].receipt_id]['checkpoint']
    assert second.tree_size>first['checkpoint'].tree_size
    # A client retaining the newer signed head detects replay of the older size.
    assert first['checkpoint'].tree_size<second.tree_size
    split=replace(second,root_hex=first['checkpoint'].root_hex)
    assert not verify_tree_head(split,key.public_key())


def test_duplicate_record_still_has_one_numerical_influence():
    collector=IntegratedCollector(_config(),1)
    record=Observation(0,0,0,0,10.,1.,1.,512,'one-effect',None,None)
    assert collector.accept(record,0)
    try:collector.accept(record,0)
    except ValueError:pass
    else:raise AssertionError('duplicate acceptance must fail')
    selected=collector.allocate([record],0)
    assert len(selected)==1 and len(collector.audit.acceptance_ids)==1

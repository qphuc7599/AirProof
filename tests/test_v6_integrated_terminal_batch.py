import json
from dataclasses import replace

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airproof.audit import IngestionReceipt, ReceiptStatusWitness, SignedTreeHead
from airproof.records import Observation
from airproof.v6_audit_return import (
    SameOriginTerminalBatchAuditReturn,
    TerminalAuditReturn,
    acceptance_reservation_frames,
    terminal_batch_reservation_frames,
    verify_returned_bundle,
)
from airproof.v6_experiment import integrated_transport
from airproof.v6_receipts import HEADER, Receiver
from airproof.v6_resources import ContactBudget
from airproof.v6_transport import TransportTrace


def _drain(engine, origin=0, start=1, stop=30):
    for epoch in range(start, stop):
        engine.queue.contact(origin, epoch, ContactBudget(), engine.receivers[origin])


def test_batch_one_has_exact_per_record_frame_parity():
    key=Ed25519PrivateKey.from_private_bytes(bytes(range(1,33)))
    individual=TerminalAuditReturn(key,batch_size=1,max_records=8)
    batched=SameOriginTerminalBatchAuditReturn(key,batch_size=1,batch_timeout_epochs=0,max_records=8)
    assert individual.accept(b'one',0,0) is not None
    assert batched.accept(b'one',0,0) is not None
    assert individual.issued_frames==batched.issued_frames==3
    assert batched.flushed_batch_sizes==[1] and batched.terminal_object_count==1


def test_first_promise_requires_acceptance_plus_full_terminal_envelope_capacity():
    required=acceptance_reservation_frames(max_records=8)+terminal_batch_reservation_frames(1,max_records=8)
    key=Ed25519PrivateKey.generate()
    short=SameOriginTerminalBatchAuditReturn(key,batch_size=2,batch_timeout_epochs=2,
        max_records=8,max_return_frames=required-1,buffer_bytes=required*512)
    assert short.accept(b'no-promise',0,0) is None
    assert not short.acceptance_ids and not short.log.audit(now=0)['receipts']
    exact=SameOriginTerminalBatchAuditReturn(key,batch_size=2,batch_timeout_epochs=2,
        max_records=8,max_return_frames=required,buffer_bytes=required*512)
    assert exact.accept(b'covered',0,0) is not None
    assert exact.committed_frames==required
    assert exact.queue.committed_occupancy==required*512


def test_partial_batch_flushes_at_fixed_timeout_and_verifies_both_records():
    key=Ed25519PrivateKey.generate()
    engine=SameOriginTerminalBatchAuditReturn(key,batch_size=4,batch_timeout_epochs=2,max_records=8)
    engine.receivers={0:Receiver(key.public_key())}
    receipts=[engine.accept(x,0,0) for x in (b'a',b'b')]
    assert not engine.terminal_ids
    engine.advance(1);assert not engine.terminal_ids
    engine.advance(2)
    assert engine.flushed_batch_sizes==[2] and engine.terminal_object_count==1
    _drain(engine,start=3)
    objects=[json.loads(p) for p in engine.receivers[0].completed.values()]
    returned=[IngestionReceipt(**o['receipt']) for o in objects if o['kind']=='acceptance']
    terminal=next(o for o in objects if o['kind']=='terminal_batch')
    head=SignedTreeHead(**terminal['checkpoint'])
    witnesses={w['receipt_id']:ReceiptStatusWitness(**w) for w in terminal['witnesses']}
    by_id={r.receipt_id:r for r in returned}
    assert all(verify_returned_bundle(by_id[r.receipt_id],witnesses[r.receipt_id],head,content,key.public_key())
               for r,content in zip(receipts,(b'a',b'b')))


def test_missing_reverse_contacts_leave_batched_promises_expired_and_visible():
    record=Observation(0,0,0,0,10.,1.,1.,512,'stranded-batch',None,None)
    trace=TransportTrace(1,1,1,25,(0,),((0,),)+((),)*25,((),)*26,'stranded-batch')
    cfg={'world':{'groups':1},'audit':{'return_mode':'same_origin_terminal_batch_v1',
        'terminal_batch_size':4,'terminal_batch_timeout_epochs':2,'max_records':8}}
    result,c=integrated_transport(trace,[record],cfg)
    assert len(c.selected)==1 and result.metrics['audit_terminal_issued']==1
    assert result.metrics['audit_terminal_objects_issued']==1
    assert result.metrics['audit_verified_return_pairs']==0
    assert result.metrics['receipt_expired']==2 and result.metrics['receipt_pending']==0


def test_batched_witnesses_reject_content_reorder_and_signature_forgery():
    key=Ed25519PrivateKey.generate()
    engine=SameOriginTerminalBatchAuditReturn(key,batch_size=2,batch_timeout_epochs=2,max_records=8)
    contents=(b'first',b'second');receipts=[engine.accept(x,0,0) for x in contents]
    batch=next(r for r in engine.queue.pending if len(r.frames)>1)
    length=HEADER.unpack(batch.frames[0][:HEADER.size])[4]
    payload=b''.join(f[HEADER.size:-64] for f in batch.frames)[:length]
    obj=json.loads(payload);head=SignedTreeHead(**obj['checkpoint'])
    witnesses=[ReceiptStatusWitness(**w) for w in obj['witnesses']]
    assert not all(verify_returned_bundle(r,w,head,c,key.public_key())
                   for r,w,c in zip(receipts,witnesses,reversed(contents)))
    forged=replace(witnesses[0],signature_b64='A'*86+'==')
    assert not verify_returned_bundle(receipts[0],forged,head,contents[0],key.public_key())


def test_batched_mode_keeps_exact_useful_lag_boundary_and_rejects_plus_one():
    records=[Observation(i,0,0,0,10.,1.,1.,512,f'batch-deadline-{i}',None,None) for i in range(2)]
    trace=TransportTrace(1,2,1,2,(0,0),((),(0,),(1,)),((),(),()),'batch-deadline')
    cfg={'world':{'groups':1},'transport':{'useful_lag_epochs':1},
         'audit':{'return_mode':'same_origin_terminal_batch_v1','terminal_batch_size':2,
                  'terminal_batch_timeout_epochs':1,'max_records':8}}
    result,c=integrated_transport(trace,records,cfg)
    assert result.metrics['raw_delivered']==2 and result.metrics['raw_payload_bytes']==1024
    assert result.metrics['audit_stale_rejected']==1 and result.metrics['audit_capacity_rejected']==0
    assert c.accepted=={'batch-deadline-0'}
    assert [r.nullifier for r in c.selected]==['batch-deadline-0']

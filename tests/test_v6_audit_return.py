from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from airproof.v6_audit_return import (TerminalAuditReturn,BatchTerminalAuditReturn,
    verify_returned_bundle,verify_returned_batch)
from airproof.v6_resources import ContactBudget

def test_terminal_evidence_requires_fragment_contacts():
    engine=TerminalAuditReturn(Ed25519PrivateKey.generate(),batch_size=2)
    engine.accept(b'one',0,0);engine.accept(b'two',0,0)
    assert len(engine.terminal_ids)==2
    assert not engine.contact(0,0,ContactBudget())
    for epoch in range(1,20):engine.contact(0,epoch,ContactBudget())
    assert len(engine.queue.delivered)==4
    assert engine.queue.transmitted_bytes>4*512
    assert engine.queue.peak_buffer<=18432

def test_delivered_terminal_proof_binds_original_content():
    import json
    from dataclasses import replace
    from airproof.audit import IngestionReceipt,ReceiptStatusWitness,SignedTreeHead,verify_receipt_inclusion
    key=Ed25519PrivateKey.generate();engine=TerminalAuditReturn(key,batch_size=1)
    engine.accept(b'committed-original',0,0)
    for epoch in range(1,16):engine.contact(0,epoch,ContactBudget())
    objects=[json.loads(p) for p in engine.receivers[0].completed.values()]
    receipt=IngestionReceipt(**next(o['receipt'] for o in objects if o['kind']=='acceptance'))
    terminal=next(o for o in objects if o['kind']=='terminal')
    witness=ReceiptStatusWitness(**terminal['witness']);checkpoint=SignedTreeHead(**terminal['checkpoint'])
    assert verify_returned_bundle(receipt,witness,checkpoint,b'committed-original',key.public_key())
    assert verify_receipt_inclusion(receipt,witness,b'committed-original',key.public_key())
    assert not verify_receipt_inclusion(receipt,witness,b'mutated',key.public_key())
    assert not verify_receipt_inclusion(receipt,replace(witness,root_hex='00'*32),b'committed-original',key.public_key())

def test_admission_reserves_terminal_before_issuing_receipt():
    key=Ed25519PrivateKey.generate();engine=TerminalAuditReturn(key,batch_size=32,max_records=32,max_return_frames=24)
    receipts=[engine.accept(f'item-{i}'.encode(),0,0) for i in range(32)]
    admitted=[r for r in receipts if r is not None]
    assert len(admitted)<32 and len(admitted)==len(engine.acceptance_ids)
    assert engine.log.audit(now=0)['receipts']==len(admitted)
    engine.finalize(0)
    assert not engine.queue.reservations and not engine.frame_reservations
    assert engine.issued_frames<=24 and engine.queue.committed_occupancy<=engine.queue.buffer_bytes
    for epoch in range(1,25):engine.contact(0,epoch,ContactBudget())
    assert not engine.queue.pending and len(engine.queue.delivered)==2*len(admitted)

def test_batch_receipt_binds_every_ordered_record_at_constant_return_objects():
    import json
    from airproof.audit import IngestionReceipt,ReceiptStatusWitness,SignedTreeHead
    key=Ed25519PrivateKey.generate();engine=BatchTerminalAuditReturn(key,max_batches=1,max_return_frames=3)
    contents=(b'a',b'b',b'c');receipt=engine.accept_batch(contents,7,0)
    assert receipt is not None and engine.accepted_records==3
    for epoch in range(1,5):engine.contact(7,epoch,ContactBudget())
    objects=[json.loads(p) for p in engine.engine.receivers[7].completed.values()]
    returned=IngestionReceipt(**next(x['receipt'] for x in objects if x['kind']=='acceptance'))
    terminal=next(x for x in objects if x['kind']=='terminal')
    witness=ReceiptStatusWitness(**terminal['witness']);head=SignedTreeHead(**terminal['checkpoint'])
    assert verify_returned_batch(returned,witness,head,contents,7,key.public_key())
    assert not verify_returned_batch(returned,witness,head,(b'a',b'changed',b'c'),7,key.public_key())
    assert not verify_returned_batch(returned,witness,head,tuple(reversed(contents)),7,key.public_key())
    assert engine.engine.issued_frames==3 and len(objects)==2

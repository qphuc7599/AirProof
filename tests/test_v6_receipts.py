import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature
from airproof.v6_receipts import fragment,Receiver,ReceiptReturnQueue
from airproof.v6_resources import ContactBudget


def test_actual_signed_multifragment_return_and_shared_raw_budget():
    key=Ed25519PrivateKey.generate(); receiver=Receiver(key.public_key())
    q=ReceiptReturnQueue(key);payload=b'evidence'*100
    assert q.issue(payload,3,0,5)
    assert not q.contact(3,0,ContactBudget(),receiver)
    used=ContactBudget();assert used.spend('raw',bytes(512))
    assert not q.contact(3,1,used,receiver)
    assert not q.delivered
    for epoch in (2,3):
        budget=ContactBudget();assert q.contact(3,epoch,budget,receiver)
        assert budget.used==528 and budget.release_used==0
    assert list(q.delivered.values())==[3] and q.occupancy==0
    assert next(iter(receiver.completed.values()))==payload
    assert q.transmitted_bytes==1024 and q.control_bytes==32


def test_tamper_expiry_buffer_and_no_reservation_borrow():
    key=Ed25519PrivateKey.generate();r=Receiver(key.public_key())
    f=fragment(b'abc',key)[0]
    with pytest.raises(InvalidSignature):r.accept(f[:-1]+bytes([f[-1]^1]))
    q=ReceiptReturnQueue(key,512)
    assert not q.issue(bytes(800),1,0,2)
    assert q.issue(b'ok',1,0,2)
    assert not q.contact(1,1,ContactBudget(capacity=512),r)
    assert not q.contact(1,3,ContactBudget(),r)
    assert len(q.expired)==1 and not q.delivered

def test_queue_reservation_is_consumed_by_actual_frames():
    key=Ed25519PrivateKey.generate();q=ReceiptReturnQueue(key,2048)
    assert q.reserve('promise',1024)
    assert q.committed_occupancy==1024
    assert q.issue(b'a',1,0,2,reservation='promise')
    assert q.occupancy==512 and q.reserved_bytes==512 and q.committed_occupancy==1024
    assert q.release_reservation('promise')==512
    assert q.committed_occupancy==512

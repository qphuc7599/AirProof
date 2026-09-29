from dataclasses import replace
import hashlib
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from airproof.audit import ReceiptAccountabilityLog, verify_receipt
from scripts.benchmark_v5_receipt_wire import concrete_raw_record, issue_wire, reverse_accounting


def test_unchanged_receipt_roundtrip_binds_concrete_raw_identity():
    key=Ed25519PrivateKey.generate()
    record=concrete_raw_record()
    before=record.public_dict()
    receipt,content,encoded,packet=issue_wire(record,ReceiptAccountabilityLog(key),key,b"a"*32)
    assert len(content)==278 and len(encoded)==280 and len(packet)==512
    assert receipt.content_hash==hashlib.sha256(content).hexdigest()
    assert record.public_dict()==before
    assert not verify_receipt(replace(receipt,content_hash="0"*64),key.public_key())


def test_envelope_overflow_fails_instead_of_changing_receipt():
    key=Ed25519PrivateKey.generate()
    with pytest.raises(ValueError,match="capacity"):
        issue_wire(concrete_raw_record(),ReceiptAccountabilityLog(key),key,b"a"*32,
                   accepted_at=10**150,inclusion_deadline=10**150+24)


def test_reverse_direction_bound_and_forward_ack_gate():
    best=reverse_accounting()
    assert best["reverse_total_bytes"]==592 and best["reverse_byte_fit"]
    assert best["conservative_fit_even_with_additional_384_reserved_bytes"]
    acknowledged=reverse_accounting(require_receipt_ack=True)
    assert acknowledged["forward_control_worst_case_bytes"]==144
    assert not acknowledged["unchanged_forward_control_cap_pass"]
    assert not reverse_accounting(receipt_count=2)["reverse_byte_fit"]

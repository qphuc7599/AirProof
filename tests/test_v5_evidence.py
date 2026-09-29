from dataclasses import asdict, replace
import hashlib
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from airproof.audit import ReceiptAccountabilityLog, verify_receipt
from airproof.records import Observation, canonical_json
from airproof.v5_evidence import build_run_evidence, attach_gateway_receipt
from airproof.v5_wire_payload import encode_observation_payload


def record():
    return Observation(9,20,3,1,25.,2.,1.,512,"raw-identity",None,None)


def test_run_provenance_identity_snapshots_and_changes_with_clock():
    metrics={"rmse":1.,"support":{"n":12}}
    a=build_run_evidence(*["a"*64]*4,"AP",915004,"frozen-live",metrics,{"duplicates":0})
    metrics["support"]["n"]=999
    assert a.metrics["support"]["n"]==12
    assert len(a.evidence_id)==64
    assert a.evidence_id != replace(a,clock="reconstructed").evidence_id
    with pytest.raises(ValueError): build_run_evidence(*["bad"]*4,"AP",1,"live",{}, {})
    with pytest.raises(ValueError): build_run_evidence(*["a"*64]*4,"AP",1,"live",{"rmse":float("nan")}, {})


def test_gateway_attachment_binds_identity_without_claiming_delivery():
    key=Ed25519PrivateKey.generate()
    log=ReceiptAccountabilityLog(key)
    r=record(); before=asdict(r)
    attached=attach_gateway_receipt(r,24,key,receipt_log=log,encryption_key=b"a"*32)
    assert attached.nullifier==r.nullifier
    assert attached.content_hash==hashlib.sha256(canonical_json(r.public_dict())).hexdigest()
    assert attached.receipt_id==attached.receipt.receipt_id
    assert attached.issue_clock==24 and attached.packet_bytes==512
    assert verify_receipt(attached.receipt,key.public_key())
    assert not attached.receipt_delivery_confirmed and not attached.origin_delivery_confirmed
    assert attached.receipt_delivery_clock is None and attached.origin_delivery_clock is None
    assert asdict(r)==before
    next_receipt=attach_gateway_receipt(replace(r,nullifier="second"),25,key,receipt_log=log)
    assert next_receipt.receipt.receive_sequence==2
    assert attached.content_hash != next_receipt.content_hash


def test_false_time_and_oversized_receipt_are_not_accepted():
    key=Ed25519PrivateKey.generate()
    with pytest.raises(ValueError): attach_gateway_receipt(record(),19,key)
    with pytest.raises(ValueError,match="capacity"):
        attach_gateway_receipt(record(),10**150,key)


def test_actual_source_bytes_bound_and_mismatch_before_issue():
    key=Ed25519PrivateKey.generate()
    log=ReceiptAccountabilityLog(key)
    original=replace(record(),nullifier="a"*64,value=25.123456789123)
    payload=encode_observation_payload(original)
    assert len(payload)==84
    for change in ({"nullifier":"b"*64},{"user_id":10},{"epoch":19},{"cell":4},
                   {"group":2},{"value":25.},{"sigma":3.},{"quality":.5}):
        with pytest.raises(ValueError,match="does not match"):
            attach_gateway_receipt(replace(original,**change),24,key,receipt_log=log,accepted_payload=payload)
    attached=attach_gateway_receipt(original,24,key,receipt_log=log,accepted_payload=payload)
    assert attached.receipt.receive_sequence==1  # all mismatches rejected before issue
    assert attached.content_encoding=="raw-source-binary"
    assert attached.canonical_content_bytes==84
    assert attached.content_hash==hashlib.sha256(payload).hexdigest()
    assert verify_receipt(attached.receipt,key.public_key())
    assert attached.content_hash != hashlib.sha256(canonical_json(original.public_dict())).hexdigest()
    # Unencoded simulator arrival/attack metadata must not change source-byte identity.
    metadata_changed=replace(original,direct_arrival=999,relay_arrival=888,corrupted=True)
    second=attach_gateway_receipt(metadata_changed,24,key,receipt_log=log,accepted_payload=payload)
    assert second.content_hash==attached.content_hash
    assert not second.receipt_delivery_confirmed and not second.origin_delivery_confirmed

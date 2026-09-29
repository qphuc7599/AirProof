import json
from dataclasses import asdict

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airproof.audit import TransparencyLog
from airproof.records import canonical_json
from airproof.v6_public_checkpoint import PublicCheckpointPublisher
from airproof.v6_receipts import FRAME_SIZE,fragment


def _cfg(epochs=(0,4,8,12),ttl=8):
    return {'uplink_epochs':list(epochs),'capacity_bytes_per_direction':1024,
        'release_reserved_bytes':256,'control_budget_bytes':128,'queue_buffer_bytes':4096,
        'auditor_reassembly_buffer_bytes':4096,'expiry_epochs':ttl}


def test_checkpoint_counts_only_after_finite_reassembly_and_exact_accounting():
    key=Ed25519PrivateKey.generate();tree=TransparencyLog(key);tree.append(b'a')
    publisher=PublicCheckpointPublisher(key,tree,_cfg())
    publisher.tick(0)
    assert publisher.publicly_checkpointed==0 and publisher.queue.pending
    for epoch in range(1,13):publisher.tick(epoch)
    assert publisher.publicly_checkpointed==1 and publisher.accepted_heads
    assert publisher.queue.transmitted_bytes%FRAME_SIZE==0
    assert publisher.queue.control_bytes==publisher.queue.transmitted_bytes//FRAME_SIZE*16
    assert publisher.queue.peak_buffer<=4096 and publisher.auditor.peak_buffer<=4096


def test_missing_and_delayed_uplink_never_imply_public_checkpoint_success():
    key=Ed25519PrivateKey.generate();tree=TransparencyLog(key);tree.append(b'a')
    missing=PublicCheckpointPublisher(key,tree,_cfg(epochs=()))
    for epoch in range(20):missing.tick(epoch)
    assert missing.publicly_checkpointed==0 and missing.queue.transmitted_bytes==0
    delayed=PublicCheckpointPublisher(key,tree,_cfg(epochs=(0,12),ttl=4))
    for epoch in range(13):delayed.tick(epoch)
    assert delayed.publicly_checkpointed==0
    assert len(delayed.queue.expired)==1 and delayed.queue.pending


def test_insufficient_publication_queue_buffer_drops_without_counting_head():
    key=Ed25519PrivateKey.generate();tree=TransparencyLog(key);tree.append(b'a')
    cfg=_cfg(epochs=(0,4));cfg['queue_buffer_bytes']=256
    publisher=PublicCheckpointPublisher(key,tree,cfg)
    publisher.tick(0);publisher.tick(4)
    assert publisher.publicly_checkpointed==0 and publisher.queue.dropped
    assert publisher.queue.transmitted_bytes==publisher.queue.control_bytes==0


def test_tamper_rollback_and_split_view_are_counted_only_when_compared():
    key=Ed25519PrivateKey.generate();tree=TransparencyLog(key);tree.append(b'a')
    publisher=PublicCheckpointPublisher(key,tree,_cfg())
    first=publisher._payload(0);assert publisher.process_payload(first)
    tree.append(b'b');second=publisher._payload(1);assert publisher.process_payload(second)
    obj=json.loads(second);obj['checkpoint']['root_hex']='0'*64
    assert not publisher.process_payload(canonical_json(obj))
    assert publisher.signature_failures==1 and publisher.publicly_checkpointed==2
    assert not publisher.process_payload(first) and publisher.rollback_detections==1
    alternate=TransparencyLog(key);alternate.append(b'x');alternate.append(b'y')
    split={'kind':'public_checkpoint','previous_size':2,
        'previous_root_hex':publisher.last_head.root_hex,'consistency_proof_b64':[],
        'checkpoint':asdict(alternate.checkpoint(timestamp_ms=2))}
    assert not publisher.process_payload(canonical_json(split))
    assert publisher.split_view_detections==1 and publisher.publicly_checkpointed==2


def test_authenticated_public_frame_tamper_fails_before_reassembly():
    key=Ed25519PrivateKey.generate();tree=TransparencyLog(key);tree.append(b'a')
    publisher=PublicCheckpointPublisher(key,tree,_cfg());frame=bytearray(fragment(publisher._payload(0),key)[0])
    frame[50]^=1
    with pytest.raises(InvalidSignature):publisher.auditor.accept(bytes(frame))

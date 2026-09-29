import hashlib
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airproof.audit import (
    ReceiptAccountabilityLog,
    ReplicatedObjectStore,
    TransparencyLog,
    verify_consistency,
    verify_inclusion,
    verify_receipt,
    verify_receipt_inclusion,
    verify_status_witness,
    verify_tree_head,
)


def payload(index):
    return hashlib.sha256(str(index).encode()).digest()


def test_transparency_log_proofs_for_balanced_and_unbalanced_sizes():
    key = Ed25519PrivateKey.generate()
    for size in range(2, 18):
        log = TransparencyLog(key)
        leaves = [payload(index) for index in range(size)]
        for leaf in leaves:
            log.append(leaf)
        assert verify_tree_head(log.checkpoint(timestamp_ms=1), key.public_key())
        for index, leaf in enumerate(leaves):
            assert verify_inclusion(
                leaf,
                index=index,
                tree_size=size,
                proof=log.inclusion_proof(index),
                expected_root=log.root(),
            )
        for old_size in range(1, size):
            assert verify_consistency(
                old_size=old_size,
                new_size=size,
                old_root=log.root(old_size),
                new_root=log.root(),
                proof=log.consistency_proof(old_size),
            )


def test_receipt_accountability_and_single_terminal_status():
    key = Ed25519PrivateKey.generate()
    log = ReceiptAccountabilityLog(key)
    receipt = log.issue(b"ciphertext", policy_id="p1", accepted_at=10, inclusion_deadline=20)
    assert verify_receipt(receipt, key.public_key())
    assert receipt.receipt_id in log.audit(now=21)["unresolved"]
    transparency = TransparencyLog(key)
    leaf_index = transparency.append(b"ciphertext")
    witness = log.resolve(
        receipt.receipt_id,
        status="INCLUDED",
        witnessed_at=19,
        batch_id="b1",
        leaf_index=leaf_index,
        tree_size=transparency.size,
        root_hex=transparency.root().hex(),
        inclusion_proof=transparency.inclusion_proof(leaf_index),
    )
    assert verify_status_witness(witness, key.public_key())
    assert verify_receipt_inclusion(receipt, witness, b"ciphertext", key.public_key())
    assert log.audit(now=21)["resolution_rate"] == 1.0
    with pytest.raises(ValueError):
        log.resolve(
            receipt.receipt_id,
            status="EXCLUDED",
            witnessed_at=19,
            reason_code="duplicate",
        )


def test_receipt_inclusion_tamper_and_key_revocation_are_rejected():
    key = Ed25519PrivateKey.generate()
    accountability = ReceiptAccountabilityLog(key)
    content = b"ciphertext"
    receipt = accountability.issue(content, policy_id="p1", accepted_at=10, inclusion_deadline=20)
    transparency = TransparencyLog(key)
    index = transparency.append(content)
    witness = accountability.resolve(
        receipt.receipt_id,
        status="INCLUDED",
        witnessed_at=19,
        batch_id="b1",
        leaf_index=index,
        tree_size=transparency.size,
        root_hex=transparency.root().hex(),
        inclusion_proof=transparency.inclusion_proof(index),
    )
    assert not verify_receipt(replace(receipt, content_hash="00" * 32), key.public_key())
    assert not verify_receipt(receipt, key.public_key(), revoked_effective_at=10)
    assert not verify_receipt_inclusion(receipt, witness, b"different", key.public_key())
    assert not verify_status_witness(replace(witness, batch_id="fork"), key.public_key())


def test_replica_deletion_is_availability_not_integrity():
    store = ReplicatedObjectStore(3)
    digest = store.put(b"object")
    store.delete_replica(digest, 0)
    store.delete_replica(digest, 1)
    assert store.get(digest) == b"object"
    assert store.available_replicas(digest) == 1
    store.delete_replica(digest, 2)
    assert store.get(digest) is None

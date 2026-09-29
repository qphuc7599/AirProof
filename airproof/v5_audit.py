"""Paired, local cryptographic audit experiment; logical clocks are not chain latency.

All policies receive identical accepted payloads and signed acceptance receipts.
The anchor is an immutable local registry with an explicitly assumed independent
read path. It neither recovers lost payloads nor proves consensus finality.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .audit import (EpochAnchorRegistry, ReceiptAccountabilityLog, TransparencyLog,
                    verify_consistency, verify_inclusion, verify_receipt, verify_tree_head)
from .records import canonical_json

POLICIES = ("signed_log", "transparency_log", "anchored_transparency_log")
FAULTS = ("content_tamper", "receipt_omission", "rollback", "equivocation",
          "auditor_partition", "checkpoint_loss", "payload_loss", "replay")
HORIZON = 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoded_size(value) -> int:
    return len(canonical_json(value))


def _flat_head(payloads, key, batch_id):
    body = {"batch_id": batch_id, "tree_size": len(payloads),
            "root_hex": hashlib.sha256(b"".join(payloads)).hexdigest(), "timestamp_ms": 10}
    return {**body, "signature_b64": base64.b64encode(key.sign(canonical_json(body))).decode()}


def _verify_flat(head, payloads, public_key):
    try:
        body = {k: v for k, v in head.items() if k != "signature_b64"}
        public_key.verify(base64.b64decode(head["signature_b64"]), canonical_json(body))
        return head["tree_size"] == len(payloads) and head["root_hex"] == hashlib.sha256(b"".join(payloads)).hexdigest()
    except (InvalidSignature, ValueError, TypeError, KeyError):
        # Verification treats malformed/untrusted responses as rejection.
        return False


def run_same_stream_audit(*, batch_sizes=(32, 128, 512, 2048), trials=100):
    if trials < 1 or any(size < 4 for size in batch_sizes):
        raise ValueError("positive trials and batch sizes >=4 required")
    rows = []
    for size in batch_sizes:
        for trial in range(trials):
            key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f"v5-audit:{size}:{trial}".encode()).digest())
            public_key = key.public_key()
            payloads = tuple(hashlib.sha256(f"v5-record:{size}:{trial}:{i}".encode()).digest() for i in range(size))
            stream_hash = hashlib.sha256(b"".join(payloads)).hexdigest()
            log_id = f"v5-{size}-{trial}"
            receipt_log = ReceiptAccountabilityLog(key)
            receipts = tuple(receipt_log.issue(p, policy_id="v5-common", accepted_at=0, inclusion_deadline=20) for p in payloads)
            if not all(verify_receipt(r, public_key) for r in receipts):
                raise AssertionError("invalid authentic receipt")
            receipt_bytes = sum(_encoded_size(asdict(r)) for r in receipts)
            started = time.perf_counter_ns()
            log = TransparencyLog(key, log_id=log_id)
            for payload in payloads:
                log.append(payload)
            head = log.checkpoint(timestamp_ms=10)
            prefix = size // 2
            old_root = log.root(prefix)
            consistency = log.consistency_proof(prefix)
            index = size // 2
            proof = log.inclusion_proof(index)
            build_ns = time.perf_counter_ns() - started
            flat = _flat_head(payloads, key, log_id)
            registry = EpochAnchorRegistry()
            registry.anchor(city=log_id, epoch=0, version=0, root_hex=head.root_hex)
            anchor_wire = {"city": log_id, "epoch": 0, "version": 0, "root_hex": head.root_hex}
            # A second validly signed same-size history: compromised log signer,
            # not an invalid-signature shortcut for detecting equivocation.
            fork_payloads = list(payloads)
            fork_payloads[index] = hashlib.sha256(payloads[index] + b"fork").digest()
            fork = TransparencyLog(key, log_id=log_id)
            for payload in fork_payloads:
                fork.append(payload)
            fork_head = fork.checkpoint(timestamp_ms=10)
            fork_flat = _flat_head(fork_payloads, key, log_id)
            if not verify_tree_head(fork_head, public_key):
                raise AssertionError("fork must be validly signed")
            for policy in POLICIES:
                merkle = policy != "signed_log"
                anchored = policy == "anchored_transparency_log"
                verification_started = time.perf_counter_ns()
                valid = (verify_tree_head(head, public_key) and verify_inclusion(
                    payloads[index], index=index, tree_size=size, proof=proof,
                    expected_root=bytes.fromhex(head.root_hex)) and verify_consistency(
                    old_size=prefix, new_size=size, old_root=old_root,
                    new_root=bytes.fromhex(head.root_hex), proof=consistency)) if merkle else _verify_flat(flat, payloads, public_key)
                verify_ns = time.perf_counter_ns() - verification_started
                if not valid:
                    raise AssertionError("clean control rejected")
                head_bytes = _encoded_size(asdict(head) if merkle else flat)
                proof_bytes = sum(len(p) for p in proof + consistency) if merkle else 0
                # Exact bytes of this explicit local canonical-JSON/binary protocol.
                # A signed flat log sends the full batch to establish membership.
                response_bytes = (len(payloads[index]) + proof_bytes if merkle else sum(map(len, payloads))) + head_bytes
                for fault in FAULTS:
                    detected_at = None
                    reason = "unobserved-by-horizon"
                    extra_bytes = 0
                    check_started = time.perf_counter_ns()
                    if fault == "content_tamper":
                        damaged = payloads[index] + b"x"
                        tampered = list(payloads); tampered[index] = damaged
                        rejected = (not verify_inclusion(damaged, index=index, tree_size=size, proof=proof,
                                    expected_root=bytes.fromhex(head.root_hex))) if merkle else not _verify_flat(flat, tampered, public_key)
                        if rejected:
                            detected_at, reason = 11, "cryptographic-content-rejection"
                    elif fault == "receipt_omission":
                        # Independently held signed acceptance receipt; no status or
                        # inclusion response arrives by its promised deadline.
                        if verify_receipt(receipts[index], public_key):
                            detected_at, reason = 21, "acceptance-obligation-timeout-not-cryptographic-noninclusion"
                    elif fault == "rollback":
                        old_payloads = payloads[:prefix]
                        if merkle:
                            old_log = TransparencyLog(key, log_id=log_id)
                            for payload in old_payloads: old_log.append(payload)
                            old_head = old_log.checkpoint(timestamp_ms=10)
                            rejected = verify_tree_head(old_head, public_key) and old_head.tree_size < head.tree_size
                            extra_bytes = _encoded_size(asdict(old_head))
                        else:
                            old_head = _flat_head(old_payloads, key, log_id)
                            rejected = _verify_flat(old_head, old_payloads, public_key) and old_head["tree_size"] < flat["tree_size"]
                            extra_bytes = _encoded_size(old_head)
                        if rejected: detected_at, reason = 11, "rollback-against-retained-checkpoint"
                    elif fault in ("equivocation", "auditor_partition", "checkpoint_loss"):
                        conflict = (verify_tree_head(fork_head, public_key) and fork_head.root_hex != head.root_hex) if merkle else (
                            _verify_flat(fork_flat, fork_payloads, public_key) and fork_flat["root_hex"] != flat["root_hex"])
                        if anchored:
                            # Honest checkpoint already anchored; external anchor
                            # channel remains available during log/gossip partition.
                            conflict = conflict and registry.root(city=log_id, epoch=0, version=0) != fork_head.root_hex
                            if conflict: detected_at, reason = 12, "fork-disagrees-with-independent-immutable-anchor"
                            extra_bytes = _encoded_size(anchor_wire)
                        elif fault == "equivocation" and conflict:
                            detected_at, reason = 18, "two-valid-conflicting-checkpoints-after-gossip"
                            extra_bytes = _encoded_size(asdict(fork_head) if merkle else fork_flat)
                    elif fault == "payload_loss":
                        detected_at, reason = 21, "retrieval-timeout-no-payload-recovery"
                    elif fault == "replay":
                        duplicate_receipts = [receipts[index], receipts[index]]
                        ids = [r.receipt_id for r in duplicate_receipts if verify_receipt(r, public_key)]
                        if len(ids) != len(set(ids)):
                            detected_at, reason = 11, "receipt-id-duplicate-rejected-by-common-validator"
                        extra_bytes = _encoded_size(asdict(receipts[index]))
                    rows.append({"batch_size": size, "trial": trial, "policy": policy, "fault": fault,
                        "stream_sha256": stream_hash, "detected": detected_at is not None,
                        "detection_epoch": detected_at, "time_from_fault": None if detected_at is None else detected_at - 10,
                        "right_censored": detected_at is None, "restricted_detection_time": HORIZON - 10 if detected_at is None else detected_at - 10,
                        "horizon_epoch": HORIZON, "reason": reason, "clean_control_valid": bool(valid),
                        "payload_bytes": sum(map(len, payloads)), "receipt_bytes": receipt_bytes,
                        "head_bytes": head_bytes, "proof_bytes": proof_bytes,
                        "anchor_write_bytes": _encoded_size(anchor_wire) if anchored else 0,
                        "audit_response_bytes": response_bytes + extra_bytes,
                        "total_protocol_bytes": sum(map(len, payloads)) + receipt_bytes + head_bytes + response_bytes + extra_bytes + (_encoded_size(anchor_wire) if anchored else 0),
                        "primitive_build_ns": build_ns if merkle else None, "clean_verify_ns": verify_ns,
                        "fault_check_ns": time.perf_counter_ns() - check_started,
                        "clock_scope": "deterministic-logical-contact-clock-not-measured-chain-latency"})
    return rows


def summarize_audit(rows):
    groups = {}
    for row in rows:
        key = (row["batch_size"], row["policy"], row["fault"])
        groups.setdefault(key, []).append(row)
    return [{"batch_size": size, "policy": policy, "fault": fault,
             "trials": len(items), "detected": sum(r["detected"] for r in items),
             "right_censored": sum(r["right_censored"] for r in items),
             "restricted_mean_detection_time": sum(r["restricted_detection_time"] for r in items) / len(items),
             "mean_total_protocol_bytes": sum(r["total_protocol_bytes"] for r in items) / len(items),
             "mean_audit_response_bytes": sum(r["audit_response_bytes"] for r in items) / len(items)}
            for (size, policy, fault), items in sorted(groups.items())]

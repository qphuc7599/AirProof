from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .audit import (
    EpochAnchorRegistry,
    ReceiptAccountabilityLog,
    TransparencyLog,
    verify_consistency,
    verify_inclusion,
    verify_receipt,
    verify_receipt_inclusion,
    verify_status_witness,
    verify_tree_head,
)
from .integrity import MerkleTree, verify_proof


def _payload(batch_size: int, index: int, replicate: int) -> bytes:
    return hashlib.sha256(f"airproof-e6:{batch_size}:{index}:{replicate}".encode()).digest()


def run_audit_benchmark(
    *,
    batch_sizes: tuple[int, ...] = (32, 128, 512, 2048),
    replicates: int = 5,
) -> list[dict[str, Any]]:
    """Benchmark local commitment primitives; this is not an Ethereum latency result."""
    if replicates < 2 or any(size < 2 for size in batch_sizes):
        raise ValueError("need at least two replicates and batch sizes of at least two")
    rows: list[dict[str, Any]] = []
    signing_key = Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(b"airproof-e6-deterministic-local-key").digest()
    )
    public_key = signing_key.public_key()
    for batch_size in batch_sizes:
        for replicate in range(replicates):
            payloads = tuple(_payload(batch_size, index, replicate) for index in range(batch_size))

            started = time.perf_counter_ns()
            tree = MerkleTree(payloads)
            build_ns = time.perf_counter_ns() - started
            index = batch_size // 2
            committed_payload = tree.payloads[index]
            proof = tree.proof(index)
            verify_started = time.perf_counter_ns()
            valid = verify_proof(committed_payload, proof, tree.root)
            verify_ns = time.perf_counter_ns() - verify_started
            rows.append(
                {
                    "study": "E6",
                    "method": "sorted_batch_merkle_local",
                    "batch_size": batch_size,
                    "replicate": replicate,
                    "append_ns_per_record": build_ns / batch_size,
                    "proof_bytes": len(proof.siblings) * 33,
                    "verify_ns": verify_ns,
                    "proof_valid": valid,
                    "consistency_proof_bytes": None,
                    "consistency_valid": None,
                    "scope": "local-cryptographic-primitive-not-ethereum",
                }
            )

            log = TransparencyLog(signing_key, log_id=f"airproof-e6-{batch_size}-{replicate}")
            prefix = batch_size // 2
            append_started = time.perf_counter_ns()
            for payload in payloads:
                log.append(payload)
            append_ns = time.perf_counter_ns() - append_started
            head = log.checkpoint(timestamp_ms=replicate)
            inclusion = log.inclusion_proof(index)
            old_root = log.root(prefix)
            consistency = log.consistency_proof(prefix)
            verify_started = time.perf_counter_ns()
            inclusion_valid = verify_inclusion(
                payloads[index],
                index=index,
                tree_size=batch_size,
                proof=inclusion,
                expected_root=log.root(),
            )
            consistency_valid = verify_consistency(
                old_size=prefix,
                new_size=batch_size,
                old_root=old_root,
                new_root=log.root(),
                proof=consistency,
            )
            head_valid = verify_tree_head(head, public_key)
            verify_ns = time.perf_counter_ns() - verify_started
            rows.append(
                {
                    "study": "E6",
                    "method": "rfc9162_style_transparency_log",
                    "batch_size": batch_size,
                    "replicate": replicate,
                    "append_ns_per_record": append_ns / batch_size,
                    "proof_bytes": len(inclusion) * 32,
                    "verify_ns": verify_ns,
                    "proof_valid": inclusion_valid and head_valid,
                    "consistency_proof_bytes": len(consistency) * 32,
                    "consistency_valid": consistency_valid,
                    "scope": "local-append-only-log",
                }
            )
    return rows


def run_availability_benchmark(
    *,
    replicas: tuple[int, ...] = (1, 2, 3),
    loss_probabilities: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    trials: int = 20_000,
    seed: int = 20260901,
) -> list[dict[str, Any]]:
    """Measure retrieval under independent loss and a correlated partition mixture."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replica_count in replicas:
        for loss_probability in loss_probabilities:
            if not 0 <= loss_probability <= 1:
                raise ValueError("loss probabilities must lie in [0, 1]")
            independent = rng.random((trials, replica_count)) >= loss_probability
            independent_success = np.any(independent, axis=1)
            correlated_partition = rng.random(trials) < 0.1
            mixed_success = independent_success & ~correlated_partition
            for scenario, success in (
                ("independent_replica_loss", independent_success),
                ("ten_percent_correlated_partition", mixed_success),
            ):
                rate = float(np.mean(success))
                standard_error = float(np.sqrt(rate * (1.0 - rate) / trials))
                rows.append(
                    {
                        "study": "E7",
                        "scenario": scenario,
                        "replicas": replica_count,
                        "loss_probability": loss_probability,
                        "trials": trials,
                        "retrieval_success": rate,
                        "ci95_low": max(0.0, rate - 1.96 * standard_error),
                        "ci95_high": min(1.0, rate + 1.96 * standard_error),
                        "replication_overhead": replica_count,
                    }
                )
    return rows


def run_receipt_fault_benchmark(*, replicates: int = 100) -> list[dict[str, Any]]:
    """Inject post-acceptance omission, replay, proof, fork and revocation faults."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    signing_key = Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(b"airproof-e6-receipt-fault-key").digest()
    )
    public_key = signing_key.public_key()
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        content = hashlib.sha256(f"receipt-content-{replicate}".encode()).digest()

        omitted_log = ReceiptAccountabilityLog(signing_key)
        omitted = omitted_log.issue(
            content,
            policy_id="policy-v3",
            accepted_at=10,
            inclusion_deadline=20,
        )
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "receipt_issued_then_omitted",
                "replicate": replicate,
                "detected": omitted.receipt_id in omitted_log.audit(now=21)["unresolved"],
            }
        )

        included_log = ReceiptAccountabilityLog(signing_key)
        included = included_log.issue(
            content,
            policy_id="policy-v3",
            accepted_at=10,
            inclusion_deadline=20,
        )
        transparency = TransparencyLog(signing_key, log_id=f"receipt-{replicate}")
        leaf_index = transparency.append(content)
        witness = included_log.resolve(
            included.receipt_id,
            status="INCLUDED",
            witnessed_at=19,
            batch_id=f"batch-{replicate}",
            leaf_index=leaf_index,
            tree_size=transparency.size,
            root_hex=transparency.root().hex(),
            inclusion_proof=transparency.inclusion_proof(leaf_index),
        )
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "valid_inclusion",
                "replicate": replicate,
                "detected": verify_receipt_inclusion(
                    included,
                    witness,
                    content,
                    public_key,
                ),
            }
        )
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "tampered_inclusion_content",
                "replicate": replicate,
                "detected": not verify_receipt_inclusion(
                    included,
                    witness,
                    content + b"x",
                    public_key,
                ),
            }
        )
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "forged_terminal_witness",
                "replicate": replicate,
                "detected": not verify_status_witness(
                    replace(witness, batch_id="competing-batch"),
                    public_key,
                ),
            }
        )
        replay_detected = False
        try:
            included_log.resolve(
                included.receipt_id,
                status="EXCLUDED",
                witnessed_at=19,
                reason_code="replay",
            )
        except ValueError:
            replay_detected = True
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "terminal_status_replay",
                "replicate": replicate,
                "detected": replay_detected,
            }
        )

        excluded_log = ReceiptAccountabilityLog(signing_key)
        excluded = excluded_log.issue(
            content,
            policy_id="policy-v3",
            accepted_at=10,
            inclusion_deadline=20,
        )
        excluded_witness = excluded_log.resolve(
            excluded.receipt_id,
            status="EXCLUDED",
            witnessed_at=19,
            reason_code="policy-ineligible",
        )
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "valid_explicit_exclusion",
                "replicate": replicate,
                "detected": verify_status_witness(excluded_witness, public_key),
            }
        )

        late_log = ReceiptAccountabilityLog(signing_key)
        late = late_log.issue(
            content,
            policy_id="policy-v3",
            accepted_at=10,
            inclusion_deadline=20,
        )
        late_transparency = TransparencyLog(signing_key)
        late_index = late_transparency.append(content)
        late_log.resolve(
            late.receipt_id,
            status="INCLUDED",
            witnessed_at=21,
            batch_id=f"late-{replicate}",
            leaf_index=late_index,
            tree_size=late_transparency.size,
            root_hex=late_transparency.root().hex(),
            inclusion_proof=late_transparency.inclusion_proof(late_index),
        )
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "late_inclusion",
                "replicate": replicate,
                "detected": late.receipt_id in late_log.audit(now=22)["late"],
            }
        )
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "revoked_signer",
                "replicate": replicate,
                "detected": not verify_receipt(
                    included,
                    public_key,
                    revoked_effective_at=10,
                ),
            }
        )

        registry = EpochAnchorRegistry()
        registry.anchor(city="beijing", epoch=replicate, version=0, root_hex="11" * 32)
        fork_detected = False
        try:
            registry.anchor(city="beijing", epoch=replicate, version=0, root_hex="22" * 32)
        except ValueError:
            fork_detected = True
        rows.append(
            {
                "study": "E6-receipt-fault",
                "fault": "competing_correction_root",
                "replicate": replicate,
                "detected": fork_detected,
            }
        )
    return rows


def write_benchmark(rows: list[dict[str, Any]], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    return output

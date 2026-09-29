from __future__ import annotations

import pytest

from airproof.audit import verify_consistency, verify_inclusion
from airproof.records import Observation
from airproof.v6_transport import TransportTrace
from airproof.v7_shared_execution import IncrementalAppendLog, integrated_transport_reviewer


class _Key:
    def sign(self, payload):
        return b"x" * 64


def test_incremental_append_proofs_verify_for_every_prefix():
    log = IncrementalAppendLog(_Key())
    for index in range(257):
        payload = f"leaf-{index}".encode()
        observed, proof = log.append_with_proof(payload)
        assert observed == index
        assert verify_inclusion(
            payload,
            index=index,
            tree_size=index + 1,
            proof=proof,
            expected_root=log.root(),
        )


def test_incremental_append_can_opt_in_to_public_consistency_proofs():
    log = IncrementalAppendLog(_Key(), retain_leaves=True)
    payloads = [f"leaf-{index}".encode() for index in range(37)]
    roots = {}
    for index, payload in enumerate(payloads, start=1):
        log.append_with_proof(payload)
        if index in {7, 19, 37}:
            roots[index] = log.root()
    for old_size, new_size in ((7, 19), (7, 37), (19, 37)):
        proof = log.consistency_proof(old_size, new_size)
        # The incremental log exposes only the current root, so reconstruct the
        # requested new prefix with a second log to exercise the generic API.
        prefix = IncrementalAppendLog(_Key(), retain_leaves=True)
        for payload in payloads[:new_size]:
            prefix.append_with_proof(payload)
        assert verify_consistency(
            old_size=old_size,
            new_size=new_size,
            old_root=roots[old_size],
            new_root=roots[new_size],
            proof=proof,
        )


def _record(user: int, group: int) -> Observation:
    return Observation(
        user,
        0,
        group,
        group,
        10.0,
        1.0,
        1.0,
        512,
        f"n-{user}",
        0,
        0,
    )


def _trace() -> TransportTrace:
    return TransportTrace(
        seed=1,
        nodes=4,
        acquisition_epochs=2,
        drain_epochs=1,
        node_groups=(0, 0, 1, 1),
        gateway_contacts=((0, 1, 2, 3), (), (0, 1, 2, 3)),
        peer_contacts=((), (), ()),
        trace_hash="fixture",
    )


def test_v4_policy_runs_inside_shared_transport_and_records_policy():
    config = {
        "world": {"groups": 2},
        "scheduler": {
            "algorithm": "v4_outcome_effective_hard_if_feasible",
            "budget_bytes_per_epoch": 2048,
            "target_contributors": 1,
            "reserve_fraction": 0.25,
            "fairness_strength": 1.0,
            "constraint_mode": "hard_if_feasible",
        },
        "transport": {"useful_lag_epochs": 1},
    }
    result, collector = integrated_transport_reviewer(
        _trace(), [_record(i, i // 2) for i in range(4)], config
    )
    assert result.metrics["allocation_policy"] == (
        "v4_outcome_effective_hard_if_feasible"
    )
    assert collector.allocations[0]["constraint_satisfied"]
    assert collector.allocations[0]["avoidable"] == {0: 0.0, 1: 0.0}
    assert result.metrics["token_violations"] == 0
    assert result.metrics["receipt_issued"] == len(collector.selected)
    assert result.metrics["raw_candidates_accepted"] == 4


def test_unknown_allocation_policy_fails_closed():
    with pytest.raises(ValueError, match="unknown reviewer allocation policy"):
        integrated_transport_reviewer(
            _trace(), [], {"scheduler": {"algorithm": "unknown"}}
        )


def test_incremental_selected_audit_can_publish_a_finite_checkpoint():
    config = {
        "world": {"groups": 2},
        "scheduler": {
            "algorithm": "v4_outcome_effective_hard_if_feasible",
            "budget_bytes_per_epoch": 2048,
            "target_contributors": 1,
            "reserve_fraction": 0.25,
            "fairness_strength": 1.0,
            "constraint_mode": "hard_if_feasible",
        },
        "transport": {
            "useful_lag_epochs": 1,
            "raw_packet_bytes": 512,
            "release_packet_bytes": 256,
            "capacity_bytes_per_direction": 1024,
            "release_gateway_reservation_bytes": 256,
            "control_budget_bytes": 128,
            "raw_buffer_bytes_per_node": 2048,
            "release_buffer_bytes_per_node": 1024,
            "ttl_epochs": 2,
            "copy_tokens": 1,
        },
        "audit": {
            "return_mode": "incremental_selected_receipts_v1",
            "max_records": 100,
            "max_content_bytes": 2048,
            "durable_log_capacity_bytes": 1_000_000,
            "collector_return_buffer_bytes": 4096,
            "inclusion_deadline_epochs": 2,
            "max_epoch": 100,
            "public_checkpoint": {
                "enabled": True,
                "uplink_epochs": [1, 2],
                "capacity_bytes_per_direction": 1024,
                "release_reserved_bytes": 256,
                "control_budget_bytes": 128,
                "queue_buffer_bytes": 4096,
                "auditor_reassembly_buffer_bytes": 4096,
                "expiry_epochs": 4,
            },
        },
    }
    result, _ = integrated_transport_reviewer(
        _trace(), [_record(i, i // 2) for i in range(4)], config
    )
    assert result.metrics["audit_public_checkpoint_heads_verified"] >= 1
    assert result.metrics["audit_publicly_checkpointed"] == result.metrics["receipt_issued"]
    assert result.metrics["audit_public_checkpoint_raw_bytes"] > 0

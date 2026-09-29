import pytest

from airproof.records import Observation
from airproof.v6_baseline_adapters import load_adapter_specs, run_baseline_adapter
from airproof.v6_transport import TransportTrace


def record(user=0, epoch=0, suffix=""):
    return Observation(user, epoch, 0, user % 2, 10.0, 1.0, 1.0, 512,
                       f"{user}:{epoch}:{suffix}", 0, 0)


def trace(nodes=2, acquisition=1, drain=3):
    horizon = acquisition + drain
    return TransportTrace(7, nodes, acquisition, drain, tuple(i % 2 for i in range(nodes)),
                          tuple((tuple(range(nodes)) if e == 0 else ()) for e in range(horizon)),
                          tuple(() for _ in range(horizon)), "shared-trace")


def test_registry_forbids_external_reproduction_claims():
    specs = load_adapter_specs()
    assert set(specs) == {
        "centralized_v6_reference", "privacy_only_v6_reference",
        "blockchain_dt_v6_reference", "opportunistic_ttl_v6_reference",
    }
    assert all(not spec.external_reproduction_claim_allowed for spec in specs.values())


@pytest.mark.parametrize("adapter_id", ["centralized_v6_reference", "opportunistic_ttl_v6_reference"])
def test_raw_adapters_share_trace_capacity_and_clock(adapter_id):
    records = [record(0), record(1)]
    run = run_baseline_adapter(adapter_id, trace(), records, {})
    assert run.transport.metrics["trace_hash"] == "shared-trace"
    assert set(run.selected_nullifiers) == {r.nullifier for r in records}
    assert run.transport.metrics["max_contact_direction_bytes"] <= 1024
    assert run.transport.metrics["max_control_direction_bytes"] <= 128
    assert run.transport.metrics["raw_generated"] == 2
    assert run.transport.metrics["release_generated"] == 0


def test_privacy_only_has_no_raw_wire_or_estimator_interface():
    records = [record(0), record(1)]
    run = run_baseline_adapter("privacy_only_v6_reference", trace(), records, {})
    assert run.selected_nullifiers == ()
    assert run.transport.raw_arrivals == {}
    assert run.transport.metrics["raw_generated"] == 0
    assert run.transport.metrics["raw_payload_bytes"] == 0
    assert run.transport.metrics["release_generated"] == 2
    assert run.transport.metrics["release_delivered"] == 2


def test_blockchain_adapter_exposes_only_audit_admitted_selection():
    cfg = {"audit": {"max_return_frames": 0},
           "world": {"groups": 2},
           "scheduler": {"budget_bytes_per_epoch": 83200, "target_contributors": 30}}
    run = run_baseline_adapter("blockchain_dt_v6_reference", trace(), [record(0)], cfg)
    assert run.selected_nullifiers == ()
    assert run.transport.metrics["raw_delivered"] == 1
    assert run.transport.metrics["audit_admission_rejected"] == 1
    assert run.transport.metrics["receipt_issued"] == 0

"""Run the pre-registered bounded IntegratedCollector accountability matrix."""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from airproof.audit import IngestionReceipt, ReceiptStatusWitness, SignedTreeHead
from airproof.records import Observation
from airproof.v6_audit_return import verify_returned_bundle
from airproof.v6_experiment import integrated_transport
from airproof.v6_transport import TransportTrace


ROOT = Path(__file__).resolve().parents[1]


def _records(case):
    if case["pattern"] == "serial":
        return [Observation(0, epoch, 0, epoch % 2, 10.0 + epoch, 1.0, 1.0,
                            512, f'{case["id"]}:{epoch}', None, None)
                for epoch in range(case["records"])]
    return [Observation(index % case["nodes"], 0, 0, index % 2, 10.0 + index,
                        1.0, 1.0, 512, f'{case["id"]}:{index}', None, None)
            for index in range(case["records"])]


def _trace(case):
    horizon = case["acquisition_epochs"] + case["drain_epochs"]
    all_nodes = tuple(range(case["nodes"]))
    if case["contacts"] == "all":
        contacts = (all_nodes,) * horizon
    elif case["contacts"] == "initial_only":
        contacts = (all_nodes,) + ((),) * (horizon - 1)
    elif case["contacts"] == "initial_and_late":
        contacts = (all_nodes,) + ((),) * (horizon - 2) + (all_nodes,)
    else:
        raise ValueError("unknown registered contact schedule")
    return TransportTrace(6606001, case["nodes"], case["acquisition_epochs"],
                          case["drain_epochs"], tuple(i % 2 for i in range(case["nodes"])),
                          contacts, ((),) * horizon, case["id"])


def _returned_objects(collector):
    objects = []
    for origin, receiver in collector.receivers.items():
        for identity, payload in receiver.completed.items():
            objects.append((origin, collector.returns.delivered[identity.hex()], json.loads(payload)))
    return objects


def run_case(case):
    records = _records(case)
    config = {"world": {"groups": 2},
              "scheduler": {"budget_bytes_per_epoch": 83200, "target_contributors": 1},
              "audit": case["audit"]}
    result, collector = integrated_transport(_trace(case), records, config)
    receipt_times = {}
    terminal_times = {}
    receipts = {}
    terminals = {}
    for _, delivered, obj in _returned_objects(collector):
        if obj["kind"] == "acceptance":
            receipt = IngestionReceipt(**obj["receipt"])
            receipts[receipt.receipt_id] = receipt
            receipt_times[receipt.receipt_id] = delivered
        else:
            witness = ReceiptStatusWitness(**obj["witness"])
            terminals[witness.receipt_id] = (witness, SignedTreeHead(**obj["checkpoint"]))
            terminal_times[witness.receipt_id] = delivered
    complete = sorted(receipts.keys() & terminals.keys())
    latencies = [max(receipt_times[rid], terminal_times[rid]) - receipts[rid].accepted_at
                 for rid in complete]
    metrics = result.metrics
    row = {
        "case": case["id"], "attempted_records": len(records),
        "physical_delivered": metrics["raw_delivered"],
        "accountability_rejected": metrics["audit_admission_rejected"],
        "admitted": metrics["receipt_issued"], "allocated": len(collector.selected),
        "collector_included": metrics["audit_collector_log"]["resolved_on_time"],
        "origin_verified": metrics["audit_verified_return_pairs"],
        "raw_bytes": metrics["raw_payload_bytes"],
        "return_bytes": metrics["receipt_payload_bytes"],
        "return_frames": metrics["receipt_payload_bytes"] // 512,
        "return_control_bytes": metrics["audit_return_control_bytes"],
        "total_control_bytes": metrics["control_bytes"],
        "expired_return_objects": metrics["receipt_expired"],
        "pending_return_objects": metrics["receipt_pending"],
        "post_admission_queue_drops": metrics["receipt_buffer_dropped"],
        "verified_pair_latencies_epochs": latencies,
        "max_verified_pair_latency_epochs": max(latencies) if latencies else None,
        "max_contact_direction_bytes": metrics["max_contact_direction_bytes"],
        "max_control_direction_bytes": metrics["max_control_direction_bytes"],
        "peak_committed_return_buffer_bytes": metrics["audit_peak_committed_buffer_bytes"],
        "raw_remaining_copies": metrics["raw_remaining_copies"],
        "token_violations": metrics["token_violations"]
    }
    return row, collector, records


def fault_checks(collector, records):
    objects = [item[2] for item in _returned_objects(collector)]
    receipt = IngestionReceipt(**next(obj["receipt"] for obj in objects if obj["kind"] == "acceptance"))
    terminal = next(obj for obj in objects if obj["kind"] == "terminal")
    witness = ReceiptStatusWitness(**terminal["witness"])
    head = SignedTreeHead(**terminal["checkpoint"])
    content = collector.contents[receipt.receipt_id]
    public_key = collector.key.public_key()
    bad_signature = base64.b64encode(bytes(64)).decode()
    checks = {
        "baseline_bundle_verified": verify_returned_bundle(receipt, witness, head, content, public_key),
        "content_mutation_detected": not verify_returned_bundle(receipt, witness, head, content + b'x', public_key),
        "checkpoint_root_mutation_detected": not verify_returned_bundle(
            receipt, witness, replace(head, root_hex='00' * 32), content, public_key),
        "receipt_signature_mutation_detected": not verify_returned_bundle(
            replace(receipt, signature_b64=bad_signature), witness, head, content, public_key)
    }
    try:
        integrated_transport(_trace({"id": "duplicate", "records": 2, "pattern": "burst",
            "nodes": 1, "acquisition_epochs": 1, "drain_epochs": 0, "contacts": "all",
            "audit": {}}), [records[0], records[0]], {"world": {"groups": 2}})
    except ValueError as exc:
        checks["duplicate_nullifier_detected"] = "nullifiers must be unique" in str(exc)
    else:
        checks["duplicate_nullifier_detected"] = False
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path,
                        default=ROOT / "configs/v6_accountability_confirmation.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/v6/accountability_integrated_confirmation")
    args = parser.parse_args()
    registration_bytes = args.registration.read_bytes()
    registration = json.loads(registration_bytes)
    if registration["registration"] != "v6-integrated-accountability-confirmation-v1":
        raise ValueError("unexpected registration")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    adequate_collector = adequate_records = None
    for case in registration["matrix"]:
        row, collector, records = run_case(case)
        rows.append(row)
        if case["id"] == "adequate_single":
            adequate_collector, adequate_records = collector, records
    faults = fault_checks(adequate_collector, adequate_records)
    sources = [args.registration, Path(__file__), ROOT / "airproof/v6_experiment.py",
               ROOT / "airproof/v6_transport.py", ROOT / "airproof/v6_audit_return.py",
               ROOT / "airproof/v6_receipts.py", ROOT / "airproof/audit.py"]
    manifest = {
        "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "source_hashes": {str(path.relative_to(ROOT)).replace('\\', '/'):
                          hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "historical_comparator": registration["historical_comparator"],
        "claim_limit": registration["claim_limit"],
        "case_count": len(rows), "all_fault_checks_passed": all(faults.values())
    }
    (args.output / "registration.json").write_bytes(registration_bytes)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "fault_checks.json").write_text(json.dumps(faults, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": rows, "fault_checks": faults, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()

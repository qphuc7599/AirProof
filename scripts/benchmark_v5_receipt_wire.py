"""Concrete acceptance-receipt codec and conditional reverse-link accounting only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from airproof.audit import ReceiptAccountabilityLog, IngestionReceipt, verify_receipt
from airproof.records import Observation, canonical_json
from airproof.v5_transport import encode_packet, decode_packet
from airproof.v5_wire_control import encode_header, encode_ack, AckStatus


def concrete_raw_record():
    # Concrete main-scale domain example, not a claimed persisted campaign packet.
    return Observation(999, 671, 1023, 3, 25.123456789, 2., 1., 512,
        hashlib.sha256(b"airproof-v5-concrete-raw").hexdigest(), None, None)


def issue_wire(record, log, signing_key, encryption_key, *, accepted_at=695, inclusion_deadline=719):
    content = canonical_json(record.public_dict())
    receipt = log.issue(content, policy_id="airproof-v5", accepted_at=accepted_at,
                        inclusion_deadline=inclusion_deadline)
    serialized = canonical_json(asdict(receipt))
    if not verify_receipt(receipt, signing_key.public_key()):
        raise AssertionError("authentic receipt signature rejected")
    # No compression, omitted fields, shortened signature, or protocol changes.
    packet = encode_packet(serialized, kind="raw", encryption_key=encryption_key, signing_key=signing_key)
    kind, decoded = decode_packet(packet, encryption_key=encryption_key, verification_key=signing_key.public_key())
    reconstructed = IngestionReceipt(**json.loads(decoded))
    if kind != "raw" or decoded != serialized or not verify_receipt(reconstructed, signing_key.public_key()):
        raise AssertionError("receipt round trip failed")
    if reconstructed.content_hash != hashlib.sha256(content).hexdigest():
        raise AssertionError("wrong concrete raw content identity")
    return receipt, content, serialized, packet


def reverse_accounting(*, direction_capacity=1024, existing_ack_count=4, receipt_count=1, require_receipt_ack=False):
    """Sufficient byte bounds; does not claim the emulator implements a reverse queue."""
    ack_bytes = len(encode_ack(1, AckStatus.DELIVERED))
    header_bytes = len(encode_header(1, packet_kind="raw"))
    existing_reverse_acks = existing_ack_count*ack_bytes
    reverse_bytes = existing_reverse_acks+receipt_count*(512+header_bytes)
    forward_extra = receipt_count*ack_bytes if require_receipt_ack else 0
    # Existing forward control can already use all 128 bytes: release32 + raw3*32.
    return {"capacity_per_direction": direction_capacity, "existing_reverse_ack_bytes": existing_reverse_acks,
        "receipt_packet_bytes": receipt_count*512, "receipt_header_bytes": receipt_count*header_bytes,
        "reverse_total_bytes": reverse_bytes, "reverse_byte_fit": reverse_bytes <= direction_capacity,
        "conservative_fit_even_with_additional_384_reserved_bytes": reverse_bytes+384 <= direction_capacity,
        "forward_extra_receipt_ack_bytes": forward_extra,
        "forward_control_worst_case_bytes": 128+forward_extra,
        "unchanged_forward_control_cap_pass": 128+forward_extra <=128,
        "receipt_return_reliability": "unacknowledged-best-effort" if not require_receipt_ack else "requires-new-forward-ACK-capacity",
        "integrated_transport_gate": "not-implemented-no-directional-receipt-queue-or-return-to-origin-proof"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/v5/receipt_wire")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    key = Ed25519PrivateKey.generate()
    log = ReceiptAccountabilityLog(key)
    record = concrete_raw_record()
    receipt, content, encoded, packet = issue_wire(record, log, key, bytes(range(32)))
    # Exercise the concrete raw candidate in the exact same unchanged codec too.
    raw_packet = encode_packet(content, kind="raw", encryption_key=bytes(range(32)), signing_key=key)
    assert decode_packet(raw_packet, encryption_key=bytes(range(32)), verification_key=key.public_key())[1] == content
    # Width-only upper bound under the finite main domain. Signatures always base64 encode64 bytes.
    maximum_main_serialized_bytes = len(encoded)+(len(str(1000*672))-len(str(receipt.receive_sequence)))
    paths = [ROOT/"airproof/audit.py", ROOT/"airproof/records.py", ROOT/"airproof/v5_transport.py", ROOT/"airproof/v5_wire_control.py", Path(__file__)]
    result = {"role": "local-wire-prototype-not-integrated-receipt-delivery",
        "raw_encoding": "canonical_json(Observation.public_dict()) concrete main-scale domain example; emulator has no committed production raw serializer",
        "raw_content_bytes": len(content), "raw_packet_bytes": len(raw_packet), "raw_payload_sha256": hashlib.sha256(content).hexdigest(),
        "receipt_canonical_json_bytes": len(encoded), "receipt_packet_bytes": len(packet), "codec_payload_capacity":413,
        "unchanged_receipt_protocol": True, "policy_id": receipt.policy_id, "receipt_signature_verified":True,
        "maximum_main_receipt_bytes_width_bound":maximum_main_serialized_bytes,
        "width_bound_domain":{"sequence_max":672000,"accepted_at_max":695,"inclusion_deadline_max":719},
        "one_receipt_best_effort": reverse_accounting(), "one_receipt_acknowledged": reverse_accounting(require_receipt_ack=True),
        "two_receipts_same_reverse_contact": reverse_accounting(receipt_count=2),
        "proposed_run_evidence_attachment":{"nullifier":record.nullifier,"content_hash":receipt.content_hash,
            "receipt_id":receipt.receipt_id,"receive_sequence":receipt.receive_sequence,"accepted_at":receipt.accepted_at,
            "inclusion_deadline":receipt.inclusion_deadline,"receipt":asdict(receipt),
            "wire_bytes":len(packet),"receipt_returned_to_origin":False,"receipt_delivery_clock":None},
        "source_sha256":{str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "no_public_transactions":True,
        "limitations":["transport ACK is link control under authenticated-session assumption, not signed acceptance",
            "first gateway uploader may be a relay, not original contributor; reverse packet to relay is not receipt delivery to origin",
            "gateway contact list and accounting do not presently implement an independent reverse receipt direction",
            "best-effort reverse feasibility assumes independent direction capacity and no other reverse payload; it is a sufficient byte bound only",
            "a receipt acknowledgement adds16 forward control bytes and can violate unchanged128-byte control cap",
            "one first raw gateway delivery per1024-byte contact follows from640-byte raw lane; this bound changes with capacity",
            "receipt issue is not terminal inclusion/status witness or blockchain anchoring; their costs are additional"]}
    (args.output/"analysis.json").write_text(json.dumps(result,indent=2))
    (args.output/"raw_content.json").write_bytes(content)
    (args.output/"receipt.json").write_bytes(encoded)
    print(json.dumps({k:result[k] for k in ("raw_content_bytes","raw_packet_bytes","receipt_canonical_json_bytes","receipt_packet_bytes","maximum_main_receipt_bytes_width_bound","one_receipt_best_effort","one_receipt_acknowledged")},indent=2))


if __name__=="__main__": main()

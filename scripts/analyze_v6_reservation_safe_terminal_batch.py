"""Registered serialization analysis for reservation-safe terminal batching."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airproof.audit import ReceiptAccountabilityLog, TransparencyLog
from airproof.records import canonical_json
from airproof.v6_audit_return import verify_returned_bundle
from airproof.v6_receipts import CHUNK, FRAME_SIZE


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v6_terminal_batch_design.json"
OUT = ROOT / "reports" / "v6" / "reservation_safe_terminal_batch_design"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(payload: bytes) -> int:
    return math.ceil(len(payload) / CHUNK)


def terminal_envelope(count: int, cfg: dict) -> bytes:
    """Conservative batch payload under the registration's numeric bounds."""
    signature = base64.b64encode(bytes(64)).decode()
    proof = base64.b64encode(bytes(32)).decode()
    proof_depth = math.ceil(math.log2(cfg["max_records"]))
    witness = {
        "receipt_id": "f" * 64,
        "status": "INCLUDED",
        "witnessed_at": cfg["max_epoch"],
        "batch_id": str(cfg["max_records"]),
        "leaf_index": cfg["max_records"] - 1,
        "tree_size": cfg["max_records"],
        "root_hex": "f" * 64,
        "inclusion_proof_b64": [proof] * proof_depth,
        "reason_code": None,
        "signature_b64": signature,
    }
    checkpoint = {
        "log_id": "airproof-log-v1",
        "tree_size": cfg["max_records"],
        "root_hex": "f" * 64,
        "timestamp_ms": cfg["max_epoch"] * 3_600_000,
        "signature_b64": signature,
    }
    return canonical_json({
        "kind": "terminal_batch",
        "origin": cfg["max_origin"],
        "witnesses": [witness] * count,
        "checkpoint": checkpoint,
    })


def analyze(count: int, cfg: dict) -> dict:
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
    log = ReceiptAccountabilityLog(key)
    tree = TransparencyLog(key)
    origin = 7
    contents = [f"same-origin-record-{i}".encode() for i in range(count)]
    receipts = []
    acceptance_payloads = []
    for content in contents:
        receipt = log.issue(content, policy_id="v6-terminal-component", accepted_at=0,
                            inclusion_deadline=24)
        receipts.append(receipt)
        acceptance_payloads.append(canonical_json({
            "kind": "acceptance", "origin": origin, "receipt": asdict(receipt)
        }))
        tree.append(content)
    checkpoint = tree.checkpoint(timestamp_ms=0)
    witnesses = []
    individual = []
    for index, receipt in enumerate(receipts):
        witness = log.resolve(
            receipt.receipt_id, status="INCLUDED", witnessed_at=0,
            batch_id=str(tree.size), leaf_index=index, tree_size=tree.size,
            root_hex=tree.root().hex(), inclusion_proof=tree.inclusion_proof(index),
        )
        witnesses.append(witness)
        individual.append(canonical_json({
            "kind": "terminal", "origin": origin,
            "witness": asdict(witness), "checkpoint": asdict(checkpoint),
        }))
    batched = canonical_json({
        "kind": "terminal_batch", "origin": origin,
        "witnesses": [asdict(w) for w in witnesses],
        "checkpoint": asdict(checkpoint),
    })
    verified = all(verify_returned_bundle(r, w, checkpoint, c, key.public_key())
                   for r, w, c in zip(receipts, witnesses, contents))
    mutation_rejected = not verify_returned_bundle(
        receipts[0], witnesses[0], checkpoint, contents[0] + b"-mutated", key.public_key()
    )
    reorder_rejected = count == 1 or not all(
        verify_returned_bundle(r, w, checkpoint, c, key.public_key())
        for r, w, c in zip(receipts, witnesses, reversed(contents))
    )
    acceptance_frames = sum(frames(p) for p in acceptance_payloads)
    individual_terminal_frames = sum(frames(p) for p in individual)
    batch_terminal_frames = frames(batched)
    envelope_frames = frames(terminal_envelope(count, cfg))
    prior_envelope_frames = 0 if count == 1 else frames(terminal_envelope(count - 1, cfg))
    incremental_terminal_reservation = envelope_frames - prior_envelope_frames
    return {
        "batch_size": count,
        "acceptance_frames": acceptance_frames,
        "individual_terminal_frames": individual_terminal_frames,
        "batched_terminal_frames": batch_terminal_frames,
        "terminal_frames_saved": individual_terminal_frames - batch_terminal_frames,
        "current_total_return_frames": acceptance_frames + individual_terminal_frames,
        "batched_total_return_frames": acceptance_frames + batch_terminal_frames,
        "batched_total_raw_bytes": (acceptance_frames + batch_terminal_frames) * FRAME_SIZE,
        "batched_total_control_bytes": (acceptance_frames + batch_terminal_frames) * cfg["return_ack_bytes"],
        "terminal_envelope_frames": envelope_frames,
        "incremental_terminal_reservation_frames": incremental_terminal_reservation,
        "envelope_covers_serialization": envelope_frames >= batch_terminal_frames,
        "all_original_pairs_verify": verified,
        "mutation_rejected": mutation_rejected,
        "reorder_rejected": reorder_rejected,
    }


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    rows = [analyze(n, cfg) for n in cfg["same_origin_batch_sizes"]]
    checks = {
        "all_original_pairs_verify": all(r["all_original_pairs_verify"] for r in rows),
        "all_mutations_rejected": all(r["mutation_rejected"] for r in rows),
        "all_reorders_rejected_when_applicable": all(r["reorder_rejected"] for r in rows),
        "all_envelopes_cover_serialization": all(r["envelope_covers_serialization"] for r in rows),
        "one_record_no_worse": rows[0]["batched_total_return_frames"] <= rows[0]["current_total_return_frames"],
    }
    result = {
        "registration": cfg["registration"],
        "role": cfg["role"],
        "serializer": {"frame_bytes": FRAME_SIZE, "payload_chunk_bytes": CHUNK,
                       "return_ack_bytes": cfg["return_ack_bytes"]},
        "rows": rows,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "interpretation": (
            "Terminal batching shares a checkpoint and frame slack, but acceptance receipts remain per record. "
            "The envelope is an admission bound for the registered maxima, not implemented queue behavior."
        ),
        "smoke_context": {
            "legacy_selected": 181,
            "first_per_record_accountability_selected": 39,
            "stale_gate_selected": 50,
        },
        "scope_limits": cfg["scope_limits"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "registration_sha256": sha256(CONFIG),
        "source_hashes": {
            "configs/v6_terminal_batch_design.json": sha256(CONFIG),
            "scripts/analyze_v6_reservation_safe_terminal_batch.py": sha256(Path(__file__)),
            "airproof/audit.py": sha256(ROOT / "airproof" / "audit.py"),
            "airproof/v6_receipts.py": sha256(ROOT / "airproof" / "v6_receipts.py"),
        },
        "all_checks_passed": result["all_checks_passed"],
        "scope_limits": cfg["scope_limits"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

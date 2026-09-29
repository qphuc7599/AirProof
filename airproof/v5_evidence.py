"""Content-bound run provenance and issued receipts; no receipt delivery is inferred."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .audit import IngestionReceipt, ReceiptAccountabilityLog, verify_receipt
from .records import Observation, canonical_json
from .v5_transport import encode_packet, decode_packet
from .v5_wire_payload import decode_raw_payload


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _json_snapshot(value):
    # Reject non-finite values and detach from mutable caller-owned mappings.
    return json.loads(canonical_json(value))


@dataclass(frozen=True)
class GatewayReceiptAttachment:
    nullifier: str
    content_hash: str
    receipt_id: str
    issue_clock: int
    inclusion_deadline: int
    receipt: IngestionReceipt
    canonical_content_bytes: int
    canonical_receipt_bytes: int
    packet_bytes: int
    packet_sha256: str
    signature_verified: bool
    codec_roundtrip_verified: bool
    content_encoding: str = "legacy-canonical-observation-json"
    receipt_delivery_confirmed: bool = False
    origin_delivery_confirmed: bool = False
    receipt_delivery_clock: int | None = None
    origin_delivery_clock: int | None = None
    scope: str = "issued-local-acceptance-receipt-not-network-return-or-terminal-inclusion"


@dataclass(frozen=True)
class RunEvidence:
    source_hash: str
    config_hash: str
    data_hash: str
    trace_hash: str
    method: str
    seed: int
    clock: str
    metrics: dict[str, Any]
    violations: dict[str, Any]
    gateway_receipts: tuple[GatewayReceiptAttachment, ...] = ()

    @property
    def evidence_id(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()


def build_run_evidence(source_hash: str, config_hash: str, data_hash: str, trace_hash: str,
                       method: str, seed: int, clock: str, metrics: Mapping, violations: Mapping) -> RunEvidence:
    """Build a JSON-safe snapshot. Hashes identify supplied artifacts, not attest their truth."""
    hashes = [_sha256(v, n) for v,n in ((source_hash,"source_hash"),(config_hash,"config_hash"),
                                       (data_hash,"data_hash"),(trace_hash,"trace_hash"))]
    if not isinstance(method,str) or not method or not isinstance(clock,str) or not clock:
        raise ValueError("explicit method and scoring clock required")
    if isinstance(seed,bool) or not isinstance(seed,int) or seed < 0:
        raise ValueError("nonnegative integer seed required")
    if not isinstance(metrics,Mapping) or not isinstance(violations,Mapping):
        raise ValueError("metrics and violations must be mappings")
    return RunEvidence(*hashes,method,seed,clock,_json_snapshot(dict(metrics)),_json_snapshot(dict(violations)))


def attach_gateway_receipt(observation: Observation, accepted_at: int, key: Ed25519PrivateKey,
                           *, receipt_log: ReceiptAccountabilityLog | None = None,
                           inclusion_deadline: int | None = None,
                           encryption_key: bytes | None = None,
                           accepted_payload: bytes | None = None) -> GatewayReceiptAttachment:
    """Issue and verify locally using unchanged receipt/codec; does not transmit bytes.

    A shared receipt_log preserves a gateway's sequence across calls. If omitted,
    the local one-receipt prototype starts a new log; its sequence is not global.
    Omitted encryption_key is an ephemeral local codec check, not a delivery key.
    The caller supplies a genuine first-gateway-acceptance event; this function
    cannot infer it from the observation's historical arrival metadata.
    """
    if isinstance(accepted_at,bool) or not isinstance(accepted_at,int) or accepted_at < observation.epoch:
        raise ValueError("acceptance must be an integer at or after acquisition")
    deadline = accepted_at+24 if inclusion_deadline is None else inclusion_deadline
    if isinstance(deadline,bool) or not isinstance(deadline,int) or deadline < accepted_at:
        raise ValueError("invalid inclusion deadline")
    encoding = "legacy-canonical-observation-json"
    if accepted_payload is None:
        content = canonical_json(observation.public_dict())
    else:
        decoded_source = decode_raw_payload(accepted_payload)
        expected = (observation.nullifier, observation.user_id, observation.epoch,
                    observation.cell, observation.group, observation.value,
                    observation.sigma, observation.quality)
        actual = (decoded_source.nullifier, decoded_source.user_id, decoded_source.acquisition_epoch,
                  decoded_source.cell, decoded_source.group, decoded_source.value,
                  decoded_source.sigma, decoded_source.quality)
        if actual != expected:
            raise ValueError("accepted source payload does not match immutable observation")
        content = accepted_payload
        encoding = "raw-source-binary"
    log = ReceiptAccountabilityLog(key) if receipt_log is None else receipt_log
    receipt = log.issue(content,policy_id="airproof-v5",accepted_at=accepted_at,inclusion_deadline=deadline)
    if not verify_receipt(receipt,key.public_key()) or receipt.content_hash != hashlib.sha256(content).hexdigest():
        raise ValueError("receipt signing key or content identity mismatch")
    serialized = canonical_json(asdict(receipt))
    encryption_key = os.urandom(32) if encryption_key is None else encryption_key
    packet = encode_packet(serialized,kind="raw",encryption_key=encryption_key,signing_key=key)
    kind, decoded = decode_packet(packet,encryption_key=encryption_key,verification_key=key.public_key())
    recovered = IngestionReceipt(**json.loads(decoded))
    if kind != "raw" or decoded != serialized or recovered != receipt or not verify_receipt(recovered,key.public_key()):
        raise AssertionError("receipt codec/signature round trip failed")
    return GatewayReceiptAttachment(observation.nullifier,receipt.content_hash,receipt.receipt_id,
        accepted_at,deadline,receipt,len(content),len(serialized),len(packet),hashlib.sha256(packet).hexdigest(),True,True,
        content_encoding=encoding)

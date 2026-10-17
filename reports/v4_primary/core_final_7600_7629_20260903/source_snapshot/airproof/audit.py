from __future__ import annotations

import base64
import hashlib
import threading
import time
from dataclasses import asdict, dataclass
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .records import canonical_json


def _leaf_hash(payload: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + payload).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _largest_power_of_two_less_than(value: int) -> int:
    if value <= 1:
        raise ValueError("value must exceed one")
    return 1 << ((value - 1).bit_length() - 1)


def _tree_hash(leaves: tuple[bytes, ...]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return _leaf_hash(leaves[0])
    split = _largest_power_of_two_less_than(len(leaves))
    return _node_hash(_tree_hash(leaves[:split]), _tree_hash(leaves[split:]))


def _audit_path(index: int, leaves: tuple[bytes, ...]) -> list[bytes]:
    if len(leaves) == 1:
        return []
    split = _largest_power_of_two_less_than(len(leaves))
    if index < split:
        return _audit_path(index, leaves[:split]) + [_tree_hash(leaves[split:])]
    return _audit_path(index - split, leaves[split:]) + [_tree_hash(leaves[:split])]


def verify_inclusion(
    payload: bytes,
    *,
    index: int,
    tree_size: int,
    proof: tuple[bytes, ...],
    expected_root: bytes,
) -> bool:
    """Verify an RFC 9162-style Merkle inclusion proof."""
    if tree_size <= 0 or not 0 <= index < tree_size:
        return False
    node = _leaf_hash(payload)
    fn = index
    sn = tree_size - 1
    for sibling in proof:
        if fn & 1 or fn == sn:
            node = _node_hash(sibling, node)
            while fn and not fn & 1:
                fn >>= 1
                sn >>= 1
        else:
            node = _node_hash(node, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and node == expected_root


def _subproof(old_size: int, leaves: tuple[bytes, ...], complete: bool) -> list[bytes]:
    if old_size == len(leaves):
        return [] if complete else [_tree_hash(leaves)]
    split = _largest_power_of_two_less_than(len(leaves))
    if old_size <= split:
        return _subproof(old_size, leaves[:split], complete) + [_tree_hash(leaves[split:])]
    return _subproof(old_size - split, leaves[split:], False) + [_tree_hash(leaves[:split])]


def verify_consistency(
    *,
    old_size: int,
    new_size: int,
    old_root: bytes,
    new_root: bytes,
    proof: tuple[bytes, ...],
) -> bool:
    """Verify that the old tree is an append-only prefix of the new tree."""
    if old_size < 0 or new_size < old_size:
        return False
    if old_size == 0:
        return not proof
    if old_size == new_size:
        return not proof and old_root == new_root
    fn = old_size - 1
    sn = new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    if fn == 0:
        first = old_root
        cursor = 0
    elif proof:
        first = proof[0]
        cursor = 1
    else:
        return False
    old_hash = first
    new_hash = first
    for sibling in proof[cursor:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            old_hash = _node_hash(sibling, old_hash)
            new_hash = _node_hash(sibling, new_hash)
            while fn and not fn & 1:
                fn >>= 1
                sn >>= 1
        else:
            new_hash = _node_hash(new_hash, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and old_hash == old_root and new_hash == new_root


@dataclass(frozen=True)
class SignedTreeHead:
    log_id: str
    tree_size: int
    root_hex: str
    timestamp_ms: int
    signature_b64: str

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "log_id": self.log_id,
            "tree_size": self.tree_size,
            "root_hex": self.root_hex,
            "timestamp_ms": self.timestamp_ms,
        }


class TransparencyLog:
    """Append-only Merkle log with signed heads, inclusion and consistency proofs."""

    def __init__(self, signing_key: Ed25519PrivateKey, *, log_id: str = "airproof-log-v1"):
        self._signing_key = signing_key
        self.log_id = log_id
        self._leaves: list[bytes] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._leaves)

    def append(self, payload: bytes) -> int:
        with self._lock:
            self._leaves.append(bytes(payload))
            return len(self._leaves) - 1

    def root(self, tree_size: int | None = None) -> bytes:
        size = self.size if tree_size is None else tree_size
        if not 0 <= size <= self.size:
            raise ValueError("invalid tree size")
        return _tree_hash(tuple(self._leaves[:size]))

    def inclusion_proof(self, index: int, tree_size: int | None = None) -> tuple[bytes, ...]:
        size = self.size if tree_size is None else tree_size
        if not 0 <= index < size <= self.size:
            raise ValueError("invalid index or tree size")
        return tuple(_audit_path(index, tuple(self._leaves[:size])))

    def consistency_proof(self, old_size: int, new_size: int | None = None) -> tuple[bytes, ...]:
        size = self.size if new_size is None else new_size
        if not 0 <= old_size <= size <= self.size:
            raise ValueError("invalid consistency range")
        if old_size in (0, size):
            return ()
        return tuple(_subproof(old_size, tuple(self._leaves[:size]), True))

    def checkpoint(self, *, timestamp_ms: int | None = None) -> SignedTreeHead:
        payload = {
            "log_id": self.log_id,
            "tree_size": self.size,
            "root_hex": self.root().hex(),
            "timestamp_ms": int(time.time() * 1000) if timestamp_ms is None else timestamp_ms,
        }
        signature = self._signing_key.sign(canonical_json(payload))
        return SignedTreeHead(**payload, signature_b64=base64.b64encode(signature).decode())


def verify_tree_head(head: SignedTreeHead, public_key: Ed25519PublicKey) -> bool:
    try:
        public_key.verify(base64.b64decode(head.signature_b64), canonical_json(head.unsigned_payload()))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


@dataclass(frozen=True)
class IngestionReceipt:
    content_hash: str
    policy_id: str
    receive_sequence: int
    accepted_at: int
    inclusion_deadline: int
    signature_b64: str

    def unsigned_payload(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("signature_b64")
        return data

    @property
    def receipt_id(self) -> str:
        return hashlib.sha256(canonical_json(self.unsigned_payload())).hexdigest()


@dataclass(frozen=True)
class ReceiptStatusWitness:
    receipt_id: str
    status: Literal["INCLUDED", "EXCLUDED"]
    witnessed_at: int
    batch_id: str | None
    leaf_index: int | None
    tree_size: int | None
    root_hex: str | None
    inclusion_proof_b64: tuple[str, ...] | None
    reason_code: str | None
    signature_b64: str

    def unsigned_payload(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("signature_b64")
        return data


class ReceiptAccountabilityLog:
    """Signed acceptance receipts with exactly one terminal status witness."""

    def __init__(self, signing_key: Ed25519PrivateKey):
        self._key = signing_key
        self._receipts: dict[str, IngestionReceipt] = {}
        self._statuses: dict[str, ReceiptStatusWitness] = {}
        self._sequence = 0
        self._lock = threading.Lock()

    def issue(
        self,
        content: bytes,
        *,
        policy_id: str,
        accepted_at: int,
        inclusion_deadline: int,
    ) -> IngestionReceipt:
        if inclusion_deadline < accepted_at:
            raise ValueError("inclusion deadline precedes acceptance")
        with self._lock:
            self._sequence += 1
            payload = {
                "content_hash": hashlib.sha256(content).hexdigest(),
                "policy_id": policy_id,
                "receive_sequence": self._sequence,
                "accepted_at": accepted_at,
                "inclusion_deadline": inclusion_deadline,
            }
            signature = self._key.sign(canonical_json(payload))
            receipt = IngestionReceipt(**payload, signature_b64=base64.b64encode(signature).decode())
            self._receipts[receipt.receipt_id] = receipt
            return receipt

    def resolve(
        self,
        receipt_id: str,
        *,
        status: Literal["INCLUDED", "EXCLUDED"],
        witnessed_at: int,
        batch_id: str | None = None,
        leaf_index: int | None = None,
        tree_size: int | None = None,
        root_hex: str | None = None,
        inclusion_proof: tuple[bytes, ...] | None = None,
        reason_code: str | None = None,
    ) -> ReceiptStatusWitness:
        with self._lock:
            if receipt_id not in self._receipts:
                raise KeyError(receipt_id)
            if receipt_id in self._statuses:
                raise ValueError("receipt already has a terminal status")
            if status == "INCLUDED" and (
                batch_id is None
                or leaf_index is None
                or tree_size is None
                or root_hex is None
                or inclusion_proof is None
            ):
                raise ValueError("included status requires a complete inclusion witness")
            if status == "EXCLUDED" and not reason_code:
                raise ValueError("excluded status requires reason_code")
            encoded_proof = (
                tuple(base64.b64encode(item).decode() for item in inclusion_proof)
                if inclusion_proof is not None
                else None
            )
            payload = {
                "receipt_id": receipt_id,
                "status": status,
                "witnessed_at": witnessed_at,
                "batch_id": batch_id,
                "leaf_index": leaf_index,
                "tree_size": tree_size,
                "root_hex": root_hex,
                "inclusion_proof_b64": encoded_proof,
                "reason_code": reason_code,
            }
            signature = self._key.sign(canonical_json(payload))
            witness = ReceiptStatusWitness(
                **payload, signature_b64=base64.b64encode(signature).decode()
            )
            self._statuses[receipt_id] = witness
            return witness

    def audit(self, *, now: int) -> dict[str, object]:
        unresolved: list[str] = []
        late: list[str] = []
        resolved_on_time = 0
        for receipt_id, receipt in self._receipts.items():
            witness = self._statuses.get(receipt_id)
            if witness is None:
                if now > receipt.inclusion_deadline:
                    unresolved.append(receipt_id)
            elif witness.witnessed_at > receipt.inclusion_deadline:
                late.append(receipt_id)
            else:
                resolved_on_time += 1
        return {
            "receipts": len(self._receipts),
            "resolved_on_time": resolved_on_time,
            "unresolved": tuple(sorted(unresolved)),
            "late": tuple(sorted(late)),
            "resolution_rate": resolved_on_time / len(self._receipts) if self._receipts else 1.0,
        }


def verify_receipt(
    receipt: IngestionReceipt,
    public_key: Ed25519PublicKey,
    *,
    revoked_effective_at: int | None = None,
) -> bool:
    if revoked_effective_at is not None and receipt.accepted_at >= revoked_effective_at:
        return False
    try:
        public_key.verify(
            base64.b64decode(receipt.signature_b64), canonical_json(receipt.unsigned_payload())
        )
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def verify_status_witness(
    witness: ReceiptStatusWitness,
    public_key: Ed25519PublicKey,
    *,
    revoked_effective_at: int | None = None,
) -> bool:
    if revoked_effective_at is not None and witness.witnessed_at >= revoked_effective_at:
        return False
    try:
        public_key.verify(
            base64.b64decode(witness.signature_b64),
            canonical_json(witness.unsigned_payload()),
        )
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def verify_receipt_inclusion(
    receipt: IngestionReceipt,
    witness: ReceiptStatusWitness,
    content: bytes,
    public_key: Ed25519PublicKey,
) -> bool:
    """Verify signature, commitment binding and the terminal RFC-9162 inclusion path."""
    if not verify_receipt(receipt, public_key) or not verify_status_witness(witness, public_key):
        return False
    if witness.receipt_id != receipt.receipt_id or witness.status != "INCLUDED":
        return False
    if hashlib.sha256(content).hexdigest() != receipt.content_hash:
        return False
    if (
        witness.leaf_index is None
        or witness.tree_size is None
        or witness.root_hex is None
        or witness.inclusion_proof_b64 is None
    ):
        return False
    try:
        proof = tuple(base64.b64decode(item) for item in witness.inclusion_proof_b64)
        expected_root = bytes.fromhex(witness.root_hex)
    except (ValueError, TypeError):
        return False
    return verify_inclusion(
        content,
        index=witness.leaf_index,
        tree_size=witness.tree_size,
        proof=proof,
        expected_root=expected_root,
    )


class ReplicatedObjectStore:
    """Small deterministic fault-injection model for post-anchor availability."""

    def __init__(self, replicas: int):
        if replicas <= 0:
            raise ValueError("replicas must be positive")
        self._nodes: list[dict[str, bytes]] = [{} for _ in range(replicas)]

    def put(self, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        for node in self._nodes:
            node[digest] = bytes(payload)
        return digest

    def delete_replica(self, digest: str, replica: int) -> None:
        self._nodes[replica].pop(digest, None)

    def get(self, digest: str) -> bytes | None:
        return next((node[digest] for node in self._nodes if digest in node), None)

    def available_replicas(self, digest: str) -> int:
        return sum(digest in node for node in self._nodes)


class EpochAnchorRegistry:
    """Executable model of immutable epoch/version roots and correction chaining."""

    def __init__(self):
        self._roots: dict[tuple[str, int, int], str] = {}

    def anchor(self, *, city: str, epoch: int, version: int, root_hex: str) -> None:
        key = (city, epoch, version)
        if key in self._roots:
            raise ValueError("epoch/version already anchored")
        if version > 0 and (city, epoch, version - 1) not in self._roots:
            raise ValueError("correction must extend the preceding version")
        self._roots[key] = root_hex

    def root(self, *, city: str, epoch: int, version: int) -> str:
        return self._roots[(city, epoch, version)]

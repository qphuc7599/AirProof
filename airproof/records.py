from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True, slots=True)
class Observation:
    user_id: int
    epoch: int
    cell: int
    group: int
    value: float
    sigma: float
    quality: float
    size_bytes: int
    nullifier: str
    direct_arrival: int | None
    relay_arrival: int | None
    corrupted: bool = False
    source_class: str = "citizen"

    def public_dict(self) -> dict[str, Any]:
        """Canonical committed fields; exact coordinates and identity stay inside ciphertext."""
        data = asdict(self)
        data["value"] = round(float(self.value), 9)
        data["sigma"] = round(float(self.sigma), 9)
        data["quality"] = round(float(self.quality), 9)
        return data

    def with_value(self, value: float, *, corrupted: bool = True) -> Observation:
        return replace(self, value=float(value), corrupted=corrupted)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


RAW_NULLIFIER_DOMAIN = b"AirProof/raw/v1"
RELEASE_NULLIFIER_DOMAIN = b"AirProof/release/v1"


def _nullifier(secret: bytes, domain: bytes, *parts: object) -> str:
    """Derive a deterministic, domain-separated pseudorandom nullifier.

    The public value never embeds the user identifier.  Linkability is restricted to
    the exact protocol domain encoded by ``parts`` and the credential secret.
    """
    if len(secret) < 16:
        raise ValueError("credential secret must contain at least 128 bits")
    message = b"\x00".join([domain, *(str(part).encode("utf-8") for part in parts)])
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def raw_nullifier(
    secret: bytes,
    *,
    city: str,
    group: int,
    epoch: int,
    interval: int,
    policy_id: str,
) -> str:
    """One raw encrypted contribution per user/group/epoch/interval."""
    return _nullifier(secret, RAW_NULLIFIER_DOMAIN, city, group, epoch, interval, policy_id)


def release_nullifier(
    secret: bytes,
    *,
    city: str,
    group: int,
    epoch: int,
    policy_id: str,
) -> str:
    """One bounded public-release contribution per user/group/epoch."""
    return _nullifier(secret, RELEASE_NULLIFIER_DOMAIN, city, group, epoch, policy_id)


def contribution_nullifier(secret: bytes, user_id: int, group: int, epoch: int) -> str:
    """Backward-compatible raw-path helper used by the synthetic simulator.

    ``user_id`` is retained only as an interval-domain input for old callers; it is not
    encoded in clear text in the returned nullifier.
    """
    return raw_nullifier(
        secret,
        city="synthetic",
        group=group,
        epoch=epoch,
        interval=epoch,
        policy_id=f"world-user-{user_id}",
    )


def seal_payload(payload: dict[str, Any], signing_key: Ed25519PrivateKey, aes_key: bytes, nonce: bytes) -> dict[str, str]:
    plaintext = canonical_json(payload)
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext, b"airproof-envelope-v1")
    signature = signing_key.sign(nonce + ciphertext)
    return {
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "signature": base64.b64encode(signature).decode(),
    }


def open_payload(envelope: dict[str, str], verify_key: Ed25519PublicKey, aes_key: bytes) -> dict[str, Any]:
    nonce = base64.b64decode(envelope["nonce"])
    ciphertext = base64.b64decode(envelope["ciphertext"])
    verify_key.verify(base64.b64decode(envelope["signature"]), nonce + ciphertext)
    plaintext = AESGCM(aes_key).decrypt(nonce, ciphertext, b"airproof-envelope-v1")
    return json.loads(plaintext)

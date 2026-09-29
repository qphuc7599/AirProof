"""Concrete source payloads; simulator arrival/attack metadata is never encoded."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import struct

from .v5_wire_control import _uint

_RAW = struct.Struct("!4sQQIIddd32s")
_RELEASE = struct.Struct("!4sQQId32s")
RAW_PAYLOAD_BYTES = _RAW.size
RELEASE_PAYLOAD_BYTES = _RELEASE.size


@dataclass(frozen=True)
class RawSourcePayload:
    user_id: int
    acquisition_epoch: int
    cell: int
    group: int
    value: float
    sigma: float
    quality: float
    nullifier: str


@dataclass(frozen=True)
class ReleaseSourcePayload:
    user_id: int
    acquisition_epoch: int
    canonical_group: int
    local_aggregate: float
    release_id: str


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite binary64 number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} does not fit binary64") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _identifier(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be canonical lowercase 256-bit hex")
    return bytes.fromhex(value)


def encode_raw_payload(value: RawSourcePayload) -> bytes:
    user = _uint(value.user_id, 64, "user id")
    epoch = _uint(value.acquisition_epoch, 64, "acquisition epoch")
    cell = _uint(value.cell, 32, "cell")
    group = _uint(value.group, 32, "group")
    measurement, sigma, quality = (_finite(x, label) for x, label in
                                    ((value.value, "value"), (value.sigma, "sigma"), (value.quality, "quality")))
    if sigma <= 0 or not 0 <= quality <= 1:
        raise ValueError("positive sigma and quality in [0,1] required")
    return _RAW.pack(b"APR5", user, epoch, cell, group, measurement, sigma, quality,
                     _identifier(value.nullifier, "raw nullifier"))


def encode_observation_payload(observation) -> bytes:
    """Project immutable acquired fields only; never fabricate a new observation."""
    return encode_raw_payload(RawSourcePayload(observation.user_id, observation.epoch,
        observation.cell, observation.group, observation.value, observation.sigma,
        observation.quality, observation.nullifier))


def decode_raw_payload(payload: bytes) -> RawSourcePayload:
    if not isinstance(payload, bytes) or len(payload) != _RAW.size:
        raise ValueError("invalid raw source payload length")
    magic, user, epoch, cell, group, value, sigma, quality, nullifier = _RAW.unpack(payload)
    if magic != b"APR5":
        raise ValueError("invalid raw source payload version")
    result = RawSourcePayload(user, epoch, cell, group, value, sigma, quality, nullifier.hex())
    encode_raw_payload(result)  # Validate finite values and semantic domains on decode too.
    return result


def encode_release_payload(value: ReleaseSourcePayload) -> bytes:
    return _RELEASE.pack(b"APL5", _uint(value.user_id, 64, "user id"),
        _uint(value.acquisition_epoch, 64, "acquisition epoch"),
        _uint(value.canonical_group, 32, "canonical group"),
        _finite(value.local_aggregate, "local aggregate"), _identifier(value.release_id, "release id"))


def decode_release_payload(payload: bytes) -> ReleaseSourcePayload:
    if not isinstance(payload, bytes) or len(payload) != _RELEASE.size:
        raise ValueError("invalid release source payload length")
    magic, user, epoch, group, aggregate, identifier = _RELEASE.unpack(payload)
    if magic != b"APL5":
        raise ValueError("invalid release source payload version")
    result = ReleaseSourcePayload(user, epoch, group, aggregate, identifier.hex())
    encode_release_payload(result)
    return result

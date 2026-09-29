"""Concrete control frames for the v5 authenticated-link transport emulation.

These are framing codecs, not signatures or standalone authentication. In
particular, a 16-byte ACK is not an Ed25519 receipt. Peer identity and opaque
packet-handle binding are supplied by the assumed authenticated link/session.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
import struct

MAGIC = b"AP5C"
_SHORT = struct.Struct("!4sBBHQ")
_HISTORY = struct.Struct("!4sBBH7Q")
HEADER_BYTES = _SHORT.size
ACK_BYTES = _SHORT.size
HISTORY_BYTES = _HISTORY.size
_HEADER_KIND, _ACK_KIND, _HISTORY_KIND = 1, 2, 3
_PAYLOAD_BYTES = {"raw": 512, "release": 256}
_PACKET_KINDS = {"raw": 1, "release": 2}


class AckStatus(IntEnum):
    ACCEPTED = 1
    DELIVERED = 2
    ALREADY_HELD = 3
    BUFFER_FULL = 4
    CONTACT_CAPACITY = 5
    KNOWN_DELIVERED = 6


@dataclass(frozen=True)
class HeaderFrame:
    packet_id: int
    packet_kind: str
    payload_bytes: int


@dataclass(frozen=True)
class AckFrame:
    packet_id: int
    status: AckStatus


@dataclass(frozen=True)
class HistoryFrame:
    node_id: int
    state: int
    observed: int
    exposures: tuple[int, int]
    successes: tuple[int, int]


def _uint(value, bits, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**bits:
        raise ValueError(f"{label} must be an unsigned {bits}-bit integer")
    return int(value)


def _unpack(frame, layout):
    if not isinstance(frame, bytes) or len(frame) != layout.size:
        raise ValueError(f"control frame must be exactly {layout.size} bytes")
    values = layout.unpack(frame)
    if values[0] != MAGIC:
        raise ValueError("wrong control frame magic/version")
    return values


def encode_header(packet_id: int, *, packet_kind: str = "raw") -> bytes:
    packet_id = _uint(packet_id, 64, "packet handle")
    if packet_kind not in _PACKET_KINDS:
        raise ValueError("unknown packet kind")
    return _SHORT.pack(MAGIC, _HEADER_KIND, _PACKET_KINDS[packet_kind],
                       _PAYLOAD_BYTES[packet_kind], packet_id)


def decode_header(frame: bytes) -> HeaderFrame:
    _, kind, packet_kind, payload_bytes, packet_id = _unpack(frame, _SHORT)
    inverse = {value: key for key, value in _PACKET_KINDS.items()}
    if kind != _HEADER_KIND or packet_kind not in inverse:
        raise ValueError("wrong header frame kind")
    name = inverse[packet_kind]
    if payload_bytes != _PAYLOAD_BYTES[name]:
        raise ValueError("packet size disagrees with signed-payload codec")
    return HeaderFrame(packet_id, name, payload_bytes)


def encode_ack(packet_id: int, status: AckStatus) -> bytes:
    packet_id = _uint(packet_id, 64, "packet handle")
    status = _uint(status, 8, "ACK status")
    try:
        status = AckStatus(status)
    except ValueError as exc:
        raise ValueError("unknown ACK status") from exc
    return _SHORT.pack(MAGIC, _ACK_KIND, status, 0, packet_id)


def decode_ack(frame: bytes) -> AckFrame:
    _, kind, status, reserved, packet_id = _unpack(frame, _SHORT)
    if kind != _ACK_KIND or reserved != 0:
        raise ValueError("wrong ACK kind or nonzero reserved bytes")
    try:
        return AckFrame(packet_id, AckStatus(status))
    except ValueError as exc:
        raise ValueError("unknown ACK status") from exc


def _validate_history(value: HistoryFrame):
    node = _uint(value.node_id, 64, "node id")
    state = _uint(value.state, 1, "hazard state")
    observed = _uint(value.observed, 64, "observed slots")
    if len(value.exposures) != 2 or len(value.successes) != 2:
        raise ValueError("exactly two hazard states required")
    exposures = tuple(_uint(x, 64, "exposure count") for x in value.exposures)
    successes = tuple(_uint(x, 64, "success count") for x in value.successes)
    if any(success > exposure for success, exposure in zip(successes, exposures)):
        raise ValueError("success count exceeds exposure count")
    if sum(exposures) != max(observed - 1, 0):
        raise ValueError("transition exposures disagree with observed slots")
    if observed == 0 and state != 0:
        raise ValueError("unobserved hazard state must be initial zero")
    return node, state, observed, exposures, successes


def encode_history(value: HistoryFrame) -> bytes:
    node, state, observed, exposures, successes = _validate_history(value)
    return _HISTORY.pack(MAGIC, _HISTORY_KIND, state, 0, node, observed,
                         *exposures, *successes, 0)


def decode_history(frame: bytes, *, expected_node: int | None = None) -> HistoryFrame:
    _, kind, state, reserved, node, observed, off_exposure, on_exposure, off_success, on_success, tail = _unpack(frame, _HISTORY)
    if kind != _HISTORY_KIND or reserved != 0 or tail != 0:
        raise ValueError("wrong history kind or nonzero reserved bytes")
    value = HistoryFrame(node, state, observed, (off_exposure, on_exposure), (off_success, on_success))
    _validate_history(value)
    if expected_node is not None and node != _uint(expected_node, 64, "expected node"):
        raise ValueError("history node does not match authenticated peer")
    return value


def encode_gateway_history(history, node: int) -> bytes:
    """Serialize precisely the local sufficient statistics used by the model."""
    node = _uint(node, 64, "node id")
    if node >= len(history.state):
        raise ValueError("node outside gateway history")
    return encode_history(HistoryFrame(node, history.state[node], history.observed,
        tuple(history.exposures[node]), tuple(history.successes[node])))

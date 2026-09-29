from dataclasses import replace
import struct

import numpy as np
import pytest

from airproof.dtn import GatewayContactHistory
from airproof.v5_transport import ACK_BYTES as OLD_ACK, HEADER_BYTES as OLD_HEADER, HISTORY_BYTES as OLD_HISTORY
from airproof.v5_wire_control import (ACK_BYTES, HEADER_BYTES, HISTORY_BYTES, AckFrame, AckStatus,
    HeaderFrame, HistoryFrame, decode_ack, decode_header, decode_history,
    encode_ack, encode_gateway_history, encode_header, encode_history)


@pytest.mark.parametrize("kind,size", [("raw", 512), ("release", 256)])
def test_header_roundtrip_fixed_size_and_large_handle(kind, size):
    frame = encode_header(2**64 - 1, packet_kind=kind)
    assert len(frame) == HEADER_BYTES == 16
    assert decode_header(frame) == HeaderFrame(2**64 - 1, kind, size)


@pytest.mark.parametrize("status", list(AckStatus))
def test_ack_roundtrip_fixed_size_for_every_protocol_response(status):
    frame = encode_ack(193, status)
    assert len(frame) == ACK_BYTES == 16
    assert decode_ack(frame) == AckFrame(193, status)


def test_actual_gateway_hazard_sufficient_statistics_fit_64_bytes():
    history = GatewayContactHistory(3)
    for epoch in range(300):
        history.observe(tuple(n for n in range(3) if (epoch + n) % (n + 2) == 0))
        for node in range(3):
            frame = encode_gateway_history(history, node)
            assert len(frame) == HISTORY_BYTES == 64
            parsed = decode_history(frame, expected_node=node)
            assert parsed.state == history.state[node]
            assert parsed.observed == history.observed
            assert parsed.exposures == tuple(history.exposures[node])
            assert parsed.successes == tuple(history.successes[node])
            # Reconstruct the precise causal hazard from the bytes, not object access.
            off_on, on_on = (1 + np.array(parsed.successes)) / (2 + np.array(parsed.exposures))
            first = on_on if parsed.state else off_on
            probability = 1 - (1 - first) * (1 - off_on)**5
            assert probability == pytest.approx(history.before_deadline(node, 6), abs=1e-15)
    initial = GatewayContactHistory(1)
    assert decode_history(encode_gateway_history(initial, 0)) == HistoryFrame(0, 0, 0, (0, 0), (0, 0))


@pytest.mark.parametrize("bad", [-1, 2**64, 1.5, True, "1"])
def test_control_integer_domains_reject_unrepresentable_values(bad):
    with pytest.raises(ValueError):
        encode_header(bad)
    with pytest.raises(ValueError):
        encode_ack(bad, AckStatus.ACCEPTED)


def test_malformed_length_magic_kind_and_reserved_bits_are_rejected():
    for encode, decode in ((lambda: encode_header(1), decode_header),
                           (lambda: encode_ack(1, AckStatus.ACCEPTED), decode_ack),
                           (lambda: encode_history(HistoryFrame(0, 0, 0, (0, 0), (0, 0))), decode_history)):
        frame = encode()
        for malformed in (frame[:-1], frame+b"x", b"BAD!"+frame[4:], frame[:4]+b"\xff"+frame[5:]):
            with pytest.raises(ValueError):
                decode(malformed)
    header = bytearray(encode_header(1))
    header[6:8] = struct.pack("!H", 400)
    with pytest.raises(ValueError, match="size"):
        decode_header(bytes(header))
    ack = bytearray(encode_ack(1, AckStatus.ACCEPTED))
    ack[7] = 1
    with pytest.raises(ValueError, match="reserved"):
        decode_ack(bytes(ack))
    with pytest.raises(ValueError):
        encode_ack(1, 255)


def test_history_count_consistency_and_peer_binding():
    good = HistoryFrame(2, 1, 5, (3, 1), (2, 1))
    assert decode_history(encode_history(good), expected_node=2) == good
    for bad in (replace(good, state=2), replace(good, observed=4), replace(good, successes=(4, 1)),
                replace(good, exposures=(-1, 5)), replace(good, successes=(1,)),
                replace(good, observed=2**64)):
        with pytest.raises(ValueError):
            encode_history(bad)
    with pytest.raises(ValueError, match="authenticated peer"):
        decode_history(encode_history(good), expected_node=3)
    frame = bytearray(encode_history(good))
    frame[-1] = 1
    with pytest.raises(ValueError, match="reserved"):
        decode_history(bytes(frame))


def test_serialized_lengths_are_byte_equivalent_to_existing_accounting():
    assert (HEADER_BYTES, ACK_BYTES, HISTORY_BYTES) == (OLD_HEADER, OLD_ACK, OLD_HISTORY)
    history = encode_gateway_history(GatewayContactHistory(1), 0)
    for with_history in (False, True):
        for release in (False, True):
            for attempts in range(5):
                frames = [history] if with_history else []
                if release:
                    frames += [encode_header(10, packet_kind="release"), encode_ack(10, AckStatus.DELIVERED)]
                for mid in range(attempts):
                    frames += [encode_header(mid), encode_ack(mid, AckStatus.CONTACT_CAPACITY)]
                actual = sum(map(len, frames))
                original = int(with_history)*OLD_HISTORY + (attempts+int(release))*(OLD_HEADER+OLD_ACK)
                assert actual == original
                assert (actual <= 128) == (original <= 128)

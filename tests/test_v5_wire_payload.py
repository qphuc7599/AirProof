from dataclasses import asdict, replace
import hashlib
import struct

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from airproof.records import Observation
from airproof.v5_transport import decode_packet, encode_packet
from airproof.v5_wire_payload import (RAW_PAYLOAD_BYTES, RELEASE_PAYLOAD_BYTES,
    RawSourcePayload, ReleaseSourcePayload, decode_raw_payload, decode_release_payload,
    encode_observation_payload, encode_raw_payload, encode_release_payload)

RAW = RawSourcePayload(17, 671, 1023, 3, 25.75, 1.5, .9, hashlib.sha256(b"raw").hexdigest())
RELEASE = ReleaseSourcePayload(17, 671, 3, 23.5, hashlib.sha256(b"release").hexdigest())


def test_actual_acquired_fields_fit_signed_encrypted_512_byte_packet():
    observation = Observation(RAW.user_id, RAW.acquisition_epoch, RAW.cell, RAW.group,
        RAW.value, RAW.sigma, RAW.quality, 512, RAW.nullifier, 991234, 991235, True, "citizen")
    original = asdict(observation)
    payload = encode_observation_payload(observation)
    assert len(payload) == RAW_PAYLOAD_BYTES == 84
    assert len(payload) <= 413
    assert decode_raw_payload(payload) == RAW
    private = Ed25519PrivateKey.generate()
    key = b"k"*32
    packet = encode_packet(payload, kind="raw", encryption_key=key, signing_key=private)
    assert len(packet) == 512
    kind, decoded = decode_packet(packet, encryption_key=key, verification_key=private.public_key())
    assert kind == "raw" and decode_raw_payload(decoded) == RAW
    assert asdict(observation) == original
    # Acquisition-only payload is invariant to hypothetical future/attack labels.
    assert encode_observation_payload(replace(observation, direct_arrival=None, relay_arrival=None,
        corrupted=False, source_class="another-local-label")) == payload


def test_actual_local_release_fields_fit_signed_encrypted_256_byte_packet():
    payload = encode_release_payload(RELEASE)
    assert len(payload) == RELEASE_PAYLOAD_BYTES == 64
    assert len(payload) <= 157
    private = Ed25519PrivateKey.generate()
    key = b"r"*32
    packet = encode_packet(payload, kind="release", encryption_key=key, signing_key=private)
    assert len(packet) == 256
    kind, decoded = decode_packet(packet, encryption_key=key, verification_key=private.public_key())
    assert kind == "release" and decode_release_payload(decoded) == RELEASE


@pytest.mark.parametrize("field,bad", [("user_id", -1), ("user_id", 2**64),
    ("acquisition_epoch", -1), ("cell", 2**32), ("group", True), ("value", float("nan")),
    ("sigma", 0.), ("quality", 1.01), ("value", 10**400), ("nullifier", "not-a-digest")])
def test_raw_domain_and_overflow_rejected(field, bad):
    with pytest.raises(ValueError):
        encode_raw_payload(replace(RAW, **{field: bad}))


@pytest.mark.parametrize("field,bad", [("user_id", 2**64), ("acquisition_epoch", -1),
    ("canonical_group", 2**32), ("local_aggregate", float("inf")), ("release_id", "a"*63)])
def test_release_domain_and_overflow_rejected(field, bad):
    with pytest.raises(ValueError):
        encode_release_payload(replace(RELEASE, **{field: bad}))


def test_decode_rejects_malformed_and_nonfinite_binary_fields():
    for encode, decode, value in ((encode_raw_payload, decode_raw_payload, RAW),
                                  (encode_release_payload, decode_release_payload, RELEASE)):
        payload = encode(value)
        for malformed in (payload[:-1], payload+b"x", b"BAD!"+payload[4:]):
            with pytest.raises(ValueError):
                decode(malformed)
    raw = bytearray(encode_raw_payload(RAW))
    raw[28:36] = struct.pack("!d", float("nan"))
    with pytest.raises(ValueError):
        decode_raw_payload(bytes(raw))
    release = bytearray(encode_release_payload(RELEASE))
    release[24:32] = struct.pack("!d", float("inf"))
    with pytest.raises(ValueError):
        decode_release_payload(bytes(release))

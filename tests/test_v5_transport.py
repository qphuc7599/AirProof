from dataclasses import asdict, replace

import pytest
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airproof.records import Observation
from airproof.v5_transport import (TransportTrace, decode_packet, encode_packet,
                                  generate_transport_trace, simulate_transport)


def record(user=0, epoch=0, group=0, suffix=""):
    return Observation(user, epoch, 0, group, 10., 1., 1., 512,
                       f"{user}:{epoch}:{group}:{suffix}", 999, 888)


def trace(nodes=2, acquisition=1, drain=2, gateways=None, peers=None):
    horizon = acquisition + drain
    return TransportTrace(1, nodes, acquisition, drain, tuple(i % 2 for i in range(nodes)),
                          tuple(gateways or [()] * horizon), tuple(peers or [()] * horizon), "manual")


@pytest.mark.parametrize("kind,size", [("raw", 512), ("release", 256)])
def test_fixed_signed_aead_codec(kind, size):
    private = Ed25519PrivateKey.generate()
    key = bytes(range(32))
    payload = b"binary\0payload" * 5
    packet = encode_packet(payload, kind=kind, encryption_key=key, signing_key=private)
    assert len(packet) == size
    assert decode_packet(packet, encryption_key=key, verification_key=private.public_key()) == (kind, payload)
    for offset in (0, 20, size - 1):
        changed = bytearray(packet)
        changed[offset] ^= 1
        with pytest.raises(InvalidSignature):
            decode_packet(bytes(changed), encryption_key=key, verification_key=private.public_key())
    with pytest.raises(InvalidTag):
        decode_packet(packet, encryption_key=b"x" * 32, verification_key=private.public_key())
    with pytest.raises(ValueError):
        encode_packet(b"x" * (size - 98), kind=kind, encryption_key=key, signing_key=private)


def test_deadline_inclusive_and_tail_and_immutable_metadata():
    r = record()
    original = asdict(r)
    gateways = [()] * 25
    gateways[24] = (0,)
    result = simulate_transport(trace(acquisition=1, drain=24, gateways=gateways), [r], {})
    assert result.raw_arrivals == {r.nullifier: 24}
    assert result.release_arrivals == {(0, 0): 24}
    assert asdict(r) == original
    gateways = [()] * 26
    gateways[25] = (0,)
    result = simulate_transport(trace(acquisition=1, drain=25, gateways=gateways), [r], {})
    assert not result.raw_arrivals and not result.release_arrivals


@pytest.mark.parametrize("policy", ["binary_spray_wait", "epidemic_cap", "airproof_deadline"])
def test_peer_copy_cannot_chain_same_slot(policy):
    r = record()
    t = trace(nodes=3, drain=2, gateways=[(), (), (2,)],
              peers=[((0, 1), (1, 2)), (), ()])
    result = simulate_transport(t, [r], {}, policy)
    assert not result.raw_arrivals
    assert result.metrics["raw_peer_transmissions"] == 1
    t = replace(t, peer_contacts=(((0, 1), (1, 2)), ((1, 2),), ()))
    result = simulate_transport(t, [r], {}, policy)
    # Epidemic transfers one token: receiver waits for gateway and cannot forward.
    assert bool(result.raw_arrivals) == (policy != "epidemic_cap")
    assert result.metrics["token_violations"] == 0
    assert result.metrics["max_live_tokens"] <= 4
    assert result.metrics["max_live_copies"] <= 4


def test_gateway_reservation_and_serialized_control_do_not_borrow():
    pool = [record(group=g) for g in range(4)]
    t = trace(drain=0, gateways=[(0,)])
    enabled = simulate_transport(t, pool, {}, release=True)
    disabled = simulate_transport(t, pool, {}, release=False)
    assert len(enabled.raw_arrivals) == len(disabled.raw_arrivals) == 1
    assert enabled.metrics["release_payload_bytes"] == 256
    assert enabled.metrics["max_contact_direction_bytes"] <= 1024
    assert enabled.metrics["max_control_direction_bytes"] <= 128
    assert disabled.metrics["gateway_reserved_bytes"] == 256


def test_four_tokens_bound_replication_over_repeated_contacts():
    t = trace(nodes=5, drain=3, peers=[((0, 1), (0, 2)), ((1, 3),),
                                     ((0, 4), (1, 4), (2, 4), (3, 4)), ()])
    result = simulate_transport(t, [record()], {}, "binary_spray_wait")
    assert result.metrics["max_live_copies"] == 4
    assert result.metrics["raw_peer_transmissions"] == 3
    assert result.metrics["raw_remaining_copies"] == 4
    assert result.metrics["token_violations"] == 0


def test_raw_and_release_buffers_are_separate_and_finite():
    pool = [record(epoch=e, group=g) for e in range(30) for g in range(4)]
    result = simulate_transport(trace(acquisition=30, drain=0), pool, {})
    assert result.metrics["max_raw_buffer_bytes"] <= 18432
    assert result.metrics["max_release_buffer_bytes"] == 6144
    assert result.metrics["release_buffer_drops"] > 0
    assert result.metrics["raw_source_drops"] > 0


def test_replace_user_history_cannot_change_other_release_admission():
    t = trace(acquisition=6, drain=2, gateways=[(0, 1)] * 8,
              peers=[((0, 1),)] * 8)
    ordinary = [record(user=u, epoch=e) for u in (0, 1) for e in range(6)]
    flooding = [r for r in ordinary if r.user_id == 1] + [record(user=0, epoch=e, group=g) for e in range(6) for g in range(50)]
    for pool in (ordinary, flooding, [r for r in ordinary if r.user_id == 1]):
        result = simulate_transport(t, pool, {})
        assert {k: v for k, v in result.release_arrivals.items() if k[0] == 1} == {(1, e): e for e in range(6)}
        assert result.metrics["max_raw_buffer_bytes"] <= 18432


def test_duplicate_gateway_ack_is_local_not_global_deletion():
    r = record()
    t = trace(drain=2, gateways=[(), (0,), (1,)], peers=[((0, 1),), (), ()])
    result = simulate_transport(t, [r], {}, "binary_spray_wait")
    assert result.raw_arrivals == {r.nullifier: 1}
    assert result.metrics["duplicate_headers"] == 1
    assert result.metrics["raw_gateway_transmissions"] == 1
    assert result.metrics["raw_remaining_copies"] == 0


def test_trace_repeatability_exogenous_horizon_and_sparse_matching():
    config = {"scale": {"agents": 20, "grid_side": 2, "groups": 3,
                        "acquisition_epochs": 6, "drain_epochs": 2},
              "network": {"contact_probability": 1.}}
    first = generate_transport_trace(config, 123)
    second = generate_transport_trace(config, 123)
    assert first == second
    assert len(first.gateway_contacts) == len(first.peer_contacts) == 8
    for pairs in first.peer_contacts:
        endpoints = [node for pair in pairs for node in pair]
        assert len(endpoints) == len(set(endpoints))
        assert len(pairs) <= 10
    assert generate_transport_trace(config, 124).trace_hash != first.trace_hash


def test_invalid_record_and_codec_budget_inputs():
    with pytest.raises(ValueError, match="unique"):
        simulate_transport(trace(), [record(), record()], {})
    with pytest.raises(ValueError, match="wire sizes"):
        simulate_transport(trace(), [], {"transport": {"raw_packet_bytes": 400}})
    with pytest.raises(ValueError, match="horizon"):
        simulate_transport(trace(), [record(epoch=2)], {})

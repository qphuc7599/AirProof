"""Finite, causal transport emulation with an independent source release lane."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import os
import struct
from typing import Mapping

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .dtn import DTNConfig, GatewayContactHistory, Message, deadline_priority
from .records import Observation
from .simulator import _connectivity, _random_walks
from .v5_wire_control import (HEADER_BYTES, ACK_BYTES, HISTORY_BYTES, AckStatus,
                              encode_header, encode_ack, encode_gateway_history)

PACKET_BYTES = {"raw": 512, "release": 256}


def encode_packet(payload: bytes, *, kind: str, encryption_key: bytes,
                  signing_key: Ed25519PrivateKey, nonce: bytes | None = None) -> bytes:
    """Fixed binary envelope: version/kind/length, nonce, padded AEAD, signature."""
    size = PACKET_BYTES[kind]
    capacity = size - 99
    if len(payload) > capacity:
        raise ValueError("payload exceeds fixed packet capacity")
    nonce = os.urandom(12) if nonce is None else nonce
    if len(nonce) != 12:
        raise ValueError("AES-GCM nonce must have 12 bytes")
    header = struct.pack("!4sBH", b"APV5", int(kind == "release"), len(payload)) + nonce
    ciphertext = AESGCM(encryption_key).encrypt(nonce, payload.ljust(capacity, b"\0"), header)
    signed = header + ciphertext
    return signed + signing_key.sign(signed)


def decode_packet(packet: bytes, *, encryption_key: bytes,
                  verification_key: Ed25519PublicKey) -> tuple[str, bytes]:
    if len(packet) not in PACKET_BYTES.values():
        raise ValueError("invalid fixed packet length")
    verification_key.verify(packet[-64:], packet[:-64])
    magic, kind_id, length = struct.unpack("!4sBH", packet[:7])
    kind = {0: "raw", 1: "release"}.get(kind_id)
    if magic != b"APV5" or kind is None or len(packet) != PACKET_BYTES[kind] or length > len(packet) - 99:
        raise ValueError("invalid packet header")
    padded = AESGCM(encryption_key).decrypt(packet[7:19], packet[19:-64], packet[:19])
    return kind, padded[:length]


@dataclass(frozen=True)
class TransportTrace:
    seed: int
    nodes: int
    acquisition_epochs: int
    drain_epochs: int
    node_groups: tuple[int, ...]
    gateway_contacts: tuple[tuple[int, ...], ...]
    peer_contacts: tuple[tuple[tuple[int, int], ...], ...]
    trace_hash: str


@dataclass(frozen=True)
class TransportResult:
    raw_arrivals: Mapping[str, int]
    release_arrivals: Mapping[tuple[int, int], int]
    metrics: dict


def generate_transport_trace(config: dict, seed: int) -> TransportTrace:
    scale, world, network = config.get("scale", {}), config.get("world", {}), config.get("network", {})
    nodes = int(world.get("agents", scale.get("agents", 1000)))
    acquisition = int(world.get("steps", scale.get("acquisition_epochs", 672)))
    drain = int(scale.get("drain_epochs", config.get("transport", {}).get("drain_epochs", 24)))
    side = int(world.get("grid_side", scale.get("grid_side", 32)))
    groups = int(world.get("groups", scale.get("groups", 4)))
    if min(nodes, acquisition, side, groups) < 1 or drain < 0:
        raise ValueError("invalid transport dimensions")
    horizon = acquisition + drain
    mobility_rng, gateway_rng, peer_rng = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 5005]).spawn(3)]
    node_groups = np.arange(nodes) % groups
    cells = _random_walks(side, horizon, nodes, mobility_rng)
    online = _connectivity(node_groups, horizon, float(network.get("availability", .4)),
                           float(network.get("group_correlation", .5)), gateway_rng,
                           float(network.get("outage_median_hours", 24)), float(network.get("outage_p95_hours", 72)))
    probability = float(network.get("contact_probability", .15))
    if not 0 <= probability <= 1:
        raise ValueError("contact probability must lie in [0,1]")
    peers = []
    for epoch in range(horizon):
        bins = defaultdict(list)
        for node, cell in enumerate(cells[epoch]):
            bins[int(cell)].append(node)
        pairs = []
        for bucket in bins.values():
            peer_rng.shuffle(bucket)
            # A sparse matching: at most one colocated peer encounter per node/slot.
            for j in range(0, len(bucket) - 1, 2):
                if peer_rng.random() < probability:
                    pairs.append((bucket[j], bucket[j + 1]))
        peers.append(tuple(pairs))
    gateways = tuple(tuple(map(int, np.flatnonzero(row))) for row in online)
    peers = tuple(peers)
    digest = hashlib.sha256(json.dumps([seed, nodes, acquisition, drain, node_groups.tolist(), gateways, peers], separators=(",", ":")).encode()).hexdigest()
    return TransportTrace(seed, nodes, acquisition, drain, tuple(map(int, node_groups)), gateways, peers, digest)


def simulate_transport(trace: TransportTrace, observations, config: dict,
                       policy: str = "airproof_deadline", release: bool = True) -> TransportResult:
    if policy not in {"direct", "binary_spray_wait", "epidemic_cap", "airproof_deadline"}:
        raise ValueError("unsupported transport policy")
    cfg = config.get("transport", {})
    capacity = int(cfg.get("capacity_bytes_per_direction", 1024))
    raw_size = int(cfg.get("raw_packet_bytes", 512))
    release_size = int(cfg.get("release_packet_bytes", 256))
    raw_buffer = int(cfg.get("raw_buffer_bytes", 18432))
    release_buffer = int(cfg.get("release_buffer_bytes", 6144))
    ttl = int(cfg.get("ttl_epochs", 24))
    token_limit = int(cfg.get("copy_tokens", 4))
    reservation = int(cfg.get("release_gateway_reservation_bytes", 256))
    control_limit = int(cfg.get("control_budget_bytes", 128))
    if (raw_size, release_size) != (512, 256):
        raise ValueError("wire sizes must match the signed AES-GCM codec")
    if min(raw_buffer, release_buffer, token_limit, control_limit) < 1 or ttl < 0:
        raise ValueError("invalid finite transport limits")
    if reservation < release_size or capacity < reservation + control_limit or control_limit < 2 * (HEADER_BYTES + ACK_BYTES):
        raise ValueError("contact cannot support independent release/control reservation")
    raw_slots, release_slots = raw_buffer // raw_size, release_buffer // release_size
    horizon = trace.acquisition_epochs + trace.drain_epochs
    if len(trace.gateway_contacts) != horizon or len(trace.peer_contacts) != horizon:
        raise ValueError("trace horizon mismatch")
    records = tuple(observations)
    if len({r.nullifier for r in records}) != len(records):
        raise ValueError("raw nullifiers must be unique")
    by_epoch = [[] for _ in range(trace.acquisition_epochs)]
    canonical = {}
    for mid, r in enumerate(records):
        if not 0 <= r.user_id < trace.nodes or not 0 <= r.epoch < trace.acquisition_epochs:
            raise ValueError("record outside public population/acquisition horizon")
        by_epoch[r.epoch].append(mid)
        key = (r.user_id, r.epoch)
        if key not in canonical or (r.group, r.nullifier) < (records[canonical[key]].group, records[canonical[key]].nullifier):
            canonical[key] = mid
    messages = [Message(i, r.user_id, r.group, r.epoch, r.epoch + ttl) for i, r in enumerate(records)]
    group_count = max([*trace.node_groups, *(r.group for r in records)], default=0) + 1
    if any(r.group < 0 for r in records):
        raise ValueError("group identifiers must be nonnegative")
    queues = [dict() for _ in range(trace.nodes)]  # mid -> (tokens, usable_from)
    releases = [deque() for _ in range(trace.nodes)]
    known = [dict() for _ in range(trace.nodes)]  # only ACKs received locally; expires with message
    seen = np.zeros((trace.nodes, group_count), dtype=int)
    acknowledged = np.zeros_like(seen)
    raw_arrivals, release_arrivals = {}, {}
    history = GatewayContactHistory(trace.nodes)
    priority_cfg = DTNConfig(nodes=trace.nodes, groups=group_count, ttl_steps=ttl,
                             deadline_deficit_weight=6., deadline_age_weight=1., deadline_slack_weight=.25)
    metrics = dict(policy=policy, trace_hash=trace.trace_hash, raw_generated=len(records),
                   release_generated=len(canonical) if release else 0, raw_source_drops=0,
                   raw_expired_copies=0, release_buffer_drops=0, release_expired=0,
                   raw_peer_transmissions=0, raw_gateway_transmissions=0, duplicate_headers=0,
                   control_bytes=0, raw_payload_bytes=0, release_payload_bytes=0,
                   max_raw_buffer_bytes=0, max_release_buffer_bytes=0,
                   max_contact_direction_bytes=0, max_control_direction_bytes=0,
                   max_live_tokens=0, max_live_copies=0, token_violations=0, gateway_reserved_bytes=0,
                   crypto_mode="numerical transport emulation; fixed sizes validated by real codec, no per-packet crypto runtime")
    # Global token counters are instrumentation only and never enter forwarding decisions.
    live_tokens = [0] * len(records)
    live_copies = [0] * len(records)

    def remove(node, mid):
        tokens, _ = queues[node].pop(mid)
        live_tokens[mid] -= tokens
        live_copies[mid] -= 1

    def remember(node, mid):
        if mid not in known[node]:
            known[node][mid] = messages[mid].deadline
            acknowledged[node, messages[mid].group] += 1
        if mid in queues[node]:
            remove(node, mid)

    def account(control, raw=0, released=0):
        total = control + raw + released
        metrics["control_bytes"] += control
        metrics["raw_payload_bytes"] += raw
        metrics["release_payload_bytes"] += released
        metrics["max_contact_direction_bytes"] = max(metrics["max_contact_direction_bytes"], total)
        metrics["max_control_direction_bytes"] = max(metrics["max_control_direction_bytes"], control)
        if total > capacity or control > control_limit:
            raise AssertionError("serialized contact budget exceeded")

    def ranked(node, epoch, peer=None):
        eligible = [mid for mid, (tokens, usable) in queues[node].items()
                    if usable <= epoch and (peer is None or tokens > 1)]
        if policy != "airproof_deadline":
            return sorted(eligible, key=lambda mid: (messages[mid].created, mid))
        rates = acknowledged[node] / np.maximum(seen[node], 1)
        deficits = np.max(rates) - rates
        destination = node if peer is None else peer
        probabilities = {}
        def rank(mid):
            m = messages[mid]
            remaining = m.deadline - epoch
            if remaining not in probabilities:
                probabilities[remaining] = history.before_deadline(destination, remaining)
            probability = 1. if peer is None else probabilities[remaining]
            return (-deadline_priority(m, epoch, deficits, priority_cfg,
                                      delivery_probability=probability, copies=queues[node][mid][0]), m.created, mid)
        return sorted(eligible, key=rank)

    for epoch in range(horizon):
        for node in range(trace.nodes):
            for mid in list(queues[node]):
                if messages[mid].deadline < epoch:
                    remove(node, mid)
                    metrics["raw_expired_copies"] += 1
            for mid, deadline in list(known[node].items()):
                if deadline < epoch:
                    del known[node][mid]
            while releases[node] and releases[node][0][1] + ttl < epoch:
                releases[node].popleft()
                metrics["release_expired"] += 1
        if epoch < trace.acquisition_epochs:
            for mid in by_epoch[epoch]:
                r = records[mid]
                seen[r.user_id, r.group] += 1
                if len(queues[r.user_id]) < raw_slots:
                    queues[r.user_id][mid] = (token_limit, epoch)
                    live_tokens[mid] = token_limit
                    live_copies[mid] = 1
                    metrics["max_live_copies"] = max(metrics["max_live_copies"], 1)
                    metrics["max_live_tokens"] = max(metrics["max_live_tokens"], token_limit)
                else:
                    metrics["raw_source_drops"] += 1
                key = (r.user_id, r.epoch)
                if release and canonical[key] == mid:
                    if len(releases[r.user_id]) < release_slots:
                        releases[r.user_id].append(key)
                    else:
                        metrics["release_buffer_drops"] += 1
        metrics["max_raw_buffer_bytes"] = max(metrics["max_raw_buffer_bytes"], max(map(len, queues), default=0) * raw_size)
        metrics["max_release_buffer_bytes"] = max(metrics["max_release_buffer_bytes"], max(map(len, releases), default=0) * release_size)
        contacts = trace.gateway_contacts[epoch]
        history.observe(contacts)
        for node in contacts:
            # Always reserve release and control capacity, even if a lane is idle.
            metrics["gateway_reserved_bytes"] += reservation
            released = 0
            release_control = 0
            if releases[node]:
                key = releases[node].popleft()
                handle = canonical[key]
                release_control = len(encode_header(handle, packet_kind="release")) + len(encode_ack(handle, AckStatus.DELIVERED))
                release_arrivals[key] = epoch
                released = release_size
            used_control = release_control
            raw_bytes = 0
            raw_lane = capacity - reservation - control_limit
            # Raw control has its own cap so raw flooding cannot consume release ACKs.
            raw_control_cap = control_limit - (HEADER_BYTES + ACK_BYTES)
            raw_control = 0
            for mid in ranked(node, epoch):
                if raw_control + HEADER_BYTES + ACK_BYTES > raw_control_cap:
                    break
                header_frame = encode_header(mid)
                if records[mid].nullifier in raw_arrivals:
                    # Gateway reveals this only after a transmitted header; sender had no oracle.
                    metrics["duplicate_headers"] += 1
                    ack_frame = encode_ack(mid, AckStatus.KNOWN_DELIVERED)
                else:
                    if raw_bytes + raw_size > raw_lane:
                        # Capacity NACK follows the header too; no pre-contact oracle.
                        raw_control += len(header_frame) + len(encode_ack(mid, AckStatus.CONTACT_CAPACITY))
                        continue
                    raw_bytes += raw_size
                    raw_arrivals[records[mid].nullifier] = epoch
                    metrics["raw_gateway_transmissions"] += 1
                    ack_frame = encode_ack(mid, AckStatus.DELIVERED)
                raw_control += len(header_frame) + len(ack_frame)
                remember(node, mid)
            account(used_control + raw_control, raw_bytes, released)
        if policy != "direct":
            for a, b in trace.peer_contacts[epoch]:
                for sender, receiver in ((a, b), (b, a)):
                    # Fixed state exchange carries each endpoint's two hazard counts/state.
                    control = len(encode_gateway_history(history, sender)) if policy == "airproof_deadline" else 0
                    raw_bytes = 0
                    for mid in ranked(sender, epoch, receiver):
                        if control + HEADER_BYTES + ACK_BYTES > control_limit:
                            break
                        if raw_bytes + raw_size > capacity - control_limit:
                            break
                        header_frame = encode_header(mid)
                        if mid in known[receiver]:
                            control += len(header_frame) + len(encode_ack(mid, AckStatus.KNOWN_DELIVERED))
                            metrics["duplicate_headers"] += 1
                            remember(sender, mid)
                            continue
                        if mid in queues[receiver] or len(queues[receiver]) >= raw_slots:
                            status = AckStatus.ALREADY_HELD if mid in queues[receiver] else AckStatus.BUFFER_FULL
                            control += len(header_frame) + len(encode_ack(mid, status))
                            metrics["duplicate_headers"] += int(mid in queues[receiver])
                            continue
                        control += len(header_frame) + len(encode_ack(mid, AckStatus.ACCEPTED))
                        tokens, usable = queues[sender][mid]
                        transfer = 1 if policy == "epidemic_cap" else tokens // 2
                        queues[sender][mid] = (tokens - transfer, usable)
                        queues[receiver][mid] = (transfer, epoch + 1)
                        live_copies[mid] += 1
                        metrics["max_live_copies"] = max(metrics["max_live_copies"], live_copies[mid])
                        seen[receiver, messages[mid].group] += 1
                        raw_bytes += raw_size
                        metrics["raw_peer_transmissions"] += 1
                        if not (0 < transfer < tokens and 0 <= live_copies[mid] <= live_tokens[mid] <= token_limit):
                            metrics["token_violations"] += 1
                    account(control, raw_bytes)
        metrics["max_raw_buffer_bytes"] = max(metrics["max_raw_buffer_bytes"], max(map(len, queues), default=0) * raw_size)
    actual_tokens = defaultdict(int)
    for queue in queues:
        for mid, (tokens, _) in queue.items():
            actual_tokens[mid] += tokens
    metrics["token_violations"] += sum(actual_tokens[mid] != count for mid, count in enumerate(live_tokens))
    metrics.update(raw_delivered=len(raw_arrivals), release_delivered=len(release_arrivals),
                   raw_undelivered=len(records) - len(raw_arrivals),
                   release_undelivered=(len(canonical) - len(release_arrivals)) if release else 0,
                   raw_remaining_copies=sum(map(len, queues)), release_remaining=sum(map(len, releases)),
                   total_wire_bytes=metrics["control_bytes"] + metrics["raw_payload_bytes"] + metrics["release_payload_bytes"])
    return TransportResult(raw_arrivals, release_arrivals, metrics)

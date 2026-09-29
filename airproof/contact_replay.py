"""Twenty-second empirical contact replay with explicit synthetic traffic roles."""
from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from .dtn import DTNConfig, DTNTrace, Message


@dataclass(frozen=True)
class ContactDataset:
    name: str
    events: np.ndarray  # canonical columns: observed slot index, endpoint id, endpoint id
    node_ids: np.ndarray
    steps: int
    time_origin_seconds: int
    step_seconds: int
    source_sha256: str
    raw_rows: int
    duplicate_rows_removed: int


def load_contacts(path: Path, *, name: str, expected_sha256: str | None = None) -> ContactDataset:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("empirical archive hash mismatch")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.namelist() if not item.endswith("/")]
            if len(members) != 1:
                raise ValueError("expected one contact table in the archive")
            with archive.open(members[0]) as stream:
                values = np.loadtxt(stream, ndmin=2)
    elif path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            values = np.loadtxt(stream, ndmin=2)
    else:
        values = np.loadtxt(path, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty t/i/j contact table required")
    if np.any(values != np.floor(values)) or np.any(values < 0):
        raise ValueError("timestamps and anonymous IDs must be nonnegative integers")
    events = values.astype(np.int64)
    if np.any(events[:, 1] == events[:, 2]):
        raise ValueError("self contacts do not supply transport capacity")
    origin = int(events[:, 0].min())
    if np.any((events[:, 0] - origin) % 20):
        raise ValueError("this adapter requires the source's native 20-second resolution")
    events[:, 0] = (events[:, 0] - origin) // 20
    events[:, 1:] = np.sort(events[:, 1:], axis=1)
    canonical = np.unique(events, axis=0)  # also sorts chronologically and by endpoint
    nodes = np.unique(canonical[:, 1:])
    return ContactDataset(name, canonical, nodes, int(canonical[:, 0].max()) + 1,
                          origin, 20, digest, len(events), len(events) - len(canonical))


def build_contact_replay(
    dataset: ContactDataset, *, seed: int, base: DTNConfig = DTNConfig(),
    message_probability: float = .002, ttl_seconds: int = 21600,
    warmup_fraction: float = .2, collector_fraction: float = .05,
) -> tuple[DTNTrace, DTNConfig, dict]:
    """Preserve measured contacts; create a prespecified MCS transport workload.

    Collectors are sampled from the public participant roster without consulting
    outcomes. They stand in for intermittently encountered uplinks: their Internet
    connectivity was NOT measured by the original collection. Four allocation
    strata are ranks of contact counts in the completed calibration prefix, not
    inferred personal/social attributes. No messages precede that prefix.
    """
    if not 0 < message_probability <= 1 or not 0 < warmup_fraction < .5 or not 0 < collector_fraction < .5:
        raise ValueError("invalid empirical workload fractions")
    if ttl_seconds < dataset.step_seconds or ttl_seconds % dataset.step_seconds:
        raise ValueError("TTL must be a positive multiple of native time resolution")
    role_rng, traffic_rng = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
    collector_count = max(2, int(np.ceil(len(dataset.node_ids) * collector_fraction)))
    collectors = set(map(int, role_rng.choice(dataset.node_ids, collector_count, replace=False)))
    mobiles = np.array([node for node in dataset.node_ids if node not in collectors], dtype=int)
    if len(mobiles) < base.groups:
        raise ValueError("not enough noncollector participants for allocation strata")
    mapping = {int(node): index for index, node in enumerate(mobiles)}
    warmup = int(np.ceil(dataset.steps * warmup_fraction))
    ttl = ttl_seconds // dataset.step_seconds
    if dataset.steps - ttl <= warmup:
        raise ValueError("trace too short for full TTL following calibration")
    config = replace(base, nodes=len(mobiles), steps=dataset.steps, ttl_steps=ttl,
                     message_probability=message_probability, peer_copy_available_next_epoch=True)
    config.validate()
    gateways = [set() for _ in range(dataset.steps)]
    peers = [set() for _ in range(dataset.steps)]
    counts = np.zeros(len(mobiles), dtype=int)
    first_presence = np.full(len(mobiles), dataset.steps, dtype=int)
    for slot, raw_left, raw_right in dataset.events:
        slot, raw_left, raw_right = int(slot), int(raw_left), int(raw_right)
        for node in (raw_left, raw_right):
            if node in mapping:
                index = mapping[node]
                first_presence[index] = min(first_presence[index], slot)
                if slot < warmup:
                    counts[index] += 1
        if raw_left in collectors and raw_right in mapping:
            gateways[slot].add(mapping[raw_right])
        elif raw_right in collectors and raw_left in mapping:
            gateways[slot].add(mapping[raw_left])
        elif raw_left in mapping and raw_right in mapping:
            peers[slot].add(tuple(sorted((mapping[raw_left], mapping[raw_right]))))
    ordering = np.lexsort((mobiles, counts))
    groups = np.empty(len(mobiles), dtype=int)
    groups[ordering] = np.minimum(base.groups - 1, np.arange(len(mobiles)) * base.groups // len(mobiles))
    messages = []
    for slot in range(warmup, dataset.steps - ttl):
        # Start a source only after its first observed contact. Silence after that
        # is retained as disconnection; do not use future last-seen times to drop
        # difficult workloads. Stop generation in time for every TTL to be observed.
        active = np.flatnonzero((traffic_rng.random(len(mobiles)) < message_probability)
                               & (first_presence <= slot))
        for source in active:
            messages.append(Message(len(messages), int(source), int(groups[source]), slot, slot + ttl))
    trace = DTNTrace(seed, tuple(map(int, groups)), tuple(messages),
                     tuple(tuple(sorted(nodes)) for nodes in gateways),
                     tuple(tuple(sorted(pairs)) for pairs in peers))
    metadata = {
        "dataset": dataset.name, "source_sha256": dataset.source_sha256,
        "raw_rows": dataset.raw_rows, "canonical_contact_rows": len(dataset.events),
        "duplicate_contact_rows_removed": dataset.duplicate_rows_removed,
        "registered_roster_size": len(dataset.node_ids), "mobile_sources": len(mobiles),
        "collector_ids": sorted(collectors), "collector_role": "synthetic mobile uplink designation, not measured Internet availability",
        "native_slot_seconds": dataset.step_seconds, "steps": dataset.steps,
        "source_time_origin_seconds": dataset.time_origin_seconds,
        "calibration_prefix_epochs": warmup, "last_generation_epoch_exclusive": dataset.steps - ttl,
        "full_record_deadlines_observable": all(message.deadline < dataset.steps for message in messages),
        "strata": "equal-size ranks of completed-prefix contact-slot counts; ties by anonymous roster ID",
        "group_sizes": np.bincount(groups, minlength=base.groups).tolist(),
        "public_prefix_contact_count_range": [int(counts.min()), int(counts.max())],
        "calendar_silence_retained": True, "full_trace_horizon": True,
        "gateway_contacts": sum(len(nodes) for nodes in trace.gateway_contacts),
        "peer_contacts": sum(len(pairs) for pairs in trace.peer_contacts),
        "transfer_semantics": "observed slot end; gateway phase before peers; new peer copies usable in following slot",
        "multiple_collectors_same_slot": "one uplink quota per mobile per slot, uniformly for all policies",
        "traffic": "synthetic Bernoulli sensor-report workload; not observed packet traffic",
    }
    return trace, config, metadata

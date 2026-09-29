from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import source_tree_digest

POLICIES = (
    "direct",
    "ttl_one_copy",
    "prophet_gtmx",
    "binary_spray_wait",
    "epidemic_cap",
    "airproof_deficit",
    "airproof_deadline",
)


@dataclass(frozen=True)
class DTNConfig:
    nodes: int = 120
    groups: int = 4
    steps: int = 168
    message_probability: float = 0.50
    record_size_bytes: int = 512
    ttl_steps: int = 24
    bytes_per_contact: int = 1024
    queue_records: int = 48
    copy_budget: int = 4
    peer_contact_rate: float = 0.35
    regular_gateway_availability: float = 0.45
    underserved_gateway_availability: float = 0.18
    gateway_outage_start_probability: float = 0.08
    prophet_p_encounter: float = 0.7
    prophet_p_first: float = 0.5
    prophet_p_first_threshold: float = 0.1
    prophet_beta: float = 0.9
    prophet_gamma: float = 0.999
    prophet_delta: float = 0.01
    deadline_deficit_weight: float = 3.0
    deadline_age_weight: float = 1.0
    deadline_slack_weight: float = 2.0
    peer_copy_available_next_epoch: bool = False

    def validate(self) -> None:
        if self.nodes < self.groups or self.groups < 2 or self.steps < 2:
            raise ValueError("invalid DTN population, groups, or horizon")
        if not 0 < self.message_probability <= 1 or not 0 <= self.peer_contact_rate <= 1:
            raise ValueError("probabilities must lie in their valid ranges")
        if self.record_size_bytes <= 0 or self.bytes_per_contact < self.record_size_bytes:
            raise ValueError("each contact must carry at least one record")
        if self.ttl_steps < 1 or self.queue_records < 1 or self.copy_budget < 1:
            raise ValueError("TTL, queue, and copy budget must be positive")
        if any(not np.isfinite(value) or value < 0 for value in (
            self.deadline_deficit_weight, self.deadline_age_weight, self.deadline_slack_weight
        )):
            raise ValueError("deadline-priority weights must be finite and nonnegative")
        for availability in (
            self.regular_gateway_availability,
            self.underserved_gateway_availability,
        ):
            if not 0 < availability < 1:
                raise ValueError("gateway availability must be strictly between zero and one")


@dataclass(frozen=True)
class Message:
    identifier: int
    source: int
    group: int
    created: int
    deadline: int


@dataclass(frozen=True)
class DTNTrace:
    seed: int
    node_groups: tuple[int, ...]
    messages: tuple[Message, ...]
    gateway_contacts: tuple[tuple[int, ...], ...]
    peer_contacts: tuple[tuple[tuple[int, int], ...], ...]


def prophet_encounter_update(
    old: float,
    *,
    p_first: float = 0.5,
    p_first_threshold: float = 0.1,
    p_encounter: float = 0.7,
    delta: float = 0.01,
) -> float:
    """RFC 6693 encounter update, including its first-encounter rule."""
    if old < p_first_threshold:
        return float(p_first)
    return float(old + (1.0 - delta - old) * p_encounter)


def prophet_transitive_update(old_ac: float, p_ab: float, p_bc: float, beta: float = 0.9) -> float:
    """RFC 6693 transitive update for one destination."""
    return float(max(old_ac, p_ab * p_bc * beta))


def _markov_gateway_series(
    steps: int,
    availability: float,
    outage_start_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    recovery = min(1.0, outage_start_probability * availability / (1.0 - availability))
    online = bool(rng.random() < availability)
    result = np.empty(steps, dtype=bool)
    for epoch in range(steps):
        result[epoch] = online
        if online and rng.random() < outage_start_probability:
            online = False
        elif not online and rng.random() < recovery:
            online = True
    return result


def generate_dtn_trace(config: DTNConfig, seed: int) -> DTNTrace:
    config.validate()
    sequence = np.random.SeedSequence(seed)
    gateway_rng, contact_rng, message_rng = [
        np.random.default_rng(child) for child in sequence.spawn(3)
    ]
    node_groups = np.arange(config.nodes, dtype=int) % config.groups
    gateway = np.empty((config.steps, config.nodes), dtype=bool)
    for node, group in enumerate(node_groups):
        availability = (
            config.underserved_gateway_availability
            if group == 0
            else config.regular_gateway_availability
        )
        gateway[:, node] = _markov_gateway_series(
            config.steps,
            availability,
            config.gateway_outage_start_probability,
            gateway_rng,
        )

    peer_contacts: list[tuple[tuple[int, int], ...]] = []
    pairs_per_epoch = round(config.nodes * config.peer_contact_rate / 2.0)
    for _ in range(config.steps):
        permutation = contact_rng.permutation(config.nodes)
        pairs = [
            (int(permutation[2 * index]), int(permutation[2 * index + 1]))
            for index in range(min(pairs_per_epoch, config.nodes // 2))
        ]
        peer_contacts.append(tuple(pairs))

    messages: list[Message] = []
    generation_horizon = max(1, config.steps - config.ttl_steps)
    for epoch in range(generation_horizon):
        participating = np.flatnonzero(message_rng.random(config.nodes) < config.message_probability)
        for source in participating:
            messages.append(
                Message(
                    identifier=len(messages),
                    source=int(source),
                    group=int(node_groups[source]),
                    created=epoch,
                    deadline=epoch + config.ttl_steps,
                )
            )
    return DTNTrace(
        seed=seed,
        node_groups=tuple(map(int, node_groups)),
        messages=tuple(messages),
        gateway_contacts=tuple(tuple(map(int, np.flatnonzero(row))) for row in gateway),
        peer_contacts=tuple(peer_contacts),
    )


def _group_deficits(
    generated_by_group: np.ndarray,
    delivered_by_group: np.ndarray,
) -> np.ndarray:
    rates = delivered_by_group / np.maximum(generated_by_group, 1.0)
    return np.max(rates) - rates


def _message_priority(
    message: Message,
    epoch: int,
    deficits: np.ndarray,
    ttl_steps: int,
) -> float:
    age = (epoch - message.created) / max(ttl_steps, 1)
    freshness = 1.0 - age
    urgency = 1.0 - (message.deadline - epoch) / max(ttl_steps, 1)
    return float(2.0 * deficits[message.group] + 1.5 * freshness + 0.75 * urgency)


class GatewayContactHistory:
    """Causal two-state contact hazard with independent Beta(1,1) smoothing.

    Each node observes its own gateway contact/no-contact epochs. At encounters it
    can exchange its two transition counts and current state. Only observed
    prefixes are used; this is a contact-opportunity model, not a guarantee that
    every queued message will be served. Queue contention is discounted below.
    """

    def __init__(self, nodes: int):
        if nodes < 1:
            raise ValueError("positive node count required")
        self.state = np.zeros(nodes, dtype=int)
        self.exposures = np.zeros((nodes, 2), dtype=int)
        self.successes = np.zeros((nodes, 2), dtype=int)
        self.observed = 0

    def observe(self, contacts: tuple[int, ...]) -> None:
        current = np.zeros(len(self.state), dtype=int)
        current[list(contacts)] = 1
        if self.observed:
            nodes = np.arange(len(self.state))
            self.exposures[nodes, self.state] += 1
            self.successes[nodes, self.state] += current
        self.state = current
        self.observed += 1

    def before_deadline(self, node: int, remaining_epochs: int) -> float:
        if remaining_epochs <= 0:
            return 0.0
        off_to_on, on_to_on = (1 + self.successes[node]) / (2 + self.exposures[node])
        first = on_to_on if self.state[node] else off_to_on
        return float(1 - (1 - first) * (1 - off_to_on)**(remaining_epochs - 1))


def deadline_priority(
    message: Message, epoch: int, deficits: np.ndarray, config: DTNConfig, *,
    delivery_probability: float, copies: int,
) -> float:
    """Deadline opportunity x value/copy cost plus remaining-slack urgency."""
    slack = max(0, message.deadline - epoch)
    age = max(0, epoch - message.created) / max(1, config.ttl_steps)
    value = (1 + config.deadline_deficit_weight * deficits[message.group]
             + config.deadline_age_weight * age)
    normalized_bytes = config.record_size_bytes / 512.
    return float(np.clip(delivery_probability, 0, 1) * value
                 / (normalized_bytes * (1 + max(1, copies)))
                 + config.deadline_slack_weight / (slack + 1))


def simulate_dtn_policy(trace: DTNTrace, config: DTNConfig, policy: str) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"unknown DTN policy {policy}")
    config.validate()
    messages = trace.messages
    by_epoch: list[list[int]] = [[] for _ in range(config.steps)]
    expiry_epoch: list[list[int]] = [[] for _ in range(config.steps)]
    for message in messages:
        by_epoch[message.created].append(message.identifier)
        if message.deadline + 1 < config.steps:
            expiry_epoch[message.deadline + 1].append(message.identifier)
    holders: list[set[int]] = [set() for _ in messages]
    node_queues: list[set[int]] = [set() for _ in range(config.nodes)]
    spray_tokens: dict[tuple[int, int], int] = {}
    available_from: dict[tuple[int, int], int] = {}
    delivered = np.full(len(messages), -1, dtype=int)
    generated_by_group = np.zeros(config.groups, dtype=int)
    delivered_by_group = np.zeros(config.groups, dtype=int)
    latest_delivered_acquisition = np.full(config.groups, -1, dtype=int)
    aoi_samples: list[float] = []
    peer_transmissions = 0
    gateway_transmissions = 0
    queue_rejections = 0
    max_copies = 0
    contact_capacity_records = config.bytes_per_contact // config.record_size_bytes

    p_gateway = np.zeros(config.nodes, dtype=float)
    p_peer = np.zeros((config.nodes, config.nodes), dtype=float)
    contact_history = GatewayContactHistory(config.nodes)

    def remove_holder(message_id: int, node: int) -> None:
        holders[message_id].discard(node)
        node_queues[node].discard(message_id)
        spray_tokens.pop((message_id, node), None)
        available_from.pop((message_id, node), None)

    def add_holder(message_id: int, node: int, *, tokens: int = 1, available_epoch: int = 0) -> bool:
        nonlocal queue_rejections, max_copies
        if node in holders[message_id]:
            return False
        if len(node_queues[node]) >= config.queue_records:
            queue_rejections += 1
            return False
        holders[message_id].add(node)
        node_queues[node].add(message_id)
        spray_tokens[(message_id, node)] = tokens
        available_from[(message_id, node)] = available_epoch
        max_copies = max(max_copies, len(holders[message_id]))
        return True

    def eligible(node: int, epoch: int) -> list[int]:
        return [
            message_id
            for message_id in node_queues[node]
            if delivered[message_id] < 0
            and messages[message_id].created <= epoch <= messages[message_id].deadline
            and available_from[(message_id, node)] <= epoch
        ]

    for epoch in range(config.steps):
        # Gateway phase precedes peer phase; a peer copy at e can next reach a
        # gateway at e+1, not through a contact that already happened at e.
        contact_history.observe(trace.gateway_contacts[epoch])
        p_gateway *= config.prophet_gamma
        p_peer *= config.prophet_gamma
        for message_id in by_epoch[epoch]:
            message = messages[message_id]
            generated_by_group[message.group] += 1
            if not add_holder(
                message_id,
                message.source,
                tokens=config.copy_budget if policy == "binary_spray_wait" else 1,
                available_epoch=epoch,
            ):
                continue

        # Each record expires once. Scanning every future/past record at every
        # 20-second empirical epoch would add quadratic work without changing state.
        for message_id in expiry_epoch[epoch]:
            if delivered[message_id] < 0:
                for node in tuple(holders[message_id]):
                    remove_holder(message_id, node)

        deficits = _group_deficits(generated_by_group, delivered_by_group)
        for node in trace.gateway_contacts[epoch]:
            p_gateway[node] = prophet_encounter_update(
                p_gateway[node],
                p_first=config.prophet_p_first,
                p_first_threshold=config.prophet_p_first_threshold,
                p_encounter=config.prophet_p_encounter,
                delta=config.prophet_delta,
            )
            queue = eligible(node, epoch)
            if policy == "airproof_deficit":
                queue.sort(
                    key=lambda item: _message_priority(
                        messages[item], epoch, deficits, config.ttl_steps
                    ),
                    reverse=True,
                )
            elif policy == "airproof_deadline":
                queue.sort(key=lambda item: (deadline_priority(
                    messages[item], epoch, deficits, config, delivery_probability=1.,
                    copies=len(holders[item])), -item), reverse=True)
            else:
                queue.sort(key=lambda item: (messages[item].deadline, messages[item].created))
            for message_id in queue[:contact_capacity_records]:
                message = messages[message_id]
                delivered[message_id] = epoch
                delivered_by_group[message.group] += 1
                latest_delivered_acquisition[message.group] = max(
                    latest_delivered_acquisition[message.group], message.created
                )
                gateway_transmissions += 1
                for holder in tuple(holders[message_id]):
                    remove_holder(message_id, holder)

        for left, right in trace.peer_contacts[epoch]:
            old_left_gateway = p_gateway[left]
            old_right_gateway = p_gateway[right]
            p_peer[left, right] = prophet_encounter_update(
                p_peer[left, right],
                p_first=config.prophet_p_first,
                p_first_threshold=config.prophet_p_first_threshold,
                p_encounter=config.prophet_p_encounter,
                delta=config.prophet_delta,
            )
            p_peer[right, left] = prophet_encounter_update(
                p_peer[right, left],
                p_first=config.prophet_p_first,
                p_first_threshold=config.prophet_p_first_threshold,
                p_encounter=config.prophet_p_encounter,
                delta=config.prophet_delta,
            )
            p_gateway[left] = prophet_transitive_update(
                old_left_gateway,
                p_peer[left, right],
                old_right_gateway,
                config.prophet_beta,
            )
            p_gateway[right] = prophet_transitive_update(
                old_right_gateway,
                p_peer[right, left],
                old_left_gateway,
                config.prophet_beta,
            )
            if policy == "direct":
                continue
            for sender, receiver in ((left, right), (right, left)):
                queue = eligible(sender, epoch)
                if policy == "prophet_gtmx":
                    queue.sort(
                        key=lambda item: (p_gateway[receiver] - p_gateway[sender], -messages[item].deadline),
                        reverse=True,
                    )
                elif policy == "airproof_deficit":
                    queue.sort(
                        key=lambda item: _message_priority(
                            messages[item], epoch, deficits, config.ttl_steps
                        )
                        + p_gateway[receiver]
                        - p_gateway[sender],
                        reverse=True,
                    )
                elif policy == "airproof_deadline":
                    def receiver_priority(item):
                        remaining = messages[item].deadline - epoch
                        probability = contact_history.before_deadline(receiver, remaining)
                        # A finite queue competes for the same future gateway slots.
                        probability *= min(1., contact_capacity_records * max(remaining, 0)
                                           / max(1, len(node_queues[receiver]) + 1))
                        return (deadline_priority(messages[item], epoch, deficits, config,
                                                  delivery_probability=probability,
                                                  copies=len(holders[item])), -item)
                    queue.sort(key=receiver_priority, reverse=True)
                else:
                    queue.sort(key=lambda item: (messages[item].deadline, messages[item].created))

                sent = 0
                for message_id in queue:
                    if sent >= contact_capacity_records or receiver in holders[message_id]:
                        continue
                    message = messages[message_id]
                    should_send = False
                    remove_sender = False
                    receiver_tokens = 1
                    if policy == "ttl_one_copy":
                        should_send = p_gateway[receiver] > p_gateway[sender]
                        remove_sender = should_send
                    elif policy == "prophet_gtmx":
                        should_send = (
                            p_gateway[receiver] > p_gateway[sender]
                            and len(holders[message_id]) < config.copy_budget
                        )
                    elif policy == "binary_spray_wait":
                        sender_tokens = spray_tokens.get((message_id, sender), 1)
                        should_send = sender_tokens > 1
                        receiver_tokens = sender_tokens // 2
                    elif policy == "epidemic_cap":
                        should_send = len(holders[message_id]) < config.copy_budget
                    elif policy == "airproof_deficit":
                        urgent = message.deadline - epoch <= max(2, config.ttl_steps // 4)
                        should_send = len(holders[message_id]) < config.copy_budget and (
                            p_gateway[receiver] > p_gateway[sender] or urgent
                        )
                    elif policy == "airproof_deadline":
                        # Use available copies early instead of waiting until expiry
                        # for a higher PRoPHET score. All policies retain the same
                        # per-direction bytes, buffer and maximum-copy constraints.
                        should_send = (len(holders[message_id]) < config.copy_budget
                                       and message.deadline > epoch)
                    if not should_send:
                        continue
                    if add_holder(message_id, receiver, tokens=receiver_tokens,
                                  available_epoch=epoch + int(config.peer_copy_available_next_epoch)):
                        sent += 1
                        peer_transmissions += 1
                        if policy == "binary_spray_wait":
                            sender_tokens = spray_tokens[(message_id, sender)]
                            spray_tokens[(message_id, sender)] = sender_tokens - receiver_tokens
                        if remove_sender:
                            remove_holder(message_id, sender)

        for group in range(config.groups):
            latest = latest_delivered_acquisition[group]
            aoi_samples.append(float(epoch - latest if latest >= 0 else epoch + 1))

    delays = np.asarray(
        [
            delivered[index] - message.created
            if delivered[index] >= 0
            else message.deadline - message.created + 1
            for index, message in enumerate(messages)
        ],
        dtype=float,
    )
    delivered_mask = delivered >= 0
    group_delivery = []
    for group in range(config.groups):
        group_mask = np.asarray([message.group == group for message in messages], dtype=bool)
        group_delivery.append(float(delivered_mask[group_mask].mean()) if group_mask.any() else 0.0)
    total_transmissions = peer_transmissions + gateway_transmissions
    return {
        "policy": policy,
        "seed": trace.seed,
        "generated_records": len(messages),
        "delivered_records": int(delivered_mask.sum()),
        "deadline_delivery_ratio": float(delivered_mask.mean()) if len(messages) else 0.0,
        "restricted_mean_delivery_delay": float(delays.mean()) if len(delays) else 0.0,
        "restricted_p95_delivery_delay": float(np.percentile(delays, 95)) if len(delays) else 0.0,
        "restricted_mean_aoi": float(np.mean(aoi_samples)),
        "peer_transmissions": peer_transmissions,
        "gateway_transmissions": gateway_transmissions,
        "transmitted_bytes": total_transmissions * config.record_size_bytes,
        "transmissions_per_delivered": float(
            total_transmissions / max(int(delivered_mask.sum()), 1)
        ),
        "queue_rejections": queue_rejections,
        "maximum_realized_copies": max_copies,
        "group_delivery_ratios": group_delivery,
        "group_delivery_gap": float(max(group_delivery) - min(group_delivery)),
    }


def run_dtn_benchmark(
    config: DTNConfig,
    seeds: list[int],
    policies: tuple[str, ...] = POLICIES,
) -> dict[str, Any]:
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"unknown DTN policies {unknown}")
    rows = []
    trace_hashes: dict[str, str] = {}
    for seed in seeds:
        trace = generate_dtn_trace(config, seed)
        canonical_trace = json.dumps(
            {
                "seed": trace.seed,
                "node_groups": trace.node_groups,
                "messages": [asdict(message) for message in trace.messages],
                "gateway_contacts": trace.gateway_contacts,
                "peer_contacts": trace.peer_contacts,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        trace_hashes[str(seed)] = hashlib.sha256(canonical_trace.encode()).hexdigest()
        rows.extend(simulate_dtn_policy(trace, config, policy) for policy in policies)
    report: dict[str, Any] = {
        "status": "capacity-constrained-equal-resource-benchmark",
        "protocol": {
            "common_trace_per_seed": True,
            "undelivered_records_censored_at_ttl_plus_one": True,
            "equal_bytes_per_contact": config.bytes_per_contact,
            "equal_queue_records": config.queue_records,
            "equal_maximum_copy_budget": config.copy_budget,
            "peer_capacity_scope": "bytes_per_contact per direction, identical for all policies",
            "acknowledgment_model": "ideal common delivery acknowledgments and copy-state bookkeeping",
            "airproof_deadline_probability": "online two-state Beta-smoothed gateway hazard with queue-contention discount",
            "peer_copy_available_next_epoch": config.peer_copy_available_next_epoch,
            "prophet_variant": "RFC-6693 equations 1-3 with GTMX copy cap",
            "spray_variant": "binary Spray-and-Wait",
            "epidemic_variant": "copy-budget-capped adaptation",
        },
        "config": asdict(config),
        "source_tree_sha256": source_tree_digest(),
        "trace_sha256": trace_hashes,
        "rows": rows,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report


def write_dtn_benchmark(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path

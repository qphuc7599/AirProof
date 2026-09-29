from __future__ import annotations

import pytest

from airproof.dtn import (
    POLICIES,
    DTNConfig,
    generate_dtn_trace,
    prophet_encounter_update,
    prophet_transitive_update,
    run_dtn_benchmark,
    simulate_dtn_policy,
    GatewayContactHistory,
    Message,
    deadline_priority,
    DTNTrace,
)


def test_prophet_updates_follow_rfc_equations() -> None:
    assert prophet_encounter_update(0.0) == pytest.approx(0.5)
    assert prophet_encounter_update(0.5) == pytest.approx(0.843)
    assert prophet_transitive_update(0.1, 0.8, 0.7) == pytest.approx(0.504)


def test_trace_is_deterministic_and_preserves_full_ttl() -> None:
    config = DTNConfig(nodes=12, steps=20, ttl_steps=4, message_probability=0.5)
    first = generate_dtn_trace(config, 7)
    second = generate_dtn_trace(config, 7)
    assert first == second
    assert first.messages
    assert all(message.deadline - message.created == 4 for message in first.messages)
    assert max(message.deadline for message in first.messages) < config.steps


@pytest.mark.parametrize("policy", POLICIES)
def test_all_dtn_policies_obey_capacity_and_copy_bounds(policy: str) -> None:
    config = DTNConfig(
        nodes=16,
        steps=24,
        ttl_steps=6,
        message_probability=0.35,
        queue_records=12,
        copy_budget=3,
        bytes_per_contact=1024,
    )
    row = simulate_dtn_policy(generate_dtn_trace(config, 11), config, policy)
    assert 0 <= row["deadline_delivery_ratio"] <= 1
    assert row["maximum_realized_copies"] <= config.copy_budget
    assert row["transmitted_bytes"] % config.record_size_bytes == 0
    assert len(row["group_delivery_ratios"]) == config.groups


def test_benchmark_reuses_one_trace_per_seed() -> None:
    config = DTNConfig(nodes=12, steps=20, ttl_steps=4, message_probability=0.25)
    report = run_dtn_benchmark(config, [3, 4], ("direct", "binary_spray_wait"))
    assert len(report["rows"]) == 4
    by_seed = {}
    for row in report["rows"]:
        by_seed.setdefault(row["seed"], set()).add(row["generated_records"])
    assert all(len(counts) == 1 for counts in by_seed.values())
    assert report["protocol"]["common_trace_per_seed"] is True


def test_contact_hazard_updates_only_with_observed_transitions() -> None:
    history = GatewayContactHistory(2)
    history.observe(())
    assert history.before_deadline(0, 2) == pytest.approx(.75)
    history.observe(())
    # One observed 0->0 transition: p01=1/3, p11=1/2.
    assert history.before_deadline(0, 2) == pytest.approx(1 - (2/3)**2)
    history.observe((0,))
    # p01=2/4, current state on, no 1->? observation yet.
    assert history.before_deadline(0, 2) == pytest.approx(.75)
    assert history.before_deadline(0, 0) == 0
    assert history.before_deadline(0, 12) >= history.before_deadline(0, 2)
    assert history.before_deadline(1, 2) < history.before_deadline(0, 2)


def test_deadline_priority_responds_to_deadline_probability_deficit_and_copies() -> None:
    import numpy as np
    cfg = DTNConfig()
    message = Message(0, 0, 0, 0, 24)
    low = deadline_priority(message, 12, np.zeros(4), cfg, delivery_probability=.1, copies=1)
    high = deadline_priority(message, 12, np.zeros(4), cfg, delivery_probability=.9, copies=1)
    assert high > low
    assert deadline_priority(message, 12, np.array([.4, 0, 0, 0]), cfg,
                             delivery_probability=.9, copies=1) > high
    assert deadline_priority(message, 12, np.zeros(4), cfg,
                             delivery_probability=.9, copies=4) < high
    assert deadline_priority(message, 23, np.zeros(4), cfg,
                             delivery_probability=.9, copies=1) > high


def test_slotted_replay_cannot_chain_new_copies_within_one_observed_slot() -> None:
    from dataclasses import replace
    cfg = DTNConfig(nodes=4, groups=2, steps=4, ttl_steps=2, copy_budget=3)
    trace = DTNTrace(0, (0, 1, 0, 1), (Message(0, 0, 0, 0, 2),),
                     ((), (2,), (2,), ()), (((0, 1), (1, 2)), ((1, 2),), (), ()))
    ordinary = simulate_dtn_policy(trace, cfg, "epidemic_cap")
    slotted = simulate_dtn_policy(trace, replace(cfg, peer_copy_available_next_epoch=True), "epidemic_cap")
    assert ordinary["restricted_mean_delivery_delay"] == 1
    assert slotted["restricted_mean_delivery_delay"] == 2
    assert slotted["delivered_records"] == ordinary["delivered_records"] == 1

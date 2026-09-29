import numpy as np
import pytest

from airproof.contact_replay import build_contact_replay, load_contacts
from airproof.dtn import DTNConfig, simulate_dtn_policy


def table(tmp_path):
    path = tmp_path / "contact.dat"
    rows = []
    for t in range(80):
        for node in range(12):
            rows.append(f"{20 * (t + 1)} {100 + node} {100 + (node + 1) % 12}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_contact_replay_preserves_contacts_and_full_deadlines(tmp_path):
    data = load_contacts(table(tmp_path), name="fixture")
    trace, config, metadata = build_contact_replay(data, seed=15, ttl_seconds=120,
                                                  message_probability=.1)
    second, _, _ = build_contact_replay(data, seed=15, ttl_seconds=120, message_probability=.1)
    assert trace == second
    assert config.steps == 80 and config.peer_copy_available_next_epoch
    assert metadata["full_record_deadlines_observable"]
    assert all(message.created >= 16 and message.deadline < 80 for message in trace.messages)
    assert all(message.deadline - message.created == 6 for message in trace.messages)
    assert sum(metadata["group_sizes"]) == config.nodes
    row = simulate_dtn_policy(trace, config, "airproof_deadline")
    assert row["maximum_realized_copies"] <= config.copy_budget
    assert 0 <= row["deadline_delivery_ratio"] <= 1


def test_duplicate_sightings_do_not_double_capacity(tmp_path):
    path = tmp_path / "duplicates.dat"
    path.write_text("20 1 2\n20 2 1\n20 1 2\n40 1 3\n", encoding="utf-8")
    data = load_contacts(path, name="fixture")
    assert data.raw_rows == 4
    assert data.duplicate_rows_removed == 2
    np.testing.assert_array_equal(data.events, [[0, 1, 2], [1, 1, 3]])
    with pytest.raises(ValueError, match="hash"):
        load_contacts(path, name="fixture", expected_sha256="0" * 64)


def test_strata_use_prefix_only_and_sources_wait_for_first_presence(tmp_path):
    from dataclasses import replace
    data = load_contacts(table(tmp_path), name="fixture")
    changed = data.events.copy()
    changed[changed[:, 0] >= 16, 1:] = np.array([100, 101])
    a, _, _ = build_contact_replay(data, seed=15, ttl_seconds=120, message_probability=.1)
    b, _, _ = build_contact_replay(replace(data, events=changed), seed=15,
                                   ttl_seconds=120, message_probability=.1)
    assert a.node_groups == b.node_groups
    assert a.messages == b.messages  # all sources were observed in the same prefix
    assert a.peer_contacts != b.peer_contacts


def test_reject_temporal_resolution_or_insufficient_horizon(tmp_path):
    data = load_contacts(table(tmp_path), name="fixture")
    with pytest.raises(ValueError, match="too short"):
        build_contact_replay(data, seed=1, ttl_seconds=3600)
    path = tmp_path / "off_grid.dat"
    path.write_text("20 1 2\n39 1 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="20-second"):
        load_contacts(path, name="fixture")

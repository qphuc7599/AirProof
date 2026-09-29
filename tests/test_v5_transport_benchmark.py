from types import SimpleNamespace

import numpy as np
import pytest

from airproof.contact_replay import ContactDataset
from airproof.records import Observation
from scripts.benchmark_v5_transport import (RESOURCES, adapt_replay,
                                          analyze_transport_family, delivery_metrics)


def test_native_ttl_conversion_and_resource_partition():
    events = np.array([[slot, a, a+1] for slot in (0, 100, 1200, 2399) for a in range(7)])
    dataset = ContactDataset("unit", events, np.arange(8), 2400, 0, 20, "unit", len(events), 0)
    manifest = dict(groups=4, resources=RESOURCES, warmup_fraction=.2)
    cell = dict(ttl_hours=6, probability_per_native_slot=.002, collector_fraction=.05)
    trace, records, cfg, metadata = adapt_replay(dataset, seed=123, cell=cell, manifest=manifest)
    assert trace.drain_epochs == cfg["transport"]["ttl_epochs"] == 1080
    assert trace.acquisition_epochs == 1320
    assert trace.acquisition_epochs + trace.drain_epochs == 2400
    assert cfg["transport"]["release_buffer_bytes"] == 6144
    assert cfg["transport"]["release_gateway_reservation_bytes"] == 256
    assert cfg["transport"]["capacity_bytes_per_direction"] == 1024
    assert all(record.epoch + 1080 < 2400 for record in records)
    assert metadata["full_record_deadlines_observable"]
    assert all(record.direct_arrival is None and record.relay_arrival is None for record in records)


def test_restricted_delay_includes_undelivered_and_group_support():
    records = [Observation(i, 2, i, i, 10., 1., 1., 512, str(i), None, None) for i in range(2)]
    result = SimpleNamespace(raw_arrivals={"0": 4})
    metrics = delivery_metrics(records, result, groups=2, ttl_slots=6, slot_seconds=20)
    assert metrics["deadline_delivery_ratio"] == .5
    assert metrics["restricted_mean_delay_seconds"] == 80.
    assert metrics["delivered_mean_delay_seconds"] == 40.
    assert metrics["group_delivery_gap"] == 1.
    unsupported = delivery_metrics(records[:1], result, groups=2, ttl_slots=6, slot_seconds=20)
    assert unsupported["group_delivery_gap"] is None


def sample_rows():
    rows = []
    for seed in range(3):
        for cell in ("first", "second"):
            for policy, delivery, delay, gap in (("airproof_deadline", .8+seed*.001, 50.+seed, .1+seed*.001), ("binary_spray_wait", .6, 100., .3), ("epidemic_cap", .8, 100., .3)):
                rows.append(dict(seed=seed, cell=cell, policy=policy, metrics=dict(
                    deadline_delivery_ratio=delivery, restricted_mean_delay_seconds=delay, group_delivery_gap=gap)))
    return rows


def test_paired_family_uses_seeds_not_cells_and_frozen_margins():
    rows = sample_rows()
    result = analyze_transport_family(rows, seeds=[0, 1, 2], cells=["first", "second"])
    assert result["paired_units"] == 3
    assert result["repeated_cells_per_unit"] == 2
    assert result["tests"]["D1"]["mean_contrast"] == pytest.approx(-.021)
    assert result["tests"]["D2"]["mean_contrast"] == pytest.approx(-34.)
    assert result["tests"]["D3"]["mean_contrast"] == pytest.approx(-.139)
    assert not result["family_pass"]  # conditional role/traffic is not population confirmation
    assert analyze_transport_family(rows, seeds=[0, 1, 2], cells=["first", "second"], independent_worlds=True)["family_pass"]


def test_paired_family_rejects_missing_duplicate_and_undefined_support():
    with pytest.raises(ValueError, match="complete"):
        analyze_transport_family(sample_rows()[:-1], seeds=[0, 1, 2], cells=["first", "second"])
    rows = sample_rows()
    with pytest.raises(ValueError, match="duplicate"):
        analyze_transport_family(rows + rows[:1], seeds=[0, 1, 2], cells=["first", "second"])
    rows[0]["metrics"]["group_delivery_gap"] = None
    with pytest.raises(ValueError, match="support"):
        analyze_transport_family(rows, seeds=[0, 1, 2], cells=["first", "second"])


def test_negative_zero_variance_is_invalid():
    rows=sample_rows()
    for row in rows:
        if row["policy"]=="airproof_deadline":
            row["metrics"].update(deadline_delivery_ratio=.8,restricted_mean_delay_seconds=50.,group_delivery_gap=.1)
    result=analyze_transport_family(rows,seeds=[0,1,2],cells=["first","second"])
    assert not result["family_valid"]
    assert all(not t["valid"] and t["one_sided_p"] is None and not t["conditional_threshold_met"] for t in result["tests"].values())

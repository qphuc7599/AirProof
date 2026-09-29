from pathlib import Path

import pytest

from airproof.config import load_config, with_overrides
from airproof.experiment import run_method
from airproof.simulator import generate_world


@pytest.mark.parametrize("method", ["airproof_v4", "squared_loss_v4"])
def test_disabling_allocation_profiler_preserves_every_scientific_metric(method):
    root = Path(__file__).resolve().parents[1]
    config = with_overrides(load_config(root / "configs/v4/public_reference_validation.yaml"), {
        "world.agents": 30, "world.grid_side": 4, "world.steps": 12, "world.burn_in_steps": 4,
        "world.reference_station_count": 4, "twin.reference_calibration_epochs": 4,
        "twin.fixed_lag": 2, "privacy.k_min": 2, "attack.start_epoch": 6,
        "attack.kind": "adversarial_drift",
    })
    world = generate_world(config, 7451)
    profiled = run_method(world, config, method)["metrics"]
    unprofiled = run_method(world, with_overrides(config, {"execution.trace_python_allocations": False}), method)["metrics"]
    instrumentation = {"runtime_seconds", "peak_memory_mb", "python_peak_alloc_mb", "python_allocation_tracing_enabled",
        "epoch_update_p50_seconds", "epoch_update_p95_seconds", "epoch_update_max_seconds", "intersectional_audit_seconds"}
    assert profiled.keys() == unprofiled.keys()
    for name in profiled:
        if name not in instrumentation:
            assert profiled[name] == unprofiled[name], name
    assert profiled["python_peak_alloc_mb"] > 0
    assert unprofiled["python_peak_alloc_mb"] is None
    assert unprofiled["peak_memory_mb"] > 0
    assert not unprofiled["python_allocation_tracing_enabled"]

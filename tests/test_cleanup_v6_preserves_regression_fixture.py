"""The cleanup allow-list must retain the executable v5 wire baseline."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tmp/v5_transport_before_wire.py"
SOURCE = (
    ROOT
    / "reports/v5/development/integrated_911000_911007_v1/source_snapshot"
    / "airproof/v5_transport.py"
)
EXPECTED_SHA256 = "43e9f35b3b18b2a3b43caebac51f880ad03048c6421c08cd89ceadce75fe399d"


def _load_cleanup_module():
    path = ROOT / "scripts/cleanup_v6_obsolete_artifacts.py"
    spec = importlib.util.spec_from_file_location("cleanup_v6_obsolete_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cleanup_preserves_exact_pre_wire_regression_fixture():
    cleanup = _load_cleanup_module()
    relative = "tmp/v5_transport_before_wire.py"
    assert relative not in cleanup.EXPLICIT
    assert relative in cleanup.PRESERVED
    assert FIXTURE.read_bytes() == SOURCE.read_bytes()
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == EXPECTED_SHA256

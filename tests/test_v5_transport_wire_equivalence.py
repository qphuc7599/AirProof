"""Mechanical wire integration against the preserved pre-integration module."""
import importlib.util
from pathlib import Path
import random
import sys

import pytest

from airproof.records import Observation
from airproof.v5_transport import TransportTrace, simulate_transport


@pytest.fixture(scope="module")
def previous():
    path = Path(__file__).resolve().parents[1] / "tmp/v5_transport_before_wire.py"
    spec = importlib.util.spec_from_file_location("airproof._v5_transport_before_wire", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("policy", ["direct", "airproof_deadline", "binary_spray_wait", "epidemic_cap"])
@pytest.mark.parametrize("scenario", [0, 1, 2])
def test_actual_frames_preserve_arrivals_and_every_old_metric(previous, policy, scenario):
    rng = random.Random(1024 + scenario)
    nodes, acquisition, drain = 5, 8, 4
    gateways = tuple(tuple(n for n in range(nodes) if rng.random() < .35)
                     for _ in range(acquisition + drain))
    peers = tuple(((0, 1), (1, 2), (2, 3), (3, 4), (0, 4))
                  for _ in range(acquisition + drain))
    trace = TransportTrace(1024 + scenario, nodes, acquisition, drain,
                           (0, 1, 0, 1, 0), gateways, peers, "equivalence")
    records = [Observation(n, epoch, n, n % 2, 10., 1., 1., 512,
                           f"{n}:{epoch}:{copy}", 999, 888)
               for epoch in range(acquisition) for n in range(nodes)
               for copy in range(1 + scenario)]
    config = {"transport": {"ttl_epochs": 2 + scenario,
                             "raw_buffer_bytes": 1024 if scenario else 18432,
                             "release_buffer_bytes": 512 if scenario else 6144}}
    before = previous.simulate_transport(trace, records, config, policy)
    after = simulate_transport(trace, records, config, policy)
    assert after.raw_arrivals == before.raw_arrivals
    assert after.release_arrivals == before.release_arrivals
    assert after.metrics == before.metrics
    assert not after.metrics["token_violations"]


def test_charged_frames_are_actually_encoded(monkeypatch):
    import airproof.v5_transport as transport
    calls = dict(header=0, ack=0, history=0)
    for name, label in (("encode_header", "header"), ("encode_ack", "ack"),
                        ("encode_gateway_history", "history")):
        original = getattr(transport, name)
        def wrapper(*args, _original=original, _label=label, **kwargs):
            calls[_label] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(transport, name, wrapper)
    trace = TransportTrace(1, 2, 1, 1, (0, 1), ((0,), (1,)), (((0, 1),), ()), "frames")
    record = Observation(0, 0, 0, 0, 10., 1., 1., 512, "opaque", None, None)
    result = transport.simulate_transport(trace, [record], {})
    assert calls["header"] == calls["ack"] == 2  # one raw and one release
    assert calls["history"] == 2
    assert result.metrics["control_bytes"] == calls["header"]*16 + calls["ack"]*16 + calls["history"]*64

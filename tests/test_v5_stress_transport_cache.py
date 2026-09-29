import ast
from dataclasses import replace
import importlib.util
import inspect
from pathlib import Path
import pytest
from airproof.records import Observation
from airproof.v5_transport import TransportTrace, simulate_transport

spec = importlib.util.spec_from_file_location("stress_cache", Path(__file__).resolve().parents[1]/"reports/v5/tools/stress_transport_cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("policy", ["direct", "binary_spray_wait", "epidemic_cap", "airproof_deadline"])
def test_payload_mutation_reuses_identical_real_transport_but_dependencies_invalidate(policy):
    record = Observation(0, 0, 0, 0, 10., 1., 1., 512, "same", None, None)
    trace = TransportTrace(1, 2, 1, 2, (0, 1), ((), (), (1,)), (((0, 1),), ((0, 1),), ()), "fixture")
    cache = module.TransportCache(simulate_transport, "source1")
    first = cache(trace, [record], {}, policy)
    changed = replace(record, value=-1e6, sigma=99., quality=.001, corrupted=True)
    actual = cache(trace, [changed], {}, policy)
    assert actual == simulate_transport(trace, [changed], {}, policy=policy)
    assert first == actual and cache.hits == 1 and cache.misses == 1
    first.metrics["control_bytes"] = -1
    assert cache(trace, [changed], {}, policy).metrics["control_bytes"] >= 0
    cache(trace, [replace(changed, group=1)], {}, policy)
    cache(trace, [changed], {"transport": {"capacity_bytes_per_direction": 512}}, policy)
    cache(replace(trace, gateway_contacts=((0,), (), (1,))), [changed], {}, policy)
    assert cache.misses == 4


def test_source_record_dependencies_match_cache_audit():
    tree = ast.parse(inspect.getsource(simulate_transport))
    fields = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
              and isinstance(n.value, ast.Name) and n.value.id == "r"}
    assert fields == {"nullifier", "user_id", "epoch", "group"}

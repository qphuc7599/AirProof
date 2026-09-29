"""Campaign bookkeeping/gates only: no world simulation or process pool."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_v5_campaign as analyzer
import lock_v5_confirmation as locker
import run_v5_campaign as runner
from airproof.config import config_hash
from airproof.statistics import holm_adjust
from airproof.v5_estimator import EstimatorConfig
from airproof.v5_experiment import cell_configuration, development_specs, method_specs

CELLS = ["anchor_clean", "outage_clean", "severe_clean", "severe_drift", "severe_hotspot"]


def mock_rows(seeds, stage):
    rows = []
    for seed in seeds:
        for cell in CELLS:
            specs = development_specs() if stage == "development" else method_specs(EstimatorConfig(), cell)
            for spec in specs:
                control = spec["method"].startswith("SQ")
                attack = cell in ("severe_drift", "severe_hotspot")
                rmse = (2. if control else 1.4) if attack else 1.
                rows.append(dict(seed=seed, stage=stage, cell=cell, method=spec["method"],
                    estimator=asdict(spec["estimator"]) if spec["estimator"] else None,
                    metrics=dict(rmse=rmse, event_recall=.9, solver_failure_rate=0.,
                                 feasible_floor_violations=0, coverage_gap=.2, worst_group_rmse=1.2),
                    transport=dict(token_violations=0), elapsed_seconds=.1))
    return rows


def indexed(rows):
    return {(r["seed"], r["cell"], r["method"]): r for r in rows}


def write_outcomes(directory, rows, stage, seeds):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(dict(stage=stage, seeds=seeds,
        cells=CELLS, selected=asdict(EstimatorConfig()))))
    path = directory / "outcomes.json"
    path.write_text(json.dumps(dict(rows=rows, expected_rows=len(rows), complete=True, failures=[])))
    return path


def test_primary_complete_matrix_has_32_per_world_and_rejects_duplicates(tmp_path):
    seeds = list(range(914000, 914030))
    rows = mock_rows(seeds, "confirmation")
    assert len(rows) == 32 * 30
    path = write_outcomes(tmp_path, rows, "confirmation", seeds)
    assert len(analyzer.read_complete(path)[0]) == 960
    data = json.loads(path.read_text())
    data["rows"][-1] = data["rows"][0]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        analyzer.read_complete(path)


def test_primary_missing_cell_cannot_self_declare_smaller_complete_matrix(tmp_path):
    seeds = list(range(914000, 914030))
    rows = [r for r in mock_rows(seeds, "confirmation") if r["cell"] != "severe_hotspot"]
    path = write_outcomes(tmp_path, rows, "confirmation", seeds)
    with pytest.raises(ValueError):
        analyzer.read_complete(path)


def test_selection_uses_all_twelve_candidates_and_eight_worlds():
    rows = mock_rows(list(range(911000, 911008)), "development")
    result = analyzer.bounded_selection(rows, indexed(rows))
    assert len(result["candidates"]) == 12
    assert result["selection_passed"]
    assert len(result["seeds"]) == 8
    assert not result["extra_search_permitted"]
    fewer = [r for r in rows if r["seed"] != 911007]
    with pytest.raises(ValueError):
        analyzer.bounded_selection(fewer, indexed(fewer))
    missing = [r for r in rows if not (r["seed"] == 911007 and r["cell"] == "severe_hotspot")]
    with pytest.raises((ValueError, KeyError)):
        analyzer.bounded_selection(missing, indexed(missing))


def test_selection_rejects_eight_wrong_namespace_worlds():
    rows = mock_rows(list(range(7000, 7008)), "development")
    with pytest.raises(ValueError):
        analyzer.bounded_selection(rows, indexed(rows))


def test_selection_numerical_invalidity_cannot_win():
    rows = mock_rows(list(range(911000, 911008)), "development")
    for row in rows:
        row["metrics"]["solver_failure_rate"] = .1
    result = analyzer.bounded_selection(rows, indexed(rows))
    assert not result["selection_passed"]
    assert result["estimator"] is None


def test_inference_no_favorable_zero_variance_and_holm_family(monkeypatch):
    for values in ([-1., -1., -1.], [0., 0.], [1., 1.], [float("nan"), 0.]):
        result = analyzer.infer(values)
        assert not result["valid"]
        assert result["p"] is None
    vectors = {f"H{i}": np.array([-i*.1, -.05, .01, .02]) for i in range(1, 8)}
    monkeypatch.setattr(analyzer, "contrasts", lambda index, seeds: vectors)
    result = analyzer.inference({}, [1, 2, 3, 4])
    expected = holm_adjust({h: item["p"] for h, item in result.items()})
    assert {h: item["holm_p"] for h, item in result.items()} == expected


def test_power_sizing_stays_30_to_100_and_blocks_zero_variance():
    strong = {f"H{i}": np.array([-1., -.9, -1.1, -.8]) for i in range(1, 8)}
    n, powers = locker.required_worlds(strong)
    assert n == 30
    assert all(power >= .9 for power in powers.values())
    weak = {f"H{i}": np.array([-.001, .5, -.502, .002]) for i in range(1, 8)}
    assert locker.required_worlds(weak)[0] is None
    constant = {f"H{i}": np.array([-1., -1., -1.]) for i in range(1, 8)}
    assert locker.required_worlds(constant)[0] is None


def test_resume_rejects_changed_source_config_or_seed(tmp_path):
    base = dict(world=dict(steps=8, agents=4, groups=4, grid_side=2), network={}, attack={})
    job = dict(output=str(tmp_path), seed=910000, cells=["anchor_clean"], source_hash="old",
        base=base, overrides={}, protocol=dict(transport={}, scale={}), stage="preflight",
        selected=asdict(EstimatorConfig()), factorial=True, save_predictions=False)
    cfg = cell_configuration(base, "anchor_clean", {})
    cfg["transport"] = {}
    cfg["scale"] = {"acquisition_epochs": 8}
    specs = method_specs(EstimatorConfig(), "anchor_clean")
    identity = config_hash(dict(source="old", config=cfg, seed=910000, stage="preflight",
        methods=[{**s, "estimator": asdict(s["estimator"]) if s["estimator"] else None} for s in specs]))
    for mutation in (dict(source_hash="new"), dict(overrides={"world.agents": 5}), dict(seed=910001)):
        changed = {**job, **mutation}
        path = tmp_path / "jobs" / str(changed["seed"]) / "anchor_clean" / "outcomes.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(execution_identity=identity, rows=[])))
        with pytest.raises(ValueError, match="identity"):
            runner.run_world(changed)


@pytest.mark.parametrize("changed", ["source", "base"])
def test_confirmation_refuses_source_or_base_changed_after_lock(tmp_path, monkeypatch, changed):
    seeds = list(range(914000, 914030))
    selected = tmp_path / "selected.json"
    protocol = dict(historical_configuration="base.yaml", seed_namespaces=dict(confirmation=[914000, 914099]),
                    primary=dict(cells=CELLS), transport={}, scale={})
    selected.write_text(json.dumps(dict(selection_passed=True, estimator=asdict(EstimatorConfig()),
        confirmation_lock=dict(seeds=seeds, source_hash="LOCKED_SOURCE", protocol_hash=config_hash(protocol),
                               base_hash=config_hash({})))))
    real_load = runner.load_config
    monkeypatch.setattr(runner, "load_config", lambda path: protocol if str(path).endswith("protocol.yaml")
        else ({"changed": True} if changed == "base" else {}) if str(path).endswith("base.yaml") else real_load(path))
    monkeypatch.setattr(runner, "source_inventory", lambda: ({}, "CHANGED_SOURCE" if changed == "source" else "LOCKED_SOURCE"))
    monkeypatch.setattr(runner.psutil, "virtual_memory", lambda: type("Memory", (), {"available": 10*2**30})())
    def no_pool(*args, **kwargs):
        raise AssertionError("source mismatch reached compute launch")
    monkeypatch.setattr(runner, "ProcessPoolExecutor", no_pool)
    monkeypatch.setattr(sys, "argv", ["run_v5_campaign.py", "--stage", "confirmation", "--output", str(tmp_path / "campaign"),
        "--selected-config", str(selected), "--seeds", *map(str, seeds)])
    with pytest.raises((ValueError, SystemExit)):
        runner.main()


def lock_fixture(tmp_path, monkeypatch):
    seeds = list(range(913000, 913012))
    rows = mock_rows(seeds, "validation")
    for row in rows:
        row["source_hash"] = "VALIDATION_SOURCE"
    validation = tmp_path / "validation"
    write_outcomes(validation, rows, "validation", seeds)
    manifest = json.loads((validation / "manifest.json").read_text())
    protocol = {"created_utc": "2099-01-01T00:00:00Z", "historical_configuration": "base.yaml"}
    manifest.update(source_hash="VALIDATION_SOURCE", source_files={"airproof/v5_estimator.py": "GOOD"},
                    protocol_hash=config_hash(protocol), base_hash=config_hash({}))
    (validation / "manifest.json").write_text(json.dumps(manifest))
    (validation / "worlds").mkdir()
    for seed in seeds:
        (validation / "worlds" / f"{seed}.json").write_text(json.dumps(dict(seed=seed, elapsed_seconds=1.)))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(dict(selection_passed=True, estimator=asdict(EstimatorConfig()))))
    monkeypatch.setattr(locker, "load_config", lambda path: {} if str(path).endswith("base.yaml") else protocol)
    monkeypatch.setattr(locker, "source_inventory", lambda: ({"airproof/v5_estimator.py": "GOOD"}, "CURRENT_TOOLS"))
    monkeypatch.setattr(locker, "contrasts", lambda index, seeds: {f"H{i}": np.array([-1., -.9, -1.1, -.8]) for i in range(1, 8)})
    monkeypatch.setattr(sys, "argv", ["lock_v5_confirmation.py", "--selection", str(selection),
        "--validation", str(validation), "--output", str(tmp_path / "lock.json")])
    return selection, validation


def test_confirmation_lock_rejects_different_selected_estimator(tmp_path, monkeypatch):
    selection, _ = lock_fixture(tmp_path, monkeypatch)
    selection.write_text(json.dumps(dict(selection_passed=True, estimator=asdict(EstimatorConfig(cap=16.)))))
    with pytest.raises(ValueError, match="estimator"):
        locker.main()


def test_confirmation_lock_rejects_changed_numerical_source(tmp_path, monkeypatch):
    lock_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(locker, "source_inventory", lambda: ({"airproof/v5_estimator.py": "CHANGED"}, "CURRENT_TOOLS"))
    with pytest.raises(ValueError):
        locker.main()


def test_confirmation_lock_rejects_wrong_runtime_world_identities(tmp_path, monkeypatch):
    _, validation = lock_fixture(tmp_path, monkeypatch)
    for path in (validation / "worlds").glob("*.json"):
        path.write_text(json.dumps(dict(seed=123, elapsed_seconds=1.)))
    with pytest.raises(ValueError):
        locker.main()


def test_valid_prospective_lock_binds_worlds_sources_and_configs(tmp_path, monkeypatch):
    lock_fixture(tmp_path, monkeypatch)
    locker.main()
    result = json.loads((tmp_path / "lock.json").read_text())
    assert not result["blocked_reasons"]
    lock = result["confirmation_lock"]
    assert lock["worlds"] == 30
    assert lock["seeds"] == list(range(914000, 914030))
    assert lock["source_files"] == {"airproof/v5_estimator.py": "GOOD"}
    assert lock["base_hash"] == config_hash({})
    assert lock["power_target"] == .9


@pytest.mark.parametrize("recall", [None, float("nan")])
def test_invalid_recall_blocks_lock_without_favorable_support_filtering(tmp_path, monkeypatch, recall):
    _, validation = lock_fixture(tmp_path, monkeypatch)
    path = validation / "outcomes.json"
    outcomes = json.loads(path.read_text())
    next(r for r in outcomes["rows"] if r["method"] == "AP")["metrics"]["event_recall"] = recall
    path.write_text(json.dumps(outcomes))
    locker.main()
    result = json.loads((tmp_path / "lock.json").read_text())
    assert result["confirmation_lock"] is None
    assert any("recall support" in reason for reason in result["blocked_reasons"])


def test_cache_dependency_key_hit_miss_and_checksum_corruption(tmp_path):
    calls = []
    def factory():
        calls.append(1)
        return {"array": np.arange(4), "generation": len(calls)}
    first = runner.cached_value(tmp_path, config_hash({"dependency": 1}), factory)
    hit = runner.cached_value(tmp_path, config_hash({"dependency": 1}), factory)
    assert len(calls) == 1
    np.testing.assert_array_equal(first["array"], hit["array"])
    miss = runner.cached_value(tmp_path, config_hash({"dependency": 2}), factory)
    assert miss["generation"] == 2
    path = tmp_path / f"{config_hash({'dependency': 1})}.pkl.gz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="corrupted cache"):
        runner.cached_value(tmp_path, config_hash({"dependency": 1}), factory)
    assert len(calls) == 2


def mocked_resume_job(tmp_path, monkeypatch):
    job = dict(output=str(tmp_path), seed=910000, cells=["anchor_clean"], source_hash="1"*64,
        base=dict(world=dict(steps=2, agents=4, groups=4, grid_side=2), network={}, attack={}, twin={"fixed_lag": 6}),
        overrides={}, protocol=dict(transport={}, scale={}), stage="preflight",
        selected=asdict(EstimatorConfig()), factorial=True, save_predictions=True)
    world = SimpleNamespace(truth=np.ones((2, 4)), observations=(), reference_observations=(),
                            cell_groups=np.arange(4), public_meteorology=None)
    monkeypatch.setattr(runner, "generate_world", lambda *args: world)
    monkeypatch.setattr(runner, "_v4_public_components", lambda *args: (np.ones((2, 4)), None, {}))
    monkeypatch.setattr(runner, "generate_transport_trace", lambda *args: SimpleNamespace(trace_hash="3"*64))
    monkeypatch.setattr(runner, "simulate_transport", lambda *args, **kwargs: SimpleNamespace(release_arrivals={},raw_arrivals={}))
    monkeypatch.setattr(runner, "release_evidence", lambda *args, **kwargs: ({}, {"values": np.zeros((2, 4))}))
    evaluations = []
    def evaluate(*args, **kwargs):
        spec = args[5]
        evaluations.append(spec["method"])
        return {"method": spec["method"], "config_hash":"2"*64,
            "metrics":{"feasible_floor_violations":0,"solver_failure_rate":0.},
            "transport":{"token_violations":0}}, {"live": np.ones((2, 4)), "reconstructed": np.ones((2, 4)), "counts": np.ones((8, 4))}
    monkeypatch.setattr(runner, "evaluate_prepared", evaluate)
    return job, evaluations, tmp_path / "jobs" / "910000" / "anchor_clean"


def test_method_resume_rejects_saved_row_identity(tmp_path, monkeypatch):
    job, evaluations, directory = mocked_resume_job(tmp_path, monkeypatch)
    runner.run_world(job)
    (directory / "outcomes.json").unlink()
    path = directory / "AP.json"
    previous = json.loads(path.read_text())
    previous["execution_identity"] = "wrong"
    path.write_text(json.dumps(previous))
    evaluations.clear()
    with pytest.raises(ValueError, match="identity"):
        runner.run_world(job)
    assert not evaluations


@pytest.mark.parametrize("completed_cell", [False, True])
def test_resume_recomputes_only_missing_predictions(tmp_path, monkeypatch, completed_cell):
    job, evaluations, directory = mocked_resume_job(tmp_path, monkeypatch)
    runner.run_world(job)
    assert evaluations == ["AP", "SQ", "PUBLIC", "HUBER"]
    if not completed_cell:
        (directory / "outcomes.json").unlink()
    (directory / "AP.npz").unlink()
    evaluations.clear()
    result = runner.run_world(job)
    assert evaluations == ["AP"]
    assert (directory / "AP.npz").exists()
    assert len(result["rows"]) == 4


def test_public_cache_does_not_reuse_different_twin_dependencies(tmp_path, monkeypatch):
    job, _, _ = mocked_resume_job(tmp_path, monkeypatch)
    job["cells"] = ["anchor_clean", "outage_clean"]
    real_configuration = runner.cell_configuration
    def configured(base, cell, overrides):
        cfg = real_configuration(base, cell, overrides)
        cfg["twin"] = {"public_parameter": 1 if cell == "anchor_clean" else 2}
        return cfg
    monkeypatch.setattr(runner, "cell_configuration", configured)
    calls = []
    def public_components(world, cfg):
        calls.append(cfg["twin"]["public_parameter"])
        return np.ones((2, 4))*calls[-1], None, {}
    monkeypatch.setattr(runner, "_v4_public_components", public_components)
    runner.run_world(job)
    assert calls == [1, 2]

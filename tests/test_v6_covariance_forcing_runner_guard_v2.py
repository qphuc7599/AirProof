import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_v6_covariance_forcing_development_v2.py"


def load_script():
    spec = importlib.util.spec_from_file_location("v2_runner_guard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_or_failed_readiness_prohibits_outcomes(tmp_path, monkeypatch):
    module = load_script()
    registration = json.loads((ROOT / "configs/v6/innovation_covariance_forcing_development_v2.json").read_text())
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="readiness artifact absent"):
        module.require_ready(registration)
    path = tmp_path / registration["readiness"]["artifact"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"readiness_pass": False}))
    with pytest.raises(RuntimeError, match="mismatch"):
        module.require_ready(registration)


def test_planner_has_exact_development_cells_and_candidates():
    module = load_script()
    registration = json.loads((ROOT / "configs/v6/innovation_covariance_forcing_development_v2.json").read_text())
    plan = module.planned_runs(registration, [6241020, 6241021])
    assert len(plan) == 2 * 3 * 4
    assert {row["scenario"] for row in plan} == {"severe_clean", "severe_drift", "severe_hotspot"}
    assert {row["candidate"] for row in plan} == {row["id"] for row in registration["candidates"]}
    with pytest.raises(ValueError, match="unregistered"):
        module.planned_runs(registration, [6241000])


def test_registration_separates_prediction_and_scoring_capabilities():
    registration = json.loads((ROOT / "configs/v6/innovation_covariance_forcing_development_v2.json").read_text())
    excluded = set(registration["producer"]["prediction_view_excludes"])
    assert {"calibration_truth", "evaluation_truth", "event_mask", "attack_label"} <= excluded
    assert registration["world_roles"]["development"]["truth_use"].startswith("scoring only")
    assert registration["same_information_controls"]["names"] == ["PUBLIC", "SQ", "HUBER"]

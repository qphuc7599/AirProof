"""Exercise proposed fixes without importing/replacing frozen production modules."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v5_campaign as campaign_tests
import test_v5_export_integrity as reproductions


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def staged(monkeypatch):
    module = module_from_path("staged_runner_integrity", ROOT/"reports/v5/staged_integrity_fix/scripts/run_v5_campaign.py")
    monkeypatch.setattr(campaign_tests, "runner", module)
    monkeypatch.setattr(reproductions, "runner", module)
    return module


@pytest.mark.parametrize("completed", [False, True])
def test_staged_corrupt_prediction_reproduction_passes(tmp_path, monkeypatch, staged, completed):
    reproductions.test_corrupted_existing_prediction_is_rejected_or_rebuilt(tmp_path, monkeypatch, completed)


def test_staged_nested_identity_reproduction_passes(tmp_path, monkeypatch, staged):
    reproductions.test_completed_cell_rejects_nested_row_execution_mismatch(tmp_path, monkeypatch)


def test_staged_exporter_complete_matrix_reproduction_passes(tmp_path, monkeypatch):
    original = reproductions.exporter_fixture
    def fixture(folder):
        result = original(folder)
        result[0].write_bytes((ROOT/"reports/v5/staged_integrity_fix/reports/v5/tools/export_development.py").read_bytes())
        return result
    monkeypatch.setattr(reproductions, "exporter_fixture", fixture)
    reproductions.test_exporter_rejects_false_complete_unselected_method_cell(tmp_path)


def test_readonly_prediction_auditor_rejects_corruption_shape_and_identity(tmp_path):
    auditor = module_from_path("standalone_integrity_auditor", ROOT/"reports/v5/tools/audit_campaign_integrity.py")
    path = tmp_path/"AP.npz"
    arrays = dict(live=np.ones((2,4)), reconstructed=np.ones((2,4)), truth=np.ones((2,4)),
                  counts=np.ones((2,4)), cell_groups=np.arange(4), execution_identity=np.array("id"),
                  source_hash=np.array("source"), config_hash=np.array("config"))
    args = dict(steps=2, cells=4, groups=4, lag=0, identity="id", source_hash="source", config_digest="config")
    np.savez(path, **arrays)
    assert auditor.audit_prediction(path, **args) == []
    before = path.read_bytes()
    auditor.audit_prediction(path, **args)
    assert path.read_bytes() == before
    for changed in ({"live":np.ones((1,4))}, {"execution_identity":np.array("wrong")},
                    {"live":np.full((2,4),np.nan)}):
        np.savez(path, **{**arrays, **changed})
        with pytest.raises(ValueError):
            auditor.audit_prediction(path, **args)
    path.write_bytes(b"not an NPZ archive")
    with pytest.raises(ValueError):
        auditor.audit_prediction(path, **args)

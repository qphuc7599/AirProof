"""Isolated reproduction fixtures; never touch a live campaign or frozen code."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v5_campaign import analyzer, indexed, mock_rows, mocked_resume_job, runner


def exporter_fixture(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "reports/v5/tools/export_development.py"
    script.parent.mkdir(parents=True)
    script.write_bytes((root / "reports/v5/tools/export_development.py").read_bytes())
    source = tmp_path / "reports/v5/development/integrated_911000_911007_v1"
    source.mkdir(parents=True)
    (tmp_path / "source_paper/AirProof_Elsevier/v5_sections").mkdir(parents=True)
    rows = mock_rows(list(range(911000, 911008)), "development")
    selection = analyzer.bounded_selection(rows, indexed(rows))
    outcomes = dict(rows=rows, expected_rows=760, complete=True, failures=[])
    return script, source, selection, outcomes


def save_fixture(source, selection, outcomes):
    raw = json.dumps(outcomes).encode()
    (source / "outcomes.json").write_bytes(raw)
    selection["source_outcomes_sha256"] = hashlib.sha256(raw).hexdigest()
    (source / "selection.json").write_text(json.dumps(selection))


def test_exporter_rejects_changed_outcomes_after_selection(tmp_path):
    script, source, selection, outcomes = exporter_fixture(tmp_path)
    save_fixture(source, selection, outcomes)
    outcomes["rows"][0]["metrics"]["rmse"] = 999.
    (source / "outcomes.json").write_text(json.dumps(outcomes))
    with pytest.raises(AssertionError):
        runpy.run_path(str(script))
    assert not (tmp_path / "reports/v5/development_export/summary.json").exists()


def test_exporter_rejects_false_complete_unselected_method_cell(tmp_path):
    script, source, selection, outcomes = exporter_fixture(tmp_path)
    chosen = selection["candidate"]
    est = selection["estimator"]
    control = f"SQ_{est['objective']}_reg{est['regularization_multiplier']:g}"
    row = next(r for r in outcomes["rows"] if r["method"] not in (chosen, control, "PUBLIC"))
    row["cell"] = "unregistered_cell"  # still 760 unique rows; one required identity is absent
    save_fixture(source, selection, outcomes)
    with pytest.raises((ValueError, AssertionError, KeyError)):
        runpy.run_path(str(script))


@pytest.mark.parametrize("completed", [False, True])
def test_corrupted_existing_prediction_is_rejected_or_rebuilt(tmp_path, monkeypatch, completed):
    job, evaluations, directory = mocked_resume_job(tmp_path, monkeypatch)
    runner.run_world(job)
    if not completed:
        (directory / "outcomes.json").unlink()
    prediction = directory / "AP.npz"
    prediction.write_bytes(b"not an NPZ archive")
    evaluations.clear()
    try:
        runner.run_world(job)
    except (ValueError, OSError):
        return  # rejection is acceptable; silent success with corrupt output is not
    with np.load(prediction, allow_pickle=False) as arrays:
        assert arrays["live"].shape == (2, 4)
        assert np.isfinite(arrays["live"]).all()


def test_completed_cell_rejects_nested_row_execution_mismatch(tmp_path, monkeypatch):
    job, evaluations, directory = mocked_resume_job(tmp_path, monkeypatch)
    runner.run_world(job)
    path = directory / "outcomes.json"
    stored = json.loads(path.read_text())
    expected = stored["execution_identity"]
    stored["rows"][0]["execution_identity"] = "different execution"
    path.write_text(json.dumps(stored))
    try:
        result = runner.run_world(job)
    except ValueError:
        return
    assert all(row["execution_identity"] == expected for row in result["rows"])

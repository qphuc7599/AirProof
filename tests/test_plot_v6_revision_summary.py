import json

import pytest

from scripts.plot_v6_revision_summary import finite, load


def test_figure_loader_rejects_missing_and_nonfinite(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load(invalid)


def test_figure_numeric_guard_rejects_nan():
    with pytest.raises(ValueError, match="NaN or infinite"):
        finite(float("nan"), "test")
    assert finite(3, "test") == 3.0

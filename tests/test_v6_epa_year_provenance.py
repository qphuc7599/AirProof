from pathlib import Path

import pytest

from airproof.data import preprocess_epa_bay_area_pm25


def test_epa_preprocessor_requires_inferable_or_explicit_year(tmp_path):
    missing = tmp_path / "ambiguous.zip"
    missing.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="year is required"):
        preprocess_epa_bay_area_pm25(missing, tmp_path / "out.parquet")


def test_epa_preprocessor_rejects_impossible_explicit_year(tmp_path):
    missing = tmp_path / "hourly_88101_2025.zip"
    missing.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="invalid EPA archive year"):
        preprocess_epa_bay_area_pm25(missing, tmp_path / "out.parquet", year=1989)

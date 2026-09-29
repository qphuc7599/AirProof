"""Normalize official EPA 2025 data and freeze outcome-independent test support."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from airproof.data import preprocess_epa_bay_area_pm25
from airproof.v6_archive_protocol import latest_disjoint_windows


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/"data/external/epa/hourly_88101_2025.zip"
DATA = ROOT/"data/external/epa/epa_south_coast_pm25_2025.parquet"
OUT = ROOT/"reports/v6/epa_2025_protocol"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not RAW.exists():
        raise FileNotFoundError(RAW)
    if not DATA.exists():
        preprocess_epa_bay_area_pm25(RAW, DATA, max_stations=64, year=2025,
            county_codes={37, 59, 65, 71}, region_name="California South Coast counties")
    OUT.mkdir(exist_ok=False)
    frame = pd.read_parquet(DATA)
    table = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    complete_index = pd.date_range(table.index.min(), table.index.max(), freq="h", tz="UTC")
    table = table.reindex(complete_index)
    count = len(table)
    train_end = int(np.floor(.65*count))
    validation_start = train_end+24
    validation_end = int(np.floor(.80*count))
    candidate_start = validation_end+24
    training_support = table.iloc[:train_end].notna().mean(0)
    pool_names = sorted(training_support[training_support >= .8].index.tolist())
    if len(pool_names) < 4:
        raise RuntimeError("fewer than four stations have 80% training support")
    pool_indices = [table.columns.get_loc(name) for name in pool_names]
    chosen = latest_disjoint_windows(table.notna().to_numpy(),
        candidate_start=candidate_start, width=336, gap=168,
        minimum_support=.8, minimum_stations=4, station_pool=pool_indices)
    station_names = [str(table.columns[index]) for index in chosen["station_indices"]]
    registration = {
        "created_utc": datetime.now(UTC).isoformat(),
        "role": "fresh official-archive protocol freeze; values and errors not scored",
        "dataset": "US EPA AQS hourly PM2.5 88101, California South Coast counties, calendar year 2025",
        "source_url": "https://aqs.epa.gov/aqsweb/airdata/hourly_88101_2025.zip",
        "source_index": "https://aqs.epa.gov/aqsweb/airdata/download_files.html",
        "raw_sha256": digest(RAW), "normalized_sha256": digest(DATA),
        "source_sha256": digest(Path(__file__)),
        "protocol_source_sha256": digest(ROOT/"airproof/v6_archive_protocol.py"),
        "timestamps": count, "start_utc": table.index[0].isoformat(),
        "end_utc": table.index[-1].isoformat(),
        "fit": [0, train_end], "purge_before_validation": [train_end, validation_start],
        "validation": [validation_start, validation_end],
        "purge_before_test_search": [validation_end, candidate_start],
        "test_windows": chosen["windows"], "test_gap_hours": chosen["gap"],
        "stations": station_names,
        "station_rule": "80% support in fit and each test window; sorted intersection; missingness only",
        "region_rule": "county codes 037,059,065,071 fixed after the 2025 88101 archive contained zero rows in the prior Bay Area county set; only state/county row counts were inspected",
        "window_rule": "latest eligible second window, then latest eligible first window with at least 168h gap",
        "minimum_window_hours": 336, "minimum_stations": 4,
        "minimum_support": .8, "errors_inspected": False,
        "model_or_estimator_selected": False, "confirmation_scored": False,
    }
    (OUT/"registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    print(json.dumps({key: registration[key] for key in
        ("timestamps", "start_utc", "end_utc", "fit", "validation", "test_windows", "stations")}, indent=2))


if __name__ == "__main__":
    main()

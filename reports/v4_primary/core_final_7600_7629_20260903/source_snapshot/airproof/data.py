from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

UCI_BEIJING_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
UCI_BEIJING_DOI = "https://doi.org/10.24432/C5RK5G"
EPA_AIRDATA_INDEX = "https://aqs.epa.gov/aqsweb/airdata/download_files.html"
EPA_PM25_HOURLY_URL = "https://aqs.epa.gov/aqsweb/airdata/hourly_88101_{year}.zip"
EPA_BAY_AREA_COUNTIES = {1, 13, 41, 55, 75, 81, 85, 95, 97}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def download_beijing(destination: str | Path = "data/raw/beijing") -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "beijing_multi_site_air_quality.zip"
    with urllib.request.urlopen(UCI_BEIJING_URL, timeout=120) as response, archive.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    extracted = destination / "extracted"
    extracted.mkdir(exist_ok=True)
    _safe_extract(archive, extracted)
    for nested in extracted.rglob("*.zip"):
        nested_dir = nested.with_suffix("")
        nested_dir.mkdir(exist_ok=True)
        _safe_extract(nested, nested_dir)
    csvs = sorted(extracted.rglob("*.csv"))
    if len(csvs) < 12:
        raise RuntimeError(f"Expected at least 12 station CSVs, found {len(csvs)}")
    manifest = {
        "dataset": "Beijing Multi-Site Air Quality",
        "source": UCI_BEIJING_URL,
        "doi": UCI_BEIJING_DOI,
        "license": "CC BY 4.0",
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": sha256_file(archive),
        "files": [{"path": str(path.relative_to(destination)), "sha256": sha256_file(path)} for path in csvs],
    }
    manifest_path = destination / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def preprocess_beijing(
    source: str | Path = "data/raw/beijing/extracted",
    output: str | Path = "data/processed/beijing_hourly.parquet",
) -> Path:
    source = Path(source)
    frames: list[pd.DataFrame] = []
    for path in sorted(source.rglob("*.csv")):
        frame = pd.read_csv(path)
        required = {"year", "month", "day", "hour", "station"}
        if not required.issubset(frame.columns):
            continue
        frame["timestamp_utc"] = pd.to_datetime(
            frame[["year", "month", "day", "hour"]], errors="raise", utc=True
        )
        frame["source_file"] = path.name
        frames.append(frame)
    if len(frames) != 12:
        raise RuntimeError(f"Expected exactly 12 station tables, found {len(frames)}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["timestamp_utc", "station", "No"]).reset_index(drop=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    summary = {
        "rows": len(combined),
        "stations": sorted(combined["station"].dropna().unique().tolist()),
        "start_utc": combined["timestamp_utc"].min().isoformat(),
        "end_utc": combined["timestamp_utc"].max().isoformat(),
        "missing_by_column": {key: int(value) for key, value in combined.isna().sum().items()},
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def preprocess_epa_bay_area_pm25(
    archive: str | Path,
    output: str | Path,
    *,
    max_stations: int = 16,
) -> Path:
    """Filter the official national hourly PM2.5 archive without loading it all in RAM."""
    archive = Path(archive)
    required = [
        "State Code",
        "County Code",
        "Site Num",
        "Latitude",
        "Longitude",
        "Date GMT",
        "Time GMT",
        "Sample Measurement",
        "Units of Measure",
    ]
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(archive) as zipped:
        csv_names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"Expected one EPA CSV in archive, found {len(csv_names)}")
        with zipped.open(csv_names[0]) as stream:
            for chunk in pd.read_csv(stream, usecols=required, chunksize=250_000, low_memory=False):
                state = pd.to_numeric(chunk["State Code"], errors="coerce")
                county = pd.to_numeric(chunk["County Code"], errors="coerce")
                keep = state.eq(6) & county.isin(EPA_BAY_AREA_COUNTIES)
                filtered = chunk.loc[keep].copy()
                if not filtered.empty:
                    frames.append(filtered)
    if not frames:
        raise RuntimeError("EPA archive contained no Bay Area PM2.5 rows")
    frame = pd.concat(frames, ignore_index=True)
    frame["value"] = pd.to_numeric(frame["Sample Measurement"], errors="coerce")
    frame = frame[frame["value"].between(-5.0, 1000.0)]
    frame["timestamp_utc"] = pd.to_datetime(
        frame["Date GMT"].astype(str) + " " + frame["Time GMT"].astype(str),
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(subset=["timestamp_utc", "Latitude", "Longitude", "value"])
    frame["station"] = (
        frame["State Code"].astype(int).astype(str).str.zfill(2)
        + "-"
        + frame["County Code"].astype(int).astype(str).str.zfill(3)
        + "-"
        + frame["Site Num"].astype(int).astype(str).str.zfill(4)
    )
    coverage = frame.groupby("station")["timestamp_utc"].nunique().sort_values(ascending=False)
    selected = coverage.head(max_stations).index
    frame = frame[frame["station"].isin(selected)]
    normalized = (
        frame.groupby(["timestamp_utc", "station"], as_index=False)
        .agg(value=("value", "median"), latitude=("Latitude", "median"), longitude=("Longitude", "median"))
        .sort_values(["timestamp_utc", "station"])
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output, index=False)
    manifest = {
        "dataset": "US EPA AQS hourly PM2.5 88101",
        "source_index": EPA_AIRDATA_INDEX,
        "source_archive": EPA_PM25_HOURLY_URL.format(year=2024),
        "archive_sha256": sha256_file(archive),
        "rows": len(normalized),
        "stations": sorted(normalized["station"].unique().tolist()),
        "start_utc": normalized["timestamp_utc"].min().isoformat(),
        "end_utc": normalized["timestamp_utc"].max().isoformat(),
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output


def chronological_split(frame: pd.DataFrame, *, purge_steps: int) -> dict[str, pd.DataFrame]:
    if purge_steps < 0:
        raise ValueError("purge_steps cannot be negative")
    times = pd.Index(sorted(frame["timestamp_utc"].unique()))
    train_end = int(len(times) * 0.60)
    validation_end = int(len(times) * 0.80)
    if train_end + purge_steps >= validation_end or validation_end + purge_steps >= len(times):
        raise ValueError("Not enough timestamps for the requested purge gaps")
    train_times = set(times[:train_end])
    validation_times = set(times[train_end + purge_steps : validation_end])
    test_times = set(times[validation_end + purge_steps :])
    return {
        "train": frame[frame["timestamp_utc"].isin(train_times)].copy(),
        "validation": frame[frame["timestamp_utc"].isin(validation_times)].copy(),
        "test": frame[frame["timestamp_utc"].isin(test_times)].copy(),
    }

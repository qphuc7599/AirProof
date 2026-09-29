"""Locked Beijing transfer: paired stations and shared-time block sensitivity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def interval(samples, point):
    return {"estimate": float(point), "lower95": float(np.quantile(samples, .025)),
            "upper95": float(np.quantile(samples, .975))}


def transfer_intervals(squared_errors: dict[str, np.ndarray], *, draws=10000, seed=20260903,
                       blocks=(24, 72, 168)) -> dict:
    """Joint blocks preserve contemporaneous dependence of all station folds."""
    rng = np.random.default_rng(seed)
    methods = list(squared_errors)
    baseline = "idw_frozen"
    arrays = [squared_errors[name] for name in methods]
    steps, stations = arrays[0].shape
    if any(array.shape != (steps, stations) for array in arrays):
        raise ValueError("time/station support mismatch")
    if any(not np.array_equal(np.isfinite(array), np.isfinite(arrays[0])) for array in arrays):
        raise ValueError("all comparators must share exactly the same target support")
    station_rmse = {name: np.sqrt(np.nanmean(array, axis=0)) for name, array in squared_errors.items()}
    if any(not np.isfinite(vector).all() for vector in station_rmse.values()):
        raise ValueError("all station folds require finite support")
    ratio = {name: float(vector.mean() / station_rmse[baseline].mean()) for name, vector in station_rmse.items()}
    sampled_stations = rng.integers(stations, size=(draws, stations))
    station_intervals = {name: interval(vector[sampled_stations].mean(axis=1)
        / station_rmse[baseline][sampled_stations].mean(axis=1), ratio[name])
        for name, vector in station_rmse.items()}
    shared_blocks = {}
    for block in blocks:
        if block < 1 or block > steps:
            raise ValueError("block length outside observed horizon")
        samples = {name: [] for name in methods}
        # Chunk draws to avoid allocating a draws x hours x stations tensor.
        for offset in range(0, draws, 100):
            count = min(100, draws - offset)
            starts = rng.integers(steps, size=(count, int(np.ceil(steps / block))))
            indices = ((starts[:, :, None] + np.arange(block)) % steps).reshape(count, -1)[:, :steps]
            means = {name: np.sqrt(np.nanmean(array[indices], axis=1)).mean(axis=1)
                     for name, array in squared_errors.items()}
            for name in methods:
                samples[name].extend((means[name] / means[baseline]).tolist())
        shared_blocks[str(block)] = {name: interval(values, ratio[name]) for name, values in samples.items()}
    return {"mean_station_rmse": {name: float(values.mean()) for name, values in station_rmse.items()},
            "paired_station_bootstrap": station_intervals, "shared_time_block_bootstrap": shared_blocks,
            "draws": draws, "scope": "fixed-network-archived-period-sensitivity-not-independent-city-sampling"}


def analyze(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    lock_bytes = (directory / "selection_lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    if hashlib.sha256(lock_bytes).hexdigest() != completion["selection_lock_sha256"]:
        raise ValueError("selection lock altered after evaluation")
    if lock["source_hash"] != manifest["source_hash"] or lock["protocol"] != manifest["protocol"]:
        raise ValueError("selection/source/protocol mismatch")
    if lock["test_rows_evaluated_before_lock"] != 0 or completion["test_folds"] != 12:
        raise ValueError("incomplete or unsealed nested evaluation")
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    indexed = {row["station"]: row for row in rows}
    if len(rows) != len(indexed) or set(indexed) != set(manifest["stations"]):
        raise ValueError("missing/duplicate station folds")
    if any(row["diagnostics"]["solver_failures"] or row["identity_diagnostics"]["solver_failures"] for row in rows):
        raise ValueError("numerical transfer failure")
    if any(row["diagnostics"]["mean_operator_distance_from_identity"] <= 0 for row in rows):
        raise ValueError("primary weather operators were not active")
    methods = list(rows[0]["methods"])
    errors = {name: [] for name in methods}
    clock = None
    selections = {item["outer_target"]: item for item in lock["selections"]}
    for station in manifest["stations"]:
        arrays = np.load(directory / (station + ".npz"), allow_pickle=False)
        if clock is None:
            clock = arrays["utc_nanoseconds"]
        elif not np.array_equal(clock, arrays["utc_nanoseconds"]):
            raise ValueError("station folds used different time intervals")
        if len(clock) != manifest["protocol"]["test_steps"]:
            raise ValueError("test horizon mismatch")
        if indexed[station]["diagnostics"]["maximum_correction"] > selections[station]["selected_graph"]["correction_clip"] + 1e-8:
            raise ValueError("bounded output correction violated")
        for name in methods:
            residual = arrays[name] - arrays["truth"]
            measured_rmse = np.sqrt(np.nanmean(residual**2))
            if not np.isclose(measured_rmse, indexed[station]["methods"][name]["rmse"], atol=1e-10):
                raise ValueError("saved per-hour arrays disagree with aggregate score")
            errors[name].append(residual**2)
    errors = {name: np.stack(values, axis=1) for name, values in errors.items()}
    inference = transfer_intervals(errors)
    upper_bounds = [inference["paired_station_bootstrap"]["airproof_v4_city"]["upper95"],
        *(item["airproof_v4_city"]["upper95"] for item in inference["shared_time_block_bootstrap"].values())]
    report = {"role": "locked-nested-public-station-weather-transfer", "source_hash": manifest["source_hash"],
        "dataset_hash": manifest["dataset_hash"], "time_windows": manifest["time_windows"],
        "station_folds": 12, "inference": inference,
        "gates": {"all_reported_upper_bounds_below_1.10": max(upper_bounds) < 1.10,
                  "all_reported_upper_bounds_below_1.05": max(upper_bounds) < 1.05},
        "folds": rows, "selection": [{key: item[key] for key in ("outer_target", "selected_public", "selected_graph")}
                                      for item in lock["selections"]],
        "interpretation": manifest["protocol"]["limitations"],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    report = analyze(args.campaign.resolve())
    print(json.dumps({"gates": report["gates"], "means": report["inference"]["mean_station_rmse"]}, indent=2))


if __name__ == "__main__":
    main()

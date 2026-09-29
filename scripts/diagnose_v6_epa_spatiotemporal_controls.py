"""Diagnose the clean cost of the fixed cap around the locked spatiotemporal public center.

This is a development-only mechanistic decomposition.  It cannot select an
estimator and never reads either frozen EPA test window.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.predictor_wrapper import synthetic_citizen_replay
from airproof.v6_estimator import EstimatorConfig, estimate_public_field
from scripts.select_v6_epa_2025_estimator import DATA, PARTITIONS, PROTOCOL, WARMUP


PUBLIC = ROOT / "reports/v6/epa_2025_public_spatiotemporal"
BASE_PUBLIC = ROOT / "reports/v6/epa_2025_public_model"
OUT = ROOT / "reports/v6/epa_2025_spatiotemporal_controls"
SEEDS = list(range(918300, 918308))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    rolling_lock = json.loads((PUBLIC / "selection_lock.json").read_text())
    base_lock = json.loads((BASE_PUBLIC / "selection_lock.json").read_text())
    if rolling_lock["test_metrics_read_for_selection"] or base_lock["test_metrics_read_for_selection"]:
        raise ValueError("public predictors must be pre-test")
    config = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05,
        lambda_spatial=.02, lag=6, time_step=1., numerical_refinements=2,
        huber_delta=2., loss="quadratic", output_cap=False, input_clip=False)
    arms = {
        "quadratic_unbounded": config,
        "quadratic_cap8": replace(config, output_cap=True),
        "quadratic_clip12_cap8": replace(config, output_cap=True,
                                            input_clip=True, clip_delta=4.),
    }
    registration = {
        "role": "development-only clean mechanistic decomposition; no selection",
        "question": "does the fixed history-uniform cap alone preclude the 1.05 clean gate?",
        "partition": PARTITIONS["development"], "seeds": SEEDS,
        "arms": {name: asdict(value) for name, value in arms.items()},
        "hashes": {str(path): digest(path) for path in (
            Path(__file__), ROOT / "airproof/v6_estimator.py", PROTOCOL, DATA,
            PUBLIC / "selection_lock.json", PUBLIC / "selected_pretest_prediction.npz",
            BASE_PUBLIC / "selection_lock.json", BASE_PUBLIC / "models.pkl")},
        "test_windows_read": False, "selection_permitted": False,
    }
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2))

    start, end = PARTITIONS["development"]
    context_start = start - WARMUP
    cutoff = pd.Timestamp(protocol["start_utc"]) + pd.Timedelta(hours=end)
    frame = pd.read_parquet(DATA, filters=[("timestamp_utc", "<", cutoff)])
    table = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    table = table.reindex(pd.date_range(table.index.min(), periods=end, freq="h", tz="UTC"))[
        protocol["stations"]]
    values = table.to_numpy()[context_start:end]
    with np.load(PUBLIC / "selected_pretest_prediction.npz") as archive:
        public = archive["prediction"][context_start:end].copy()
    with (BASE_PUBLIC / "models.pkl").open("rb") as stream:
        coordinates = pickle.load(stream)["coordinates"]
    coordinates = (coordinates - coordinates.mean(0)) * np.array([
        111.2, 111.2 * np.cos(np.deg2rad(coordinates[:, 0].mean()))])
    target = values[WARMUP:]
    observed = np.isfinite(target)
    channel_truth = np.maximum(np.where(np.isfinite(values), values, public), 0.)
    rows = []
    started = time.perf_counter()
    with (OUT / "development.jsonl").open("w") as output:
        for seed in SEEDS:
            records = synthetic_citizen_replay(channel_truth, seed=seed, kind="clean")
            records = [item for item in records if np.isfinite(values[item.epoch, item.cell])]
            for name, arm in arms.items():
                result = estimate_public_field(public, coordinates, records, arm,
                                               innovation_scales=3.)
                estimate = result.reconstructed[WARMUP:]
                row = {"seed": seed, "method": name,
                       "rmse": float(np.sqrt(np.mean((estimate[observed] - target[observed]) ** 2))),
                       "maximum_correction": result.diagnostics["maximum_correction"],
                       "solver_failure_rate": result.diagnostics["solver_failure_rate"]}
                rows.append(row)
                output.write(json.dumps(row) + "\n")
                output.flush()
            print(json.dumps({"seed": seed, "rows": len(rows)}), flush=True)
    means = {name: float(np.mean([row["rmse"] for row in rows if row["method"] == name]))
             for name in arms}
    denominator = means["quadratic_unbounded"]
    summary = {"means": means,
               "ratios_to_quadratic_unbounded": {name: value / denominator
                                                   for name, value in means.items()},
               "cap8_compatible_with_clean_gate": means["quadratic_cap8"] / denominator <= 1.05,
               "all_numerical_pass": all(row["solver_failure_rate"] == 0 for row in rows),
               "rows": len(rows), "elapsed_seconds": time.perf_counter() - started,
               "test_windows_read": False, "selection_permitted": False}
    (OUT / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


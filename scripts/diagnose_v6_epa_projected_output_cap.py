"""Evaluate the literal post-estimation output-cap control on development."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from dataclasses import asdict
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
OUT = ROOT / "reports/v6/epa_2025_projected_output_cap"
SEEDS = list(range(918400, 918408))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(public, estimate, cap=8.):
    correction = np.clip(estimate - public, -cap, cap)
    return np.maximum(public + correction, 0.)


def main() -> None:
    OUT.mkdir(exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    public_lock = json.loads((PUBLIC / "selection_lock.json").read_text())
    config = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05,
        lambda_spatial=.02, lag=6, time_step=1., numerical_refinements=2,
        loss="quadratic", output_cap=False, input_clip=False)
    registration = {
        "role": "development-only literal output-cap control; no selection",
        "method": "unbounded same-information quadratic solve followed by componentwise correction projection to [-8,8]",
        "partition": PARTITIONS["development"], "seeds": SEEDS,
        "config": asdict(config), "cap": 8.,
        "hashes": {str(path): sha(path) for path in (
            Path(__file__), ROOT / "airproof/v6_estimator.py", PROTOCOL, DATA,
            PUBLIC / "selection_lock.json", PUBLIC / "selected_pretest_prediction.npz",
            BASE_PUBLIC / "models.pkl")},
        "estimator_validation_read": False, "test_windows_read": False,
        "selection_permitted": False,
    }
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2))
    start, end = PARTITIONS["development"]; context_start = start - WARMUP
    cutoff = pd.Timestamp(protocol["start_utc"]) + pd.Timedelta(hours=end)
    frame = pd.read_parquet(DATA, filters=[("timestamp_utc", "<", cutoff)])
    table = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    table = table.reindex(pd.date_range(table.index.min(), periods=end, freq="h", tz="UTC"))[
        protocol["stations"]]
    values = table.to_numpy()[context_start:end]
    with np.load(PUBLIC / "selected_pretest_prediction.npz") as item:
        public = item["prediction"][context_start:end].copy()
    with (BASE_PUBLIC / "models.pkl").open("rb") as stream:
        coordinates = pickle.load(stream)["coordinates"]
    coordinates = (coordinates - coordinates.mean(0)) * np.array([
        111.2, 111.2 * np.cos(np.deg2rad(coordinates[:, 0].mean()))])
    truth = values[WARMUP:]; observed = np.isfinite(truth)
    channel_truth = np.maximum(np.where(np.isfinite(values), values, public), 0.)
    rows = []; started = time.perf_counter()
    with (OUT / "development.jsonl").open("w") as output:
        for seed in SEEDS:
            records = synthetic_citizen_replay(channel_truth, seed=seed, kind="clean")
            records = [item for item in records if np.isfinite(values[item.epoch, item.cell])]
            result = estimate_public_field(public, coordinates, records, config,
                                           innovation_scales=3.)
            unbounded = result.reconstructed[WARMUP:]
            projected = project(public, result.reconstructed)[WARMUP:]
            for name, estimate in (("quadratic_unbounded", unbounded),
                                   ("quadratic_projected_cap8", projected)):
                row = {"seed": seed, "method": name,
                       "rmse": float(np.sqrt(np.mean((estimate[observed] - truth[observed]) ** 2))),
                       "solver_failure_rate": result.diagnostics["solver_failure_rate"]}
                rows.append(row); output.write(json.dumps(row) + "\n"); output.flush()
            print(json.dumps({"seed": seed, "rows": len(rows)}), flush=True)
    means = {name: float(np.mean([row["rmse"] for row in rows if row["method"] == name]))
             for name in ("quadratic_unbounded", "quadratic_projected_cap8")}
    ratio = means["quadratic_projected_cap8"] / means["quadratic_unbounded"]
    result = {"means": means, "clean_ratio": ratio, "passes_1_05": ratio <= 1.05,
              "history_uniform_absolute_correction_cap": 8.,
              "all_numerical_pass": all(row["solver_failure_rate"] == 0 for row in rows),
              "rows": len(rows), "elapsed_seconds": time.perf_counter() - started,
              "estimator_validation_read": False, "test_windows_read": False,
              "selection_permitted": False}
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

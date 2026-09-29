"""Select nonlinear rolling public nowcasts with target-excluding peer inputs."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

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

from airproof.v6_feasibility import cap_rmse_floor
from airproof.v6_public_archive_model import causal_archive_design
from airproof.v6_public_rolling_hgb import rolling_stationwise_hgb
from airproof.v6_public_spatial import spatial_public_features
from scripts.select_v6_epa_2025_estimator import DATA, PARTITIONS, PROTOCOL
from scripts.select_v6_epa_public_spatiotemporal import SPATIAL_ALL, SPATIAL_CORE


BASE_PUBLIC = ROOT / "reports/v6/epa_2025_public_model"
LINEAR = ROOT / "reports/v6/epa_2025_public_spatiotemporal"
OUT = ROOT / "reports/v6/epa_2025_public_spatiotemporal_hgb"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(prediction, truth, threshold):
    observed = np.isfinite(truth)
    event = observed & (truth >= threshold)
    return {"rmse": float(np.sqrt(np.mean((prediction[observed] - truth[observed]) ** 2))),
            "event_recall": float(np.mean(prediction[event] >= threshold)) if event.any() else None,
            "event_support": int(event.sum()),
            "cap8_floor": cap_rmse_floor(truth[observed], prediction[observed], 8.),
            "beyond_cap_fraction": float(np.mean(np.abs(truth[observed] - prediction[observed]) > 8.))}


def main() -> None:
    OUT.mkdir(exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    base_lock = json.loads((BASE_PUBLIC / "selection_lock.json").read_text())
    linear_lock = json.loads((LINEAR / "selection_lock.json").read_text())
    family = [
        {"id": "st_hgb_core_h720_u24", "lookback": 720, "spatial": "core"},
        {"id": "st_hgb_core_h2160_u24", "lookback": 2160, "spatial": "core"},
        {"id": "st_hgb_all_h2160_u24", "lookback": 2160, "spatial": "all"},
    ]
    common = {"update_every": 24, "minimum_history": 168, "max_iter": 120,
              "max_leaf_nodes": 15, "learning_rate": .05,
              "l2_regularization": 1., "random_state": 20260908}
    registration = {
        "role": "nonlinear spatiotemporal public selection on exposed development; later partitions unread for selection",
        "hypothesis": "nonlinear regional-spike interactions reduce cap8 tail innovations beyond the locked linear model",
        "family": family, "common": common,
        "selection_partition": PARTITIONS["development"],
        "selection": "lowest observed-entry RMSE; exact tie by id; event and cap metrics retained",
        "information": "locked temporal features plus contemporaneous peer references that exclude the target station",
        "hashes": {str(path): sha(path) for path in (
            Path(__file__), ROOT / "airproof/v6_public_rolling_hgb.py",
            ROOT / "airproof/v6_public_spatial.py",
            ROOT / "airproof/v6_public_archive_model.py", PROTOCOL, DATA,
            BASE_PUBLIC / "selection_lock.json", BASE_PUBLIC / "models.pkl",
            LINEAR / "selection_lock.json")},
        "estimator_validation_read": False, "test_windows_read": False,
        "confirmation_authorized": False,
    }
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2))
    end = PARTITIONS["validation"][1]
    cutoff = pd.Timestamp(protocol["start_utc"]) + pd.Timedelta(hours=end)
    frame = pd.read_parquet(DATA, filters=[("timestamp_utc", "<", cutoff)])
    table = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    table = table.reindex(pd.date_range(table.index.min(), periods=end, freq="h", tz="UTC"))[
        protocol["stations"]]
    values = table.to_numpy()
    coordinates = frame.groupby("station")[["latitude", "longitude"]].first().loc[
        protocol["stations"]].to_numpy()
    with (BASE_PUBLIC / "models.pkl").open("rb") as stream:
        archive = pickle.load(stream)
    medians = archive["medians"]
    temporal = causal_archive_design(values, medians, coordinates)
    feature_sets = {
        "core": np.concatenate((temporal, spatial_public_features(
            values, coordinates, medians, SPATIAL_CORE)), axis=2),
        "all": np.concatenate((temporal, spatial_public_features(
            values, coordinates, medians, SPATIAL_ALL)), axis=2),
    }
    with np.load(LINEAR / "selected_pretest_prediction.npz") as item:
        base = item["prediction"].copy()
    predictions = {}; logs = {}; started = time.perf_counter()
    for candidate in family:
        predictions[candidate["id"]], logs[candidate["id"]] = rolling_stationwise_hgb(
            feature_sets[candidate["spatial"]], values, base,
            start_epoch=7008, lookback=candidate["lookback"], **common)
        print(json.dumps({"candidate": candidate["id"],
                          "blocks": len(logs[candidate["id"]])}), flush=True)
    a, b = PARTITIONS["development"]
    rows = [{"id": candidate["id"],
             "metrics": metrics(predictions[candidate["id"]][a:b], values[a:b],
                                base_lock["event_threshold"])} for candidate in family]
    selected = min(rows, key=lambda row: (row["metrics"]["rmse"], row["id"]))["id"]
    np.savez_compressed(OUT / "selected_pretest_prediction.npz",
                        prediction=predictions[selected], selected=selected)
    result = {"selected": selected, "development": rows, "fit_blocks": logs[selected],
              "elapsed_seconds": time.perf_counter() - started,
              "estimator_validation_metrics_read_for_selection": False,
              "test_metrics_read_for_selection": False,
              "prediction_sha256": sha(OUT / "selected_pretest_prediction.npz"),
              "confirmation_authorized": False}
    (OUT / "selection_lock.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "fit_blocks"}, indent=2))


if __name__ == "__main__":
    main()

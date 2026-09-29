"""Select rolling public regressions with explicit target-excluding peer vectors."""
from __future__ import annotations

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
from airproof.v6_public_peer_features import contemporaneous_peer_features
from airproof.v6_public_rolling import rolling_stationwise_ridge
from scripts.select_v6_epa_2025_estimator import DATA, PARTITIONS, PROTOCOL


BASE_PUBLIC = ROOT / "reports/v6/epa_2025_public_model"
SPATIOTEMPORAL = ROOT / "reports/v6/epa_2025_public_spatiotemporal"
OUT = ROOT / "reports/v6/epa_2025_public_peer_vector"


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
    prior_lock = json.loads((SPATIOTEMPORAL / "selection_lock.json").read_text())
    family = [
        {"id": "peer_vector_h720_p1", "lookback": 720, "penalty": 1.},
        {"id": "peer_vector_h2160_p1", "lookback": 2160, "penalty": 1.},
        {"id": "peer_vector_h2160_p10", "lookback": 2160, "penalty": 10.},
    ]
    registration = {
        "role": "public-model selection on exposed development; later partitions unread for selection",
        "hypothesis": "station-specific current peer vector improves on compressed IDW summaries while excluding self-target values",
        "family": family, "update_every_hours": 24,
        "features": "locked twelve temporal features plus carried current values and availability flags for every other station",
        "selection_partition": PARTITIONS["development"],
        "selection": "lowest observed-entry RMSE; exact tie by id; event and cap metrics retained",
        "hashes": {str(path): sha(path) for path in (
            Path(__file__), ROOT / "airproof/v6_public_peer_features.py",
            ROOT / "airproof/v6_public_rolling.py",
            ROOT / "airproof/v6_public_archive_model.py", PROTOCOL, DATA,
            BASE_PUBLIC / "selection_lock.json", BASE_PUBLIC / "models.pkl",
            SPATIOTEMPORAL / "selection_lock.json")},
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
    with (BASE_PUBLIC / "models.pkl").open("rb") as stream:
        archive = pickle.load(stream)
    temporal = causal_archive_design(values, archive["medians"], archive["coordinates"])
    peers = contemporaneous_peer_features(values, archive["medians"])
    features = np.concatenate((temporal, peers), axis=2)
    with np.load(SPATIOTEMPORAL / "selected_pretest_prediction.npz") as item:
        base = item["prediction"].copy()
    predictions = {}; logs = {}; started = time.perf_counter()
    for candidate in family:
        predictions[candidate["id"]], logs[candidate["id"]] = rolling_stationwise_ridge(
            features, values, base, start_epoch=7008, lookback=candidate["lookback"],
            update_every=24, penalty=candidate["penalty"], minimum_history=168)
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

"""Select a causal nonlinear rolling public model on exposed development only."""
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
from scripts.select_v6_epa_2025_estimator import (DATA, PARTITIONS, PROTOCOL,
    PUBLIC, public_prediction)


OUT = ROOT / "reports/v6/epa_2025_public_rolling_hgb"


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
    base_lock = json.loads((PUBLIC / "selection_lock.json").read_text())
    family = [
        {"id": "rolling_hgb_h720_u168", "lookback": 720, "update_every": 168},
        {"id": "rolling_hgb_h2160_u168", "lookback": 2160, "update_every": 168},
        {"id": "rolling_hgb_h2160_u24", "lookback": 2160, "update_every": 24},
    ]
    common = {"minimum_history": 168, "max_iter": 120, "max_leaf_nodes": 15,
              "learning_rate": .05, "l2_regularization": 1.,
              "random_state": 20260908}
    registration = {
        "role": "nonlinear public-model selection on exposed estimator development; later partitions unread for selection",
        "hypothesis": "causal block refitting reduces public innovations outside the immutable cap8 envelope",
        "family": family, "common": common,
        "selection_partition": PARTITIONS["development"],
        "selection": "lowest observed-entry RMSE; exact tie by id; event and cap metrics retained",
        "information": "twelve causal public features and regulatory targets strictly before each prediction block",
        "hashes": {str(path): sha(path) for path in (
            Path(__file__), ROOT / "airproof/v6_public_rolling_hgb.py",
            ROOT / "airproof/v6_public_archive_model.py", PUBLIC / "models.pkl",
            PUBLIC / "selection_lock.json", PROTOCOL, DATA)},
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
    with (PUBLIC / "models.pkl").open("rb") as stream:
        archive = pickle.load(stream)
    base = public_prediction(values, archive, base_lock["selected"])
    features = causal_archive_design(values, archive["medians"], archive["coordinates"])
    predictions = {}; logs = {}; started = time.perf_counter()
    for candidate in family:
        predictions[candidate["id"]], logs[candidate["id"]] = rolling_stationwise_hgb(
            features, values, base, start_epoch=7008,
            lookback=candidate["lookback"], update_every=candidate["update_every"], **common)
        print(json.dumps({"candidate": candidate["id"], "blocks": len(logs[candidate["id"]])}), flush=True)
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

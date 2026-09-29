"""Select a leave-one-station-out public reference on exposed development."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_feasibility import cap_rmse_floor
from airproof.v6_public_spatial import spatial_public_idw
from scripts.select_v6_epa_2025_estimator import DATA, PARTITIONS, PROTOCOL


PUBLIC = ROOT / "reports/v6/epa_2025_public_model"
OUT = ROOT / "reports/v6/epa_2025_public_spatial"


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
    family = [{"id": f"spatial_k{k}_p{power:g}", "neighbors": k, "power": power}
              for k in (2, 4, 10) for power in (1., 2.)]
    registration = {
        "role": "public-reference selection on exposed estimator development; later partitions unread for selection",
        "family": family, "delay_hours": 0,
        "information": "contemporaneous public regulatory peers; target station current value excluded; no future values",
        "selection_partition": PARTITIONS["development"],
        "selection": "lowest observed-entry RMSE; exact tie by id; event and cap metrics retained",
        "hashes": {str(path): sha(path) for path in (
            Path(__file__), ROOT / "airproof/v6_public_spatial.py", PROTOCOL, DATA,
            PUBLIC / "selection_lock.json", PUBLIC / "models.pkl")},
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
    medians = np.nanmedian(values[:protocol["fit"][1]], axis=0)
    predictions = {candidate["id"]: spatial_public_idw(
        values, coordinates, medians, neighbors=candidate["neighbors"],
        power=candidate["power"], delay=0) for candidate in family}
    a, b = PARTITIONS["development"]
    rows = [{"id": candidate["id"],
             "metrics": metrics(predictions[candidate["id"]][a:b], values[a:b],
                                base_lock["event_threshold"])} for candidate in family]
    selected = min(rows, key=lambda row: (row["metrics"]["rmse"], row["id"]))["id"]
    np.savez_compressed(OUT / "selected_pretest_prediction.npz",
                        prediction=predictions[selected], selected=selected)
    result = {"selected": selected, "development": rows,
              "estimator_validation_metrics_read_for_selection": False,
              "test_metrics_read_for_selection": False,
              "prediction_sha256": sha(OUT / "selected_pretest_prediction.npz"),
              "confirmation_authorized": False}
    (OUT / "selection_lock.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

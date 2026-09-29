"""Register and select causal public-only bias adaptation before estimator validation."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_feasibility import cap_rmse_floor
from airproof.v6_public_online_bias import causal_ewma_bias
from scripts.select_v6_epa_2025_estimator import (DATA, PARTITIONS, PROTOCOL,
    PUBLIC, public_prediction)


OUT = ROOT/"reports/v6/epa_2025_public_adaptation"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(prediction, truth, threshold):
    observed = np.isfinite(truth)
    event = observed & (truth >= threshold)
    return {"rmse": float(np.sqrt(np.mean((prediction[observed]-truth[observed])**2))),
            "event_recall": float(np.mean(prediction[event]>=threshold)) if event.any() else None,
            "event_support": int(event.sum()),
            "cap8_floor": cap_rmse_floor(truth[observed], prediction[observed], 8.),
            "beyond_cap_fraction": float(np.mean(np.abs(truth[observed]-prediction[observed])>8))}


def main():
    protocol = json.loads(PROTOCOL.read_text())
    public_lock = json.loads((PUBLIC/"selection_lock.json").read_text())
    OUT.mkdir(exist_ok=False)
    family = [{"id": "base", "half_life": None}]+[
        {"id": f"ewma_h{hours}", "half_life": hours} for hours in (24, 168, 720)]
    registration = {
        "role": "post-public-selection causal adaptation on exposed estimator-development only; estimator-validation and tests unread",
        "mechanism": "per-station EWMA of lagged public residual, initialized by public-validation mean",
        "family": family, "initialization": protocol["validation"],
        "selection_partition": PARTITIONS["development"],
        "selection": "lowest observed-entry RMSE; exact tie by id; event and cap-floor outcomes retained",
        "hashes": {str(path): digest(path) for path in
                   (Path(__file__), ROOT/"airproof/v6_public_online_bias.py",
                    PUBLIC/"selection_lock.json", PUBLIC/"models.pkl", PROTOCOL, DATA)},
        "estimator_validation_read": False, "test_windows_read": False,
        "confirmation_authorized": False,
    }
    (OUT/"registration.json").write_text(json.dumps(registration, indent=2))
    end = PARTITIONS["validation"][1]
    cutoff = pd.Timestamp(protocol["start_utc"])+pd.Timedelta(hours=end)
    frame = pd.read_parquet(DATA, filters=[("timestamp_utc", "<", cutoff)])
    table = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    table = table.reindex(pd.date_range(table.index.min(), periods=end, freq="h", tz="UTC"))[protocol["stations"]]
    values = table.to_numpy()
    with (PUBLIC/"models.pkl").open("rb") as stream:
        archive = pickle.load(stream)
    base = public_prediction(values, archive, public_lock["selected"])
    init_start, init_end = protocol["validation"]
    initial_bias = np.nanmean(values[init_start:init_end]-base[init_start:init_end], axis=0)
    predictions = {"base": base}
    for candidate in family[1:]:
        predictions[candidate["id"]], _ = causal_ewma_bias(base, values,
            start_epoch=init_end, initial_bias=initial_bias,
            half_life=candidate["half_life"])
    start, stop = PARTITIONS["development"]
    rows = [{"id": candidate["id"],
             "metrics": metrics(predictions[candidate["id"]][start:stop],
                                values[start:stop], public_lock["event_threshold"])}
            for candidate in family]
    selected = min(rows, key=lambda row: (row["metrics"]["rmse"], row["id"]))["id"]
    lock = {"selected": selected, "development": rows,
            "initial_bias": initial_bias.tolist(),
            "estimator_validation_metrics_read_for_selection": False,
            "test_metrics_read_for_selection": False, "confirmation_authorized": False}
    (OUT/"selection_lock.json").write_text(json.dumps(lock, indent=2))
    np.savez_compressed(OUT/"selected_pretest_prediction.npz",
                        prediction=predictions[selected], selected=selected)
    lock["prediction_sha256"] = digest(OUT/"selected_pretest_prediction.npz")
    (OUT/"selection_lock.json").write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

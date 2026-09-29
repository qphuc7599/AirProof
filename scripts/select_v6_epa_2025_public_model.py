"""Select a strong causal public model on the frozen EPA-2025 validation only."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from airproof.v6_feasibility import cap_rmse_floor
from airproof.v6_public_archive_model import (causal_archive_design,
    fit_stationwise_ridge, predict_stationwise_ridge)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"reports/v6/epa_2025_protocol/registration.json"
DATA = ROOT/"data/external/epa/epa_south_coast_pm25_2025.parquet"
OUT = ROOT/"reports/v6/epa_2025_public_model"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(prediction, truth, event_threshold):
    observed = np.isfinite(truth)
    event = observed & (truth >= event_threshold)
    return {"rmse": float(np.sqrt(np.mean((prediction[observed]-truth[observed])**2))),
            "event_recall": float(np.mean(prediction[event]>=event_threshold)) if event.any() else None,
            "event_support": int(event.sum()),
            "cap8_floor": cap_rmse_floor(truth[observed], prediction[observed], 8.)}


def main():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["errors_inspected"] or protocol["model_or_estimator_selected"]:
        raise ValueError("expected an unscored protocol without a selected model")
    OUT.mkdir(exist_ok=False)
    family = [
        {"id": "persistence", "kind": "fixed"},
        {"id": "ridge_compact", "kind": "ridge", "features": [0, 4, 6, 8, 9]},
        {"id": "ridge_expanded", "kind": "ridge", "features": list(range(12))},
        {"id": "hgb_expanded", "kind": "hist_gradient_boosting", "features": list(range(12)),
         "max_iter": 200, "max_leaf_nodes": 15, "learning_rate": .05, "l2_regularization": 1.},
    ]
    registration = {
        "role": "public-model selection using fit and validation only; test windows remain unread",
        "protocol_sha256": digest(PROTOCOL), "data_sha256": digest(DATA),
        "source_sha256": digest(Path(__file__)),
        "feature_source_sha256": digest(ROOT/"airproof/v6_public_archive_model.py"),
        "family": family,
        "features": "own lags1,2,3,6,24,168; peer and four-neighbor lag1; daily and weekly Fourier terms",
        "fitting": protocol["fit"], "validation": protocol["validation"],
        "test_windows_read": False,
        "selection": "lowest observed-entry validation RMSE; exact tie by method id; event and cap-floor metrics fully reported",
        "event_threshold": "95th percentile of observed fit targets",
        "confirmation_authorized": False,
    }
    (OUT/"registration.json").write_text(json.dumps(registration, indent=2))
    frame = pd.read_parquet(DATA)
    table = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    table = table.reindex(pd.date_range(table.index.min(), table.index.max(), freq="h", tz="UTC"))
    table = table[protocol["stations"]]
    coordinates = frame.groupby("station")[["latitude", "longitude"]].first().loc[protocol["stations"]].to_numpy()
    train_end = protocol["fit"][1]
    validation_start, validation_end = protocol["validation"]
    # Truncation makes reading any future test value impossible in this process.
    values = table.iloc[:validation_end].to_numpy()
    medians = np.nanmedian(values[:train_end], axis=0)
    features = causal_archive_design(values, medians, coordinates)
    event_threshold = float(np.nanquantile(values[:train_end], .95))
    truth = values[validation_start:validation_end]
    models = {}
    rows = []
    inclusive = np.where(np.isfinite(values), values, np.nan)
    previous = medians.copy(); persistence = np.empty_like(values)
    for epoch in range(len(values)):
        persistence[epoch] = previous
        previous = np.where(np.isfinite(values[epoch]), values[epoch], previous)
    predictions = {"persistence": persistence}
    for candidate in family[1:3]:
        model = fit_stationwise_ridge(features, values, train_end,
                                      candidate["features"], penalty=1.)
        models[candidate["id"]] = model
        predictions[candidate["id"]] = predict_stationwise_ridge(features, model)
    hgb_models = []
    hgb_prediction = np.empty_like(values)
    hgb = family[3]
    for station in range(values.shape[1]):
        valid = np.isfinite(values[:train_end, station]) & (np.arange(train_end) >= 168)
        model = HistGradientBoostingRegressor(max_iter=hgb["max_iter"],
            max_leaf_nodes=hgb["max_leaf_nodes"], learning_rate=hgb["learning_rate"],
            l2_regularization=hgb["l2_regularization"], early_stopping=False,
            random_state=20260908)
        model.fit(features[:train_end, station][valid], values[:train_end, station][valid])
        hgb_models.append(model)
        hgb_prediction[:, station] = np.maximum(model.predict(features[:, station]), 0.)
    models["hgb_expanded"] = hgb_models; predictions["hgb_expanded"] = hgb_prediction
    for candidate in family:
        prediction = predictions[candidate["id"]][validation_start:validation_end]
        rows.append({"id": candidate["id"], "metrics": metrics(prediction, truth, event_threshold)})
        np.savez_compressed(OUT/f"{candidate['id']}_validation.npz", prediction=prediction)
    selected = min(rows, key=lambda row: (row["metrics"]["rmse"], row["id"]))["id"]
    with (OUT/"models.pkl").open("wb") as stream:
        pickle.dump({"models": models, "medians": medians, "coordinates": coordinates,
                     "stations": protocol["stations"], "family": family}, stream, protocol=5)
    lock = {"selected": selected, "validation": rows, "event_threshold": event_threshold,
            "test_metrics_read_for_selection": False, "models_sha256": digest(OUT/"models.pkl"),
            "confirmation_authorized": False}
    (OUT/"selection_lock.json").write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

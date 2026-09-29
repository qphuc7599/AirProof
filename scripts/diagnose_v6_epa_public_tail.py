"""Explain rolling-public errors outside the immutable correction envelope.

The exposed estimator-development partition is used only for mechanism
diagnosis.  Neither estimator validation nor the two EPA test windows is read.
"""
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

from scripts.select_v6_epa_2025_estimator import (DATA, PARTITIONS, PROTOCOL,
    PUBLIC, public_prediction)


ROLLING = ROOT / "reports/v6/epa_2025_public_rolling"
ENSEMBLE = ROOT / "reports/v6/epa_2025_public_ensemble_v2"
OUT = ROOT / "reports/v6/epa_2025_public_tail_diagnostic"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(prediction, truth, mask):
    error = prediction[mask] - truth[mask]
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)), "support": int(mask.sum())}


def main() -> None:
    OUT.mkdir(exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    rolling_lock = json.loads((ROLLING / "selection_lock.json").read_text())
    ensemble_lock = json.loads((ENSEMBLE / "selection_lock.json").read_text())
    base_lock = json.loads((PUBLIC / "selection_lock.json").read_text())
    registration = {
        "role": "exposed-development mechanism diagnosis; no selection",
        "question": "which observable public expert/trend states explain rolling errors beyond cap8?",
        "partition": PARTITIONS["development"],
        "hashes": {str(path): sha(path) for path in (
            Path(__file__), PROTOCOL, DATA, PUBLIC / "models.pkl",
            PUBLIC / "selection_lock.json", ROLLING / "selection_lock.json",
            ROLLING / "selected_pretest_prediction.npz",
            ENSEMBLE / "selection_lock.json", ENSEMBLE / "selected_pretest_prediction.npz")},
        "estimator_validation_read": False, "test_windows_read": False,
        "selection_permitted": False,
    }
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2))
    end = PARTITIONS["development"][1]
    cutoff = pd.Timestamp(protocol["start_utc"]) + pd.Timedelta(hours=end)
    frame = pd.read_parquet(DATA, filters=[("timestamp_utc", "<", cutoff)])
    table = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    table = table.reindex(pd.date_range(table.index.min(), periods=end, freq="h", tz="UTC"))[
        protocol["stations"]]
    values = table.to_numpy()
    with (PUBLIC / "models.pkl").open("rb") as stream:
        archive = pickle.load(stream)
    hgb = public_prediction(values, archive, base_lock["selected"])
    previous = archive["medians"].copy()
    persistence = np.empty_like(values)
    for epoch in range(len(values)):
        persistence[epoch] = previous
        previous = np.where(np.isfinite(values[epoch]), values[epoch], previous)
    persistence = np.maximum(persistence, 0.)
    with np.load(ROLLING / "selected_pretest_prediction.npz") as item:
        rolling = item["prediction"][:end]
    with np.load(ENSEMBLE / "selected_pretest_prediction.npz") as item:
        ensemble = item["prediction"][:end]
    a, b = PARTITIONS["development"]
    truth = values[a:b]
    experts = {"rolling": rolling[a:b], "persistence": persistence[a:b],
               "hgb": hgb[a:b], "hgb_persistence": ensemble[a:b]}
    observed = np.isfinite(truth)
    rolling_error = experts["rolling"] - truth
    tail = observed & (np.abs(rolling_error) > 8.)
    under = tail & (rolling_error < 0.)
    over = tail & (rolling_error > 0.)
    latest_change = persistence[a:b] - persistence[a-1:b-1]
    rising = tail & (latest_change > 3.)
    falling = tail & (latest_change < -3.)
    stable = tail & ~(rising | falling)
    result = {
        "locks": {"rolling": rolling_lock["selected"], "ensemble": ensemble_lock["selected"],
                  "base": base_lock["selected"]},
        "overall": {name: score(prediction, truth, observed)
                    for name, prediction in experts.items()},
        "rolling_tail_fraction": float(tail.sum() / observed.sum()),
        "tail": {name: score(prediction, truth, tail) for name, prediction in experts.items()},
        "underprediction_tail": {name: score(prediction, truth, under)
                                 for name, prediction in experts.items()},
        "overprediction_tail": {name: score(prediction, truth, over)
                                for name, prediction in experts.items()},
        "tail_by_public_trend": {label: {name: score(prediction, truth, mask)
                                         for name, prediction in experts.items()}
                                 for label, mask in (("rising_gt3", rising),
                                                     ("falling_lt_minus3", falling),
                                                     ("stable", stable))},
        "oracle_public_expert_rmse": float(np.sqrt(np.mean(np.min(np.stack([
            (prediction[observed] - truth[observed]) ** 2 for prediction in experts.values()]), axis=0)))),
        "estimator_validation_read": False, "test_windows_read": False,
        "selection_permitted": False,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

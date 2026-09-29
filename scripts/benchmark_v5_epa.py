"""Frozen causal EPA wrapper: preflight first, explicit selected-model scoring later."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airproof.v5_estimator import EstimatorConfig, enumerate_candidates, estimate_public_field
from airproof.predictor_wrapper import synthetic_citizen_replay

TRAIN_END = 7032
VALIDATION = (7056, 7920)
WINDOWS = ((7944, 8280), (8448, 8784))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_stations(inventory):
    windows = inventory["epa_candidate_windows"]
    if [tuple(w["indices"]) for w in windows] != list(WINDOWS):
        raise ValueError("inventory windows differ from registered chronology")
    return sorted(set.intersection(*(set(w["stations_at_least_80pct"]) for w in windows)))


def design(values, medians):
    """At t every feature uses timestamps strictly before t or public calendar."""
    values = np.asarray(values, float)
    filled = np.empty_like(values)
    previous = np.array(medians, float)
    for t in range(len(values)):
        filled[t] = previous
        previous = np.where(np.isfinite(values[t]), values[t], previous)
    lag24 = np.broadcast_to(medians, values.shape).copy()
    if len(values) > 24:
        # filled[t-23] is the latest value available through t-24.
        lag24[24:] = filled[1:-23]
    t = np.arange(len(values))
    features = np.stack((np.ones_like(filled), filled, lag24,
                         np.broadcast_to(filled.mean(1)[:, None], filled.shape),
                         np.broadcast_to(np.sin(2*np.pi*t/24)[:, None], filled.shape),
                         np.broadcast_to(np.cos(2*np.pi*t/24)[:, None], filled.shape)), axis=-1)
    return features


def fit_public(values, train_end=TRAIN_END):
    training = np.asarray(values, float)[:train_end]
    medians = np.nanmedian(training, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("each selected station needs observed training support")
    features = design(training, medians)
    coefficients = []
    for station in range(training.shape[1]):
        valid = np.isfinite(training[:, station]) & (np.arange(len(training)) >= 24)
        x, y = features[valid, station], training[valid, station]
        penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
        coefficients.append(np.linalg.solve(x.T@x+penalty, x.T@y))
    return medians, np.array(coefficients)


def predict_public(values, medians, coefficients):
    return np.maximum(np.einsum("tsf,sf->ts", design(values, medians), coefficients), 0.)


def observed_rmse(predictions, truth):
    mask = np.isfinite(truth)
    return float(np.sqrt(np.mean((predictions[mask]-truth[mask])**2)))


def load_archive(inventory_path):
    inventory = json.loads(inventory_path.read_text())
    archive = ROOT/inventory["epa_dataset"]["path"]
    if digest(archive) != inventory["epa_dataset"]["sha256"]:
        raise ValueError("archive hash changed")
    stations = common_stations(inventory)
    frame = pd.read_parquet(archive)
    table = frame.pivot(index="timestamp_utc", columns="station", values="value").sort_index()[stations]
    if len(table) != 8784 or not np.all(np.diff(table.index.asi8) == 3600*10**9):
        raise ValueError("registered complete hourly timestamp axis required")
    coordinates = frame.groupby("station")[["latitude", "longitude"]].first().loc[stations].to_numpy()
    return archive, stations, table, coordinates


def preflight(inventory_path, output):
    archive, stations, table, coordinates = load_archive(inventory_path)
    output.mkdir(parents=True, exist_ok=False)
    # Never compute test predictions or test errors in this mode.
    fitting_values = table.iloc[:VALIDATION[1]].to_numpy()
    medians, coefficients = fit_public(fitting_values)
    prediction = predict_public(fitting_values, medians, coefficients)
    np.savez_compressed(output/"frozen_public_model.npz", medians=medians, coefficients=coefficients,
                        coordinates=coordinates, stations=np.array(stations))
    validation_truth = fitting_values[slice(*VALIDATION)]
    manifest = {"role": "two inventoried unscored windows in previously used EPA archive; causal wrapper diagnostic",
                "archive": str(archive), "archive_sha256": digest(archive), "inventory_sha256": digest(inventory_path),
                "train_indices": [0, TRAIN_END], "validation_indices": list(VALIDATION),
                "purge_hours": 24, "test_windows": WINDOWS, "stations": stations,
                "station_rule": "sorted intersection of >=80% missingness support in both inventory windows",
                "public_model": "stationwise ridge1 AR with lag1,lag24,peer-lag1 mean,daily sin/cos; intercept unpenalized",
                "model_fit_count": 1, "model_selection": "none; no compatible saved EPA backbone weights identified",
                "missing_features": "causal forward fill; training-only station median initial fallback",
                "scoring_mask": "observed targets only; identical stations/times for every method",
                "public_information": "EPA history strictly before t; no current target in public predictor",
                "additional_information": "synthetic citizens generated from archive truth; not real citizen measurements",
                "methods": ["PUBLIC", "SQ", "AP", "HUBER"],
                "estimator_selection": "must be supplied from synthetic development lock; no EPA candidate search",
                "test_scored": False, "source_sha256": digest(__file__),
                "frozen_model_sha256": digest(output/"frozen_public_model.npz"),
                "validation_public_rmse": observed_rmse(prediction[slice(*VALIDATION)], validation_truth),
                "validation_observed_entries": int(np.isfinite(validation_truth).sum())}
    (output/"protocol_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"mode": "preflight", "stations": len(stations), "test_scored": False,
                      "validation_public_rmse": manifest["validation_public_rmse"]}))


def score(inventory_path, output, selected_config, seeds):
    if selected_config is None:
        raise ValueError("explicit locked selected estimator JSON is required to score")
    manifest = json.loads((output/"protocol_manifest.json").read_text())
    if digest(inventory_path) != manifest["inventory_sha256"]:
        raise ValueError("frozen archive inventory changed")
    if digest(output/"frozen_public_model.npz") != manifest["frozen_model_sha256"]:
        raise ValueError("frozen public model changed")
    config = EstimatorConfig(**json.loads(selected_config.read_text()))
    if config not in enumerate_candidates():
        raise ValueError("selection must match exactly one registered candidate")
    archive, stations, table, _ = load_archive(inventory_path)
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("preflight archive identity changed")
    if stations != manifest["stations"]:
        raise ValueError("station mask changed")
    model = np.load(output/"frozen_public_model.npz")
    public = predict_public(table.to_numpy(), model["medians"], model["coefficients"])
    xy = model["coordinates"]
    xy = (xy-xy.mean(0))*np.array([111.2, 111.2*np.cos(np.deg2rad(xy[:, 0].mean()))])
    scored = output/"scored"; scored.mkdir(exist_ok=False)
    lock = {"selected_config": asdict(config), "selected_config_sha256": digest(selected_config),
            "seeds": seeds, "role": manifest["role"], "protocol_sha256": digest(output/"protocol_manifest.json"),
            "huber_max_irls": max(200, config.max_irls),
            "channel_proxy": "max(signed observed target,0); original scoring mask/target retained",
            "execution_source_sha256": digest(__file__)}
    (scored/"score_manifest.json").write_text(json.dumps(lock, indent=2))
    controls = {"AP": config, "SQ": replace(config, input_clip=False, output_cap=False),
                "HUBER": replace(config, input_clip=False, output_cap=False, loss="huber", max_irls=max(200, config.max_irls))}
    rows = []
    for start, end in WINDOWS:
        truth = table.iloc[start:end].to_numpy()
        reference = public[start:end]
        for seed in seeds:
            for kind in ("clean", "adversarial_drift", "negative_bias"):
                # Missing target entries never generate synthetic observations.
                # Signed observed targets remain unchanged for scoring; only latent citizen inputs are nonnegative.
                channel_truth = np.maximum(np.where(np.isfinite(truth), truth, reference), 0.)
                stream = synthetic_citizen_replay(channel_truth, seed=seed, kind=kind)
                stream = [r for r in stream if np.isfinite(truth[r.epoch, r.cell])]
                for name, cfg in {"PUBLIC": None, **controls}.items():
                    result = None if cfg is None else estimate_public_field(reference, xy, stream, cfg)
                    for clock in ("live", "reconstructed"):
                        estimate = reference if result is None else getattr(result, clock)
                        rows.append({"window": [start, end], "seed": seed, "kind": kind,
                                     "method": name, "clock": clock, "rmse": observed_rmse(estimate, truth),
                                     "observed_entries": int(np.isfinite(truth).sum()),
                                     "solver_failure_rate": 0 if result is None else result.diagnostics["solver_failure_rate"]})
    (scored/"results.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(json.dumps({"mode": "scored", "rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT/"configs/v5/archive_usage.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"reports/v5/epa")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--selected-config", type=Path)
    parser.add_argument("--seeds", default="916002,916003,916004,916005")
    args = parser.parse_args()
    if args.score:
        score(args.inventory, args.output_dir, args.selected_config, [int(s) for s in args.seeds.split(",")])
    else:
        preflight(args.inventory, args.output_dir)


if __name__ == "__main__":
    main()

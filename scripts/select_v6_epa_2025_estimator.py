"""Select and validate bounded Huber score on pre-test EPA-2025 partitions."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd

from airproof.predictor_wrapper import synthetic_citizen_replay
from airproof.v6_estimator import EstimatorConfig, estimate_public_field
from airproof.v6_public_archive_model import causal_archive_design, predict_stationwise_ridge


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"reports/v6/epa_2025_protocol/registration.json"
PUBLIC = ROOT/"reports/v6/epa_2025_public_model"
DATA = ROOT/"data/external/epa/epa_south_coast_pm25_2025.parquet"
OUT = ROOT/"reports/v6/epa_2025_estimator"
PARTITIONS = {"development": [7032, 7440], "validation": [7464, 7896]}
WARMUP = 24
DRAIN = 24


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_prediction(values, archive, selected):
    features = causal_archive_design(values, archive["medians"], archive["coordinates"])
    if selected == "persistence":
        previous = archive["medians"].copy(); prediction = np.empty_like(values)
        for epoch in range(len(values)):
            prediction[epoch] = previous
            previous = np.where(np.isfinite(values[epoch]), values[epoch], previous)
        return prediction
    if selected.startswith("ridge_"):
        return predict_stationwise_ridge(features, archive["models"][selected])
    if selected == "hgb_expanded":
        prediction = np.column_stack([np.maximum(model.predict(features[:, station]), 0.)
                                      for station, model in enumerate(archive["models"][selected])])
        return prediction
    raise ValueError("unknown locked public model")


def summarize(rows, candidates):
    output = []
    kinds = ("clean", "adversarial_drift", "negative_bias")
    for candidate in candidates:
        identifier = candidate["id"]
        robust = [row for row in rows if row["method"] == identifier]
        squared = [row for row in rows if row["method"] == "quadratic"]
        means = {kind: float(np.mean([row["rmse"] for row in robust if row["kind"] == kind])) for kind in kinds}
        sq_means = {kind: float(np.mean([row["rmse"] for row in squared if row["kind"] == kind])) for kind in kinds}
        attacks = {}
        for kind in kinds[1:]:
            robust_growth = means[kind]-means["clean"]
            squared_growth = sq_means[kind]-sq_means["clean"]
            attenuation = 1-robust_growth/squared_growth if squared_growth > 0 else None
            attacks[kind] = {"robust_growth": robust_growth, "quadratic_growth": squared_growth,
                             "attenuation": attenuation,
                             "passes": attenuation is not None and attenuation >= .2}
        recall_losses = {kind: float(np.mean([row["event_recall"] for row in squared if row["kind"] == kind])
                                          -np.mean([row["event_recall"] for row in robust if row["kind"] == kind]))
                         for kind in kinds}
        ratio = means["clean"]/sq_means["clean"]
        numerical = all(row["solver_failure_rate"] == 0 for row in robust)
        eligible = ratio <= 1.05 and all(item["passes"] for item in attacks.values()) and max(recall_losses.values()) <= .05 and numerical
        output.append({"id": identifier, "clean_rmse": means["clean"],
                       "quadratic_clean_rmse": sq_means["clean"], "clean_ratio": ratio,
                       "attack": attacks, "event_recall_loss": recall_losses,
                       "numerical_pass": numerical, "eligible": bool(eligible)})
    return output


def run_partition(name, span, seeds, candidates, values, public, coordinates, threshold, base):
    start, end = span
    context_start = start-WARMUP
    truth = values[context_start:end]
    reference = public[context_start:end]
    extended = np.concatenate((reference, np.repeat(reference[-1:], DRAIN, axis=0)))
    score_slice = slice(WARMUP, len(truth))
    rows = []
    path = OUT/f"{name}.jsonl"
    with path.open("w") as stream:
        for seed in seeds:
            channel_truth = np.maximum(np.where(np.isfinite(truth), truth, reference), 0.)
            for kind in ("clean", "adversarial_drift", "negative_bias"):
                records = synthetic_citizen_replay(channel_truth, seed=seed, kind=kind)
                records = [item for item in records if np.isfinite(truth[item.epoch, item.cell])]
                methods = [(candidate["id"], replace(base, huber_delta=candidate["delta"])) for candidate in candidates]
                methods.append(("quadratic", replace(base, loss="quadratic", output_cap=False)))
                for identifier, config in methods:
                    result = estimate_public_field(extended, coordinates, records, config,
                                                   innovation_scales=3.)
                    estimate = result.reconstructed[:len(truth)][score_slice]
                    live = result.live[:len(truth)][score_slice]
                    target = truth[score_slice]
                    observed = np.isfinite(target)
                    event = observed & (target >= threshold)
                    row = {"partition": name, "seed": seed, "kind": kind, "method": identifier,
                           "rmse": float(np.sqrt(np.mean((estimate[observed]-target[observed])**2))),
                           "event_recall": float(np.mean(live[event]>=threshold)) if event.any() else None,
                           "event_support": int(event.sum()),
                           "solver_failure_rate": result.diagnostics["solver_failure_rate"]}
                    rows.append(row); stream.write(json.dumps(row)+"\n"); stream.flush()
            print(json.dumps({"partition": name, "seed": seed, "rows": len(rows)}), flush=True)
    return rows


def main():
    protocol = json.loads(PROTOCOL.read_text())
    public_lock = json.loads((PUBLIC/"selection_lock.json").read_text())
    if public_lock["test_metrics_read_for_selection"]:
        raise ValueError("public model selection is not pre-test")
    OUT.mkdir(exist_ok=False)
    candidates = [{"id": f"huber_d{delta:g}", "delta": delta}
                  for delta in (1.345, 2., 3., 4.)]
    base = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05, lambda_spatial=.02,
                           lag=6, time_step=1., numerical_refinements=2,
                           loss="huber", output_cap=True)
    registration = {
        "role": "bounded estimator development and validation before both frozen EPA test windows",
        "protocol_sha256": digest(PROTOCOL),
        "public_selection_sha256": digest(PUBLIC/"selection_lock.json"),
        "public_models_sha256": digest(PUBLIC/"models.pkl"),
        "source_sha256": digest(Path(__file__)), "estimator_source_sha256": digest(ROOT/"airproof/v6_estimator.py"),
        "public_model": public_lock["selected"], "partitions": PARTITIONS,
        "purges": [[7008, 7032], [7440, 7464], [7896, 7920]],
        "warmup_hours": WARMUP, "drain_hours": DRAIN,
        "candidate_family": candidates, "base_config": asdict(base),
        "development_seeds": list(range(917000, 917008)),
        "validation_seeds": list(range(917100, 917112)),
        "selection_gates": {"clean_ratio": 1.05, "attack_attenuation": .2,
                            "event_recall_loss_each_condition": .05, "solver_failure_rate": 0.},
        "selection": "eligible only; lowest clean RMSE, then smaller delta",
        "test_windows_read": False, "confirmation_authorized": False,
    }
    (OUT/"registration.json").write_text(json.dumps(registration, indent=2))
    end = PARTITIONS["validation"][1]
    cutoff = pd.Timestamp(protocol["start_utc"])+pd.Timedelta(hours=end)
    full = pd.read_parquet(DATA, filters=[("timestamp_utc", "<", cutoff)])
    table = full.pivot(index="timestamp_utc", columns="station", values="value").sort_index()
    table = table.reindex(pd.date_range(table.index.min(), periods=end, freq="h", tz="UTC"))[protocol["stations"]]
    values = table.to_numpy()
    with (PUBLIC/"models.pkl").open("rb") as stream:
        archive = pickle.load(stream)
    public = public_prediction(values, archive, public_lock["selected"])
    coordinates = archive["coordinates"]
    coordinates = (coordinates-coordinates.mean(0))*np.array(
        [111.2, 111.2*np.cos(np.deg2rad(coordinates[:, 0].mean()))])
    threshold = public_lock["event_threshold"]
    started = time.perf_counter()
    development_rows = run_partition("development", PARTITIONS["development"],
        registration["development_seeds"], candidates, values, public, coordinates, threshold, base)
    development = summarize(development_rows, candidates)
    eligible = [row for row in development if row["eligible"]]
    selected = min(eligible, key=lambda row: (row["clean_rmse"], float(row["id"].split("d")[1]))) ["id"] if eligible else None
    lock = {"selected": selected, "development": development,
            "validation_metrics_read_for_selection": False,
            "test_metrics_read_for_selection": False, "confirmation_authorized": False}
    (OUT/"selection_lock.json").write_text(json.dumps(lock, indent=2))
    validation_rows = []
    validation = []
    if selected is not None:
        chosen = [candidate for candidate in candidates if candidate["id"] == selected]
        validation_rows = run_partition("validation", PARTITIONS["validation"],
            registration["validation_seeds"], chosen, values, public, coordinates, threshold, base)
        validation = summarize(validation_rows, chosen)
    result = {"selected": selected, "development": development, "validation": validation,
              "development_rows": len(development_rows), "validation_rows": len(validation_rows),
              "validation_passes": bool(validation and validation[0]["eligible"]),
              "elapsed_seconds": time.perf_counter()-started,
              "confirmation_authorized": bool(validation and validation[0]["eligible"])}
    (OUT/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

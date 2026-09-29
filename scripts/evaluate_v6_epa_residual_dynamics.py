"""Registered residual-AR repair after the EPA Huber-delta family selected none."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from dataclasses import replace
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
from airproof.v6_residual_dynamics import fit_frozen_residual_ar
from scripts.select_v6_epa_2025_estimator import (DATA, PARTITIONS, PROTOCOL,
    PUBLIC, DRAIN, WARMUP, public_prediction, summarize)


PREVIOUS = ROOT/"reports/v6/epa_2025_estimator"
OUT = ROOT/"reports/v6/epa_2025_residual_ar"
CANDIDATE = {"id": "huber_d2_ar", "delta": 2.}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_partition(name, span, seeds, values, public, coordinates, threshold, base, dynamics):
    start, end = span
    context_start = start-WARMUP
    truth = values[context_start:end]
    reference = public[context_start:end]
    extended = np.concatenate((reference, np.repeat(reference[-1:], DRAIN, axis=0)))
    transitions, forcing = dynamics.system(len(extended))
    score_slice = slice(WARMUP, len(truth))
    rows = []
    with (OUT/f"{name}.jsonl").open("w") as stream:
        for seed in seeds:
            channel_truth = np.maximum(np.where(np.isfinite(truth), truth, reference), 0.)
            for kind in ("clean", "adversarial_drift", "negative_bias"):
                records = synthetic_citizen_replay(channel_truth, seed=seed, kind=kind)
                records = [item for item in records if np.isfinite(truth[item.epoch, item.cell])]
                methods = [(CANDIDATE["id"], replace(base, loss="huber", output_cap=True)),
                           ("quadratic", replace(base, loss="quadratic", output_cap=False))]
                for identifier, config in methods:
                    result = estimate_public_field(extended, coordinates, records, config,
                        innovation_scales=3., transitions=transitions, residual_forcing=forcing)
                    estimate = result.reconstructed[:len(truth)][score_slice]
                    live = result.live[:len(truth)][score_slice]
                    target = truth[score_slice]
                    observed = np.isfinite(target)
                    event = observed & (target >= threshold)
                    row = {"partition": name, "seed": seed, "kind": kind,
                           "method": identifier,
                           "rmse": float(np.sqrt(np.mean((estimate[observed]-target[observed])**2))),
                           "event_recall": float(np.mean(live[event]>=threshold)) if event.any() else None,
                           "event_support": int(event.sum()),
                           "solver_failure_rate": result.diagnostics["solver_failure_rate"]}
                    rows.append(row); stream.write(json.dumps(row)+"\n"); stream.flush()
            print(json.dumps({"partition": name, "seed": seed, "rows": len(rows)}), flush=True)
    return rows


def main():
    previous = json.loads((PREVIOUS/"result.json").read_text())
    if previous["selected"] is not None or previous["validation_rows"] != 0:
        raise ValueError("expected a closed null-selection delta family")
    protocol = json.loads(PROTOCOL.read_text())
    public_lock = json.loads((PUBLIC/"selection_lock.json").read_text())
    OUT.mkdir(exist_ok=False)
    registration = {
        "role": "post-delta registered residual-dynamics development; EPA test windows remain unread",
        "reason": "delta2 preserves attacks/events but zero-forcing residual prior fails clean margin",
        "candidate": CANDIDATE,
        "residual_calibration": protocol["validation"],
        "residual_model": "per-station centered AR1, ridge0.1, coefficient clipped to[0,.99], minimum100 consecutive pairs",
        "measurement_scale": 3.,
        "scale_interpretation": "known synthetic sensor likelihood noise; public residual error enters dynamics rather than being double-counted as sensor noise",
        "partitions": PARTITIONS, "development_seeds": list(range(917200, 917208)),
        "validation_seeds": list(range(917300, 917312)),
        "matched_control": "unbounded SQ receives identical fitted transitions, forcing, records, public center and clock",
        "selection_gates": {"clean_ratio": 1.05, "attack_attenuation": .2,
                            "event_recall_loss_each_condition": .05, "solver_failure_rate": 0.},
        "hashes": {str(path): digest(path) for path in
                   (Path(__file__), ROOT/"airproof/v6_residual_dynamics.py",
                    ROOT/"airproof/v6_estimator.py", PUBLIC/"selection_lock.json",
                    PUBLIC/"models.pkl", PREVIOUS/"result.json", PROTOCOL, DATA)},
        "test_windows_read": False, "confirmation_authorized": False,
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
    public = public_prediction(values, archive, public_lock["selected"])
    fit_start, fit_end = protocol["validation"]
    dynamics = fit_frozen_residual_ar(public, values, training_start=fit_start,
                                      training_end=fit_end, ridge=.1, minimum_pairs=100)
    np.savez_compressed(OUT/"frozen_residual_ar.npz", mean=dynamics.mean,
        coefficient=dynamics.coefficient, pair_counts=dynamics.pair_counts,
        training_start=dynamics.training_start, training_end=dynamics.training_end)
    registration["frozen_dynamics_sha256"] = digest(OUT/"frozen_residual_ar.npz")
    (OUT/"registration.json").write_text(json.dumps(registration, indent=2))
    coordinates = archive["coordinates"]
    coordinates = (coordinates-coordinates.mean(0))*np.array(
        [111.2, 111.2*np.cos(np.deg2rad(coordinates[:, 0].mean()))])
    base = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05, lambda_spatial=.02,
                           lag=6, time_step=1., numerical_refinements=2,
                           huber_delta=2.)
    started = time.perf_counter()
    development_rows = run_partition("development", PARTITIONS["development"],
        registration["development_seeds"], values, public, coordinates,
        public_lock["event_threshold"], base, dynamics)
    development = summarize(development_rows, [CANDIDATE])
    selected = CANDIDATE["id"] if development[0]["eligible"] else None
    lock = {"selected": selected, "development": development,
            "validation_metrics_read_for_selection": False,
            "test_metrics_read_for_selection": False, "confirmation_authorized": False}
    (OUT/"selection_lock.json").write_text(json.dumps(lock, indent=2))
    validation_rows = []
    validation = []
    if selected is not None:
        validation_rows = run_partition("validation", PARTITIONS["validation"],
            registration["validation_seeds"], values, public, coordinates,
            public_lock["event_threshold"], base, dynamics)
        validation = summarize(validation_rows, [CANDIDATE])
    result = {"selected": selected, "development": development,
              "validation": validation, "development_rows": len(development_rows),
              "validation_rows": len(validation_rows),
              "validation_passes": bool(validation and validation[0]["eligible"]),
              "residual_ar": {"mean": dynamics.mean.tolist(),
                              "coefficient": dynamics.coefficient.tolist(),
                              "pair_counts": dynamics.pair_counts.tolist()},
              "elapsed_seconds": time.perf_counter()-started,
              "confirmation_authorized": bool(validation and validation[0]["eligible"])}
    (OUT/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Registered public-residual shrinkage after delta and residual-AR null selections."""
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
from scripts.select_v6_epa_2025_estimator import (DATA, PARTITIONS, PROTOCOL,
    PUBLIC, DRAIN, WARMUP, public_prediction)


PREVIOUS = ROOT/"reports/v6/epa_2025_estimator"
AR_PREVIOUS = ROOT/"reports/v6/epa_2025_residual_ar"
OUT = ROOT/"reports/v6/epa_2025_shrinkage"
ALPHAS = (.5, .75, 1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    kinds = ("clean", "adversarial_drift", "negative_bias")
    output = []
    for alpha in ALPHAS:
        identifier = f"huber_d2_a{alpha:g}"
        robust = [row for row in rows if row["method"] == identifier]
        squared = [row for row in rows if row["method"] == "quadratic"]
        public = [row for row in rows if row["method"] == "public_only"]
        means = {kind: float(np.mean([row["rmse"] for row in robust if row["kind"] == kind])) for kind in kinds}
        sq_means = {kind: float(np.mean([row["rmse"] for row in squared if row["kind"] == kind])) for kind in kinds}
        public_clean = float(np.mean([row["rmse"] for row in public if row["kind"] == "clean"]))
        attacks = {}
        for kind in kinds[1:]:
            growth = means[kind]-means["clean"]
            sq_growth = sq_means[kind]-sq_means["clean"]
            attenuation = 1-growth/sq_growth if sq_growth > 0 else None
            attacks[kind] = {"robust_growth": growth, "quadratic_growth": sq_growth,
                             "attenuation": attenuation,
                             "passes": attenuation is not None and attenuation >= .2}
        recall_loss = {kind: float(np.mean([row["event_recall"] for row in squared if row["kind"] == kind])
                                         -np.mean([row["event_recall"] for row in robust if row["kind"] == kind]))
                       for kind in kinds}
        ratio = means["clean"]/sq_means["clean"]
        numerical = all(row["solver_failure_rate"] == 0 for row in robust)
        citizen_value = means["clean"] < public_clean
        eligible = (ratio <= 1.05 and all(item["passes"] for item in attacks.values())
                    and max(recall_loss.values()) <= .05 and numerical and citizen_value)
        output.append({"id": identifier, "alpha": alpha, "clean_rmse": means["clean"],
                       "quadratic_clean_rmse": sq_means["clean"], "public_clean_rmse": public_clean,
                       "clean_ratio": ratio, "citizen_value_direction": citizen_value,
                       "attack": attacks, "event_recall_loss": recall_loss,
                       "numerical_pass": numerical, "eligible": bool(eligible)})
    return output


def run_partition(name, span, seeds, values, public, coordinates, threshold, base):
    start, end = span
    context_start = start-WARMUP
    truth = values[context_start:end]
    reference = public[context_start:end]
    extended = np.concatenate((reference, np.repeat(reference[-1:], DRAIN, axis=0)))
    score_slice = slice(WARMUP, len(truth))
    rows = []
    with (OUT/f"{name}.jsonl").open("w") as stream:
        for seed in seeds:
            channel_truth = np.maximum(np.where(np.isfinite(truth), truth, reference), 0.)
            for kind in ("clean", "adversarial_drift", "negative_bias"):
                records = synthetic_citizen_replay(channel_truth, seed=seed, kind=kind)
                records = [item for item in records if np.isfinite(truth[item.epoch, item.cell])]
                huber = estimate_public_field(extended, coordinates, records, base,
                                              innovation_scales=3.)
                quadratic = estimate_public_field(extended, coordinates, records,
                    replace(base, loss="quadratic", output_cap=False), innovation_scales=3.)
                target = truth[score_slice]
                observed = np.isfinite(target)
                event = observed & (target >= threshold)
                estimates = {"quadratic": (quadratic.reconstructed[:len(truth)][score_slice],
                                             quadratic.live[:len(truth)][score_slice]),
                             "public_only": (reference[score_slice], reference[score_slice])}
                for alpha in ALPHAS:
                    estimates[f"huber_d2_a{alpha:g}"] = (
                        reference[score_slice]+alpha*(huber.reconstructed[:len(truth)][score_slice]-reference[score_slice]),
                        reference[score_slice]+alpha*(huber.live[:len(truth)][score_slice]-reference[score_slice]))
                for identifier, (estimate, live) in estimates.items():
                    row = {"partition": name, "seed": seed, "kind": kind, "method": identifier,
                           "rmse": float(np.sqrt(np.mean((estimate[observed]-target[observed])**2))),
                           "event_recall": float(np.mean(live[event]>=threshold)) if event.any() else None,
                           "event_support": int(event.sum()),
                           "solver_failure_rate": (quadratic.diagnostics["solver_failure_rate"]
                                                   if identifier == "quadratic" else
                                                   huber.diagnostics["solver_failure_rate"] if identifier.startswith("huber") else 0.)}
                    rows.append(row); stream.write(json.dumps(row)+"\n"); stream.flush()
            print(json.dumps({"partition": name, "seed": seed, "rows": len(rows)}), flush=True)
    return rows


def main():
    for predecessor in (PREVIOUS, AR_PREVIOUS):
        result = json.loads((predecessor/"result.json").read_text())
        if result["selected"] is not None or result["validation_rows"] != 0:
            raise ValueError("expected closed predecessor null selection")
    protocol = json.loads(PROTOCOL.read_text())
    public_lock = json.loads((PUBLIC/"selection_lock.json").read_text())
    OUT.mkdir(exist_ok=False)
    base = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05, lambda_spatial=.02,
                           lag=6, time_step=1., numerical_refinements=2,
                           huber_delta=2., loss="huber", output_cap=True)
    registration = {
        "role": "post-delta and post-AR registered shrinkage development; test windows unread",
        "mechanism": "public plus alpha times bounded Huber correction",
        "alphas": ALPHAS, "base_config": vars(base), "partitions": PARTITIONS,
        "development_seeds": list(range(917400, 917408)),
        "validation_seeds": list(range(917500, 917512)),
        "selection_gates": {"clean_ratio": 1.05, "attack_attenuation": .2,
                            "event_recall_loss_each_condition": .05,
                            "citizen_clean_rmse_below_public": True,
                            "solver_failure_rate": 0.},
        "selection": "eligible only; lowest clean RMSE, then larger alpha",
        "hashes": {str(path): digest(path) for path in
                   (Path(__file__), ROOT/"airproof/v6_estimator.py",
                    PUBLIC/"selection_lock.json", PUBLIC/"models.pkl",
                    PREVIOUS/"result.json", AR_PREVIOUS/"result.json", PROTOCOL, DATA)},
        "test_windows_read": False, "confirmation_authorized": False,
    }
    # vars() is unavailable for a slots/frozen dataclass on some Python versions.
    from dataclasses import asdict
    registration["base_config"] = asdict(base)
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
    coordinates = archive["coordinates"]
    coordinates = (coordinates-coordinates.mean(0))*np.array(
        [111.2, 111.2*np.cos(np.deg2rad(coordinates[:, 0].mean()))])
    started = time.perf_counter()
    development_rows = run_partition("development", PARTITIONS["development"],
        registration["development_seeds"], values, public, coordinates,
        public_lock["event_threshold"], base)
    development = summarize(development_rows)
    eligible = [row for row in development if row["eligible"]]
    selected = min(eligible, key=lambda row: (row["clean_rmse"], -row["alpha"]))["id"] if eligible else None
    lock = {"selected": selected, "development": development,
            "validation_metrics_read_for_selection": False,
            "test_metrics_read_for_selection": False, "confirmation_authorized": False}
    (OUT/"selection_lock.json").write_text(json.dumps(lock, indent=2))
    validation_rows = []
    validation = []
    if selected is not None:
        validation_rows = run_partition("validation", PARTITIONS["validation"],
            registration["validation_seeds"], values, public, coordinates,
            public_lock["event_threshold"], base)
        validation = [row for row in summarize(validation_rows) if row["id"] == selected]
    passes = bool(validation and validation[0]["eligible"])
    result = {"selected": selected, "development": development, "validation": validation,
              "development_rows": len(development_rows), "validation_rows": len(validation_rows),
              "validation_passes": passes, "elapsed_seconds": time.perf_counter()-started,
              "confirmation_authorized": passes}
    (OUT/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

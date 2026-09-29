"""Single registered adaptive-consensus repair on pre-test EPA partitions."""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from dataclasses import asdict, replace
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
    PUBLIC, DRAIN, WARMUP, public_prediction, summarize)


OUT = ROOT/"reports/v6/epa_2025_adaptive_consensus"
PREDECESSORS = [ROOT/"reports/v6/epa_2025_estimator",
                ROOT/"reports/v6/epa_2025_residual_ar",
                ROOT/"reports/v6/epa_2025_shrinkage"]
CANDIDATE = {"id": "adaptive_consensus_s1.5", "spread_threshold": 1.5}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_partition(name, span, seeds, values, public, coordinates, threshold, robust_config):
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
                robust = estimate_public_field(extended, coordinates, records, robust_config,
                                               innovation_scales=3.)
                quadratic = estimate_public_field(extended, coordinates, records,
                    replace(robust_config, loss="quadratic", output_cap=False, input_clip=False),
                    innovation_scales=3.)
                target = truth[score_slice]
                observed = np.isfinite(target)
                event = observed & (target >= threshold)
                estimates = {CANDIDATE["id"]: (robust.reconstructed[:len(truth)][score_slice],
                                                robust.live[:len(truth)][score_slice], robust),
                             "quadratic": (quadratic.reconstructed[:len(truth)][score_slice],
                                           quadratic.live[:len(truth)][score_slice], quadratic),
                             "public_only": (reference[score_slice], reference[score_slice], None)}
                for identifier, (estimate, live, result) in estimates.items():
                    row = {"partition": name, "seed": seed, "kind": kind,
                           "method": identifier,
                           "rmse": float(np.sqrt(np.mean((estimate[observed]-target[observed])**2))),
                           "event_recall": float(np.mean(live[event]>=threshold)) if event.any() else None,
                           "event_support": int(event.sum()),
                           "solver_failure_rate": 0 if result is None else result.diagnostics["solver_failure_rate"]}
                    rows.append(row); stream.write(json.dumps(row)+"\n"); stream.flush()
            print(json.dumps({"partition": name, "seed": seed, "rows": len(rows)}), flush=True)
    return rows


def add_citizen_gate(summary_row, rows):
    public_clean = float(np.mean([row["rmse"] for row in rows
                                  if row["method"] == "public_only" and row["kind"] == "clean"]))
    summary_row["public_clean_rmse"] = public_clean
    summary_row["citizen_value_direction"] = summary_row["clean_rmse"] < public_clean
    summary_row["eligible"] = bool(summary_row["eligible"] and summary_row["citizen_value_direction"])
    return summary_row


def main():
    for predecessor in PREDECESSORS:
        result = json.loads((predecessor/"result.json").read_text())
        if result["selected"] is not None or result["validation_rows"] != 0:
            raise ValueError("expected closed predecessor null selection")
    protocol = json.loads(PROTOCOL.read_text())
    public_lock = json.loads((PUBLIC/"selection_lock.json").read_text())
    OUT.mkdir(exist_ok=False)
    robust_config = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05,
        lambda_spatial=.02, lag=6, time_step=1., numerical_refinements=2,
        huber_delta=2., loss="consensus_huber", output_cap=True, input_clip=True,
        clip_delta=4., consensus_spread_threshold=1.5,
        consensus_singleton_quadratic=True)
    registration = {
        "role": "single post-diagnosis adaptive-consensus development; EPA tests unread",
        "candidate": CANDIDATE, "config": asdict(robust_config), "partitions": PARTITIONS,
        "mechanism": "tight same-cell/acquisition groups and clipped singletons use quadratic score; dispersed groups use median plus Huber; correction cap8",
        "development_seeds": list(range(917600, 917608)),
        "validation_seeds": list(range(917700, 917712)),
        "selection_gates": {"clean_ratio": 1.05, "attack_attenuation": .2,
                            "event_recall_loss_each_condition": .05,
                            "citizen_clean_rmse_below_public": True,
                            "solver_failure_rate": 0.},
        "coordinated_inlier": "mandatory post-selection stress; no robustness theorem for a tight corrupted majority",
        "hashes": {str(path): digest(path) for path in
                   (Path(__file__), ROOT/"airproof/v6_estimator.py",
                    PUBLIC/"selection_lock.json", PUBLIC/"models.pkl", PROTOCOL, DATA,
                    *(item/"result.json" for item in PREDECESSORS))},
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
    coordinates = archive["coordinates"]
    coordinates = (coordinates-coordinates.mean(0))*np.array(
        [111.2, 111.2*np.cos(np.deg2rad(coordinates[:, 0].mean()))])
    started = time.perf_counter()
    development_rows = run_partition("development", PARTITIONS["development"],
        registration["development_seeds"], values, public, coordinates,
        public_lock["event_threshold"], robust_config)
    development = [add_citizen_gate(row, development_rows)
                   for row in summarize(development_rows, [CANDIDATE])]
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
            public_lock["event_threshold"], robust_config)
        validation = [add_citizen_gate(row, validation_rows)
                      for row in summarize(validation_rows, [CANDIDATE])]
    passes = bool(validation and validation[0]["eligible"])
    result = {"selected": selected, "development": development,
              "validation": validation, "development_rows": len(development_rows),
              "validation_rows": len(validation_rows), "validation_passes": passes,
              "elapsed_seconds": time.perf_counter()-started,
              "confirmation_authorized": passes}
    (OUT/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

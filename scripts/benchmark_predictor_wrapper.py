"""Fixed production wrapper over frozen ESN--TNN predictions, synthetic citizens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airproof.predictor_wrapper import synthetic_citizen_replay, wrap_public_predictions


def interval(candidate_errors, baseline_errors, *, block=12, draws=4000):
    # Error arrays are noise-world x time, averaged over the full station vector.
    rng = np.random.default_rng(20260903)
    worlds, epochs = candidate_errors.shape
    samples = []
    for _ in range(draws):
        selected_worlds = rng.integers(worlds, size=worlds)
        starts = rng.integers(epochs, size=int(np.ceil(epochs / block)))
        times = ((starts[:, None] + np.arange(block)) % epochs).ravel()[:epochs]
        a = candidate_errors[np.ix_(selected_worlds, times)].mean()
        b = baseline_errors[np.ix_(selected_worlds, times)].mean()
        samples.append(np.sqrt(a / b))
    return {"ratio": float(np.sqrt(candidate_errors.mean() / baseline_errors.mean())),
            "lower95": float(np.quantile(samples, .025)), "upper95": float(np.quantile(samples, .975)),
            "time_block_epochs": block, "draws": draws,
            "scope": "joint noise-realization and circular temporal-block bootstrap; stations kept together; one public archive, not independent cities"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, range(7850, 7862))))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not (args.predictor_dir / "summary.json").exists():
        raise ValueError("completed locked predictor benchmark required")
    path = args.predictor_dir / "heldout_predictions.npz"
    archive = np.load(path)
    truth, backbone = archive["truth"], archive["ESN_TNN"]
    lat_lon = archive["coordinates_lat_lon"]
    xy = (lat_lon - lat_lon.mean(axis=0)) * np.array([111.2, 111.2 * np.cos(np.deg2rad(lat_lon[:, 0].mean()))])
    seeds = list(map(int, args.seeds.split(",")))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique noise seeds required")
    kinds = ("clean", "adversarial_drift", "negative_bias")
    configs = {"AirProof_Ptheta": {"delta": 4., "correction_clip": 8.},
               "unbounded_graph_Ptheta": {"delta": 1e6, "correction_clip": 1e6}}
    manifest = {
        "role": "fixed-estimator-interface-test-on-public-real-field-with-synthetic-citizen-channel",
        "predictor_prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "module_sha256": {name: hashlib.sha256((ROOT / "airproof" / name).read_bytes()).hexdigest()
                          for name in ("predictor_wrapper.py", "twin.py", "meteorology.py")},
        "seeds": seeds, "attack_kinds": kinds, "wrapper_configs": configs,
        "configuration_selection": "none on this test; bounded delta/cap/lag/regularization fixed from primary v4 implementation",
        "shared_wrapper_parameters": {"fixed_lag": 6, "lambda_temporal": .5, "lambda_spatial": .2},
        "citizen_channel": {"reports_per_station": 3, "sigma": 3., "availability": .6,
                            "attack_fraction": .2, "attack_amplitude": 12., "delay_epochs": [0, 1, 2, 8],
                            "delay_probabilities": [.7, .2, .08, .02]},
        "comparison_scope": "Ptheta uses public information through t-1; both graph wrappers additionally use exactly the same synthetic citizen records that arrived by t; reconstruction accuracy retention, NOT equal-information forecasting superiority",
        "geometry": "44 real stations, local kilometre coordinates, symmetric 4-neighbor graph; 5 isolated padded solver vertices",
        "meteorology": "unavailable in this archive; identity transition is explicit for this interface-only benchmark",
        "independence": "noise/participation/delay replications share one public field and predictor; do not count as independent physical worlds",
        "created_unix": time.time(), "total_jobs": len(seeds) * len(kinds) * len(configs),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, errors = [], {}
    started = time.perf_counter()
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for kind in kinds:
            for method in configs:
                errors[kind, method] = []
            for seed in seeds:
                observations = synthetic_citizen_replay(truth, seed=seed, kind=kind)
                for method, configuration in configs.items():
                    prediction, diagnostics = wrap_public_predictions(backbone, xy, observations, **configuration)
                    squared = np.mean((prediction - truth)**2, axis=1)
                    errors[kind, method].append(squared)
                    row = {"seed": seed, "kind": kind, "method": method,
                           "rmse": float(np.sqrt(squared.mean())), "diagnostics": diagnostics}
                    rows.append(row)
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
            print(json.dumps({"event": "kind_complete", "kind": kind, "rows": len(rows),
                              "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    baseline_errors = np.tile(np.mean((backbone - truth)**2, axis=1), (len(seeds), 1))
    contrasts = {}
    for kind in kinds:
        bounded = np.asarray(errors[kind, "AirProof_Ptheta"])
        unbounded = np.asarray(errors[kind, "unbounded_graph_Ptheta"])
        contrasts[kind] = {
            "wrapper_rmse": float(np.sqrt(bounded.mean())),
            "unbounded_graph_rmse": float(np.sqrt(unbounded.mean())),
            "versus_backbone": [interval(bounded, baseline_errors, block=block) for block in (6, 12, 24)],
            "versus_same_information_unbounded": interval(bounded, unbounded),
        }
    summary = {"role": manifest["role"], "jobs": len(rows), "contrasts": contrasts,
               "backbone_rmse": float(np.sqrt(baseline_errors.mean())),
               "max_solver_failure_rate": max(row["diagnostics"]["solver_failure_rate"] for row in rows),
               "maximum_bounded_correction": max(row["diagnostics"]["maximum_correction"] for row in rows if row["method"] == "AirProof_Ptheta"),
               "elapsed_seconds": time.perf_counter() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "squared_errors.npz", backbone=baseline_errors,
                        **{f"{kind}__{method}": np.asarray(values) for (kind, method), values in errors.items()})
    print(json.dumps({"event": "complete", "jobs": len(rows), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

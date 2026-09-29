"""Diagnostic only: frozen, previously viewed PurpleAir archive; no retraining."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
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
from airproof.predictor_wrapper import synthetic_citizen_replay
from airproof.v5_estimator import EstimatorConfig, enumerate_candidates, estimate_public_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT/"reports/v4_validation/esntnn_purpleair_7800/heldout_predictions.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="916000,916001")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    archive = np.load(args.archive)
    truth, public = archive["truth"], np.maximum(archive["ESN_TNN"], 0.)
    xy = archive["coordinates_lat_lon"]
    xy = (xy-xy.mean(0))*np.array([111.2, 111.2*np.cos(np.deg2rad(xy[:, 0].mean()))])
    candidates = enumerate_candidates()
    controls = {f"SQ_{objective}_reg{reg:g}": EstimatorConfig(objective=objective, regularization_multiplier=reg, input_clip=False, output_cap=False)
                for objective in ("full_field", "residual") for reg in (.1, .3, 1.)}
    base = EstimatorConfig()
    controls.update({"HUBER": replace(base, input_clip=False, output_cap=False, loss="huber"),
                     "INPUT_CLIP_ONLY": replace(base, output_cap=False),
                     "OUTPUT_CAP_ONLY": replace(base, input_clip=False)})
    configs = {c.candidate_id: c for c in candidates} | controls
    seeds = [int(v) for v in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("unique seeds required")
    manifest = {"role": "diagnostic previously viewed archive; not fresh pollution holdout or selection/confirmation",
                "seeds": seeds, "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                "candidate_count": 12, "configuration_selection": "none; all retained",
                "calibration_or_retraining": False, "transitions": "identity; archive lacks meteorology",
                "configs": {k: asdict(v) for k, v in configs.items()},
                "public_preprocessing": "nonnegative projection", "source_sha256": hashlib.sha256((ROOT/"airproof/v5_estimator.py").read_bytes()).hexdigest()}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    rows = []
    with (args.output_dir/"results.jsonl").open("w") as out:
        for seed in seeds:
            for kind in ("clean", "adversarial_drift", "negative_bias"):
                observations = synthetic_citizen_replay(truth, seed=seed, kind=kind)
                for method, cfg in {"PUBLIC": None, **configs}.items():
                    tick = time.perf_counter()
                    if cfg is None:
                        live = reconstructed = public
                        diagnostic = {}
                    else:
                        result = estimate_public_field(public, xy, observations, cfg)
                        live, reconstructed = result.live, result.reconstructed
                        diagnostic = {k: v for k, v in result.diagnostics.items() if k != "epoch_terms"}
                    row = {"seed": seed, "kind": kind, "method": method,
                           "live_rmse": float(np.sqrt(np.mean((live-truth)**2))),
                           "reconstructed_rmse": float(np.sqrt(np.mean((reconstructed-truth)**2))),
                           "live_event_recall_35": float(np.sum((live >= 35)&(truth >= 35))/max(np.sum(truth >= 35), 1)),
                           "runtime_seconds": time.perf_counter()-tick, "diagnostics": diagnostic}
                    rows.append(row)
                    out.write(json.dumps(row, allow_nan=False)+"\n"); out.flush()
                print(json.dumps({"seed": seed, "kind": kind, "rows": len(rows)}), flush=True)
    summary = {"role": manifest["role"], "rows": len(rows), "runtime_seconds": time.perf_counter()-started,
               "max_solver_failure_rate": max(r["diagnostics"].get("solver_failure_rate", 0) for r in rows),
               "mean_live_rmse": {kind: {method: float(np.mean([r["live_rmse"] for r in rows if r["kind"] == kind and r["method"] == method]))
                                         for method in ("PUBLIC", *configs)} for kind in ("clean", "adversarial_drift", "negative_bias")}}
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

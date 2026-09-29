"""Bounded synthetic release-only diagnostic with separate frozen calibration."""
from __future__ import annotations
import argparse
from dataclasses import replace
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
from airproof.v5_release_twin import PublicReleaseObservation, ReleaseOnlyTwin, fit_release_calibration
from airproof.v5_uncertainty import fit_interval_calibrator, predict_intervals, interval_metrics


def synthetic(seed, epochs=672, groups=4):
    rng = np.random.default_rng(seed)
    time_axis = np.arange(epochs)[:, None]
    public = 22+4*np.sin(time_axis/24+np.arange(groups)[None, :])
    correction = np.zeros((epochs, groups))
    for t in range(1, epochs):
        correction[t] = .96*correction[t-1]+rng.normal(0, .8, groups)+rng.normal(0, .3)
    truth = np.maximum(public+correction, 0)
    queries = np.zeros_like(truth)
    for t in range(epochs):
        for g in range(groups):
            count = rng.poisson(17+8*(1+np.sin(t/72+g)))
            # Calibration generator is synthetic; count never enters online twin.
            samples = truth[t, g]-public[t, g]+rng.normal(1.5, 3, count)
            queries[t, g] = public[t, g]+np.clip(samples, -10, 10).sum()/max(count, 20)
    releases = []
    # User-history replacement may move one user's contribution between groups.
    # Preserve continual_release.py's vector L1 bound 4C/k, not fixed-group 2C/k.
    scale = 4*10*28/(20*8)
    for acquisition in range(0, epochs, 24):
        for group in range(groups):
            # Public outage calendar, independent of private contribution count.
            suppressed = (acquisition//24+group)%9 == 0
            value = None if suppressed else float(queries[acquisition, group]+rng.laplace(0, scale))
            releases.append(PublicReleaseObservation(f"{seed}-{acquisition}-{group}", acquisition,
                                                     acquisition+24, group, value, scale))
    return truth, public, queries, releases


def evaluate(calibration, arrays):
    truth, public, _, releases = arrays
    groups = public.shape[1]
    covariance = calibration.initial_covariance.copy()
    process = calibration.process_covariance.copy()
    # Naive control assumes query bias is zero and lacks correlated bias learning.
    covariance[groups:, :] = 0; covariance[:, groups:] = 0
    process[groups:, :] = 0; process[:, groups:] = 0
    covariance[groups:, groups:] = np.eye(groups)*1e-8
    process[groups:, groups:] = np.eye(groups)*1e-8
    mean = calibration.mean.copy(); mean[groups:] = 0
    transition = calibration.transition.copy(); transition[groups:, :] = 0
    naive = replace(calibration, mean=mean, transition=transition,
                    initial_covariance=covariance, process_covariance=process)
    models = {"NAIVE": ReleaseOnlyTwin(naive), "CALIBRATED": ReleaseOnlyTwin(calibration)}
    direct_live, direct_reconstructed = public.copy(), public.copy()
    last_correction = np.zeros(groups)
    for t in range(len(public)+24):
        incoming = [r for r in releases if r.publication_epoch == t]
        baseline = public[min(t, len(public)-1)]
        for twin in models.values():
            twin.update(t, baseline, incoming)
        for r in incoming:
            if r.value is not None:
                last_correction[r.group] = r.value-public[r.acquisition_epoch, r.group]
                direct_reconstructed[r.acquisition_epoch, r.group] = max(r.value, 0.)
        if t < len(public):
            direct_live[t] = np.maximum(public[t]+last_correction, 0)
    results = {"PUBLIC": (public, public, {}),
               "DIRECT": (direct_live, direct_reconstructed, {"rule": "latest published residual carried forward; reconstruction only at released acquisitions"})}
    for name, twin in models.items():
        results[name] = (np.array([twin.live[t] for t in range(len(public))]),
                         np.array([twin.reconstructed[t] for t in range(len(public))]), twin.diagnostics())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-worlds", type=int, default=8)
    parser.add_argument("--heldout-worlds", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration_seeds = list(range(912000, 912000+args.calibration_worlds))
    heldout_seeds = list(range(915000, 915000+args.heldout_worlds))
    manifest = {"role": "synthetic component diagnostic; not integrated confirmation",
                "calibration_seeds": calibration_seeds, "heldout_seeds": heldout_seeds,
                "groups": 4, "acquisition_epochs": 672, "query_epochs": list(range(0, 672, 24)),
                "publication_delay": 24, "drain_epochs": 24, "drain_public_baseline": "carry final acquisition baseline",
                "suppression": "public exogenous outage calendar; no private count gate",
                "clip": 10, "stabilization_denominator_min": 20, "epsilon_history": 8,
                "queries_per_group": 28, "laplace_scale": 7.0, "lag": 48,
                "sensitivity": "vector user-history replacement including group move: 4C/k",
                "uncertainty": "in-sample calibration-world residual radii; heldout evaluation only; descriptive coverage"}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    worlds = [synthetic(seed) for seed in calibration_seeds]
    calibration = fit_release_calibration(*[np.stack([w[index] for w in worlds]) for index in range(3)])
    np.savez_compressed(args.output_dir/"frozen_calibration.npz", mean=calibration.mean,
                        transition=calibration.transition, process_covariance=calibration.process_covariance,
                        initial_covariance=calibration.initial_covariance)
    calibration_results = [evaluate(calibration, w) for w in worlds]
    radii = {(method, clock): fit_interval_calibrator(np.stack([r[method][clock] for r in calibration_results]),
                                                     np.stack([w[0] for w in worlds]))
             for method in ("PUBLIC", "DIRECT", "NAIVE", "CALIBRATED") for clock in (0, 1)}
    rows = []
    for seed in heldout_seeds:
        world = synthetic(seed)
        estimates = evaluate(calibration, world)
        for method, (live, reconstructed, diagnostics) in estimates.items():
            for clock, prediction in enumerate((live, reconstructed)):
                lower, upper = predict_intervals(radii[method, clock], prediction)
                # All epochs and released acquisition mask answer different questions.
                mask = np.zeros(world[0].shape, bool)
                for r in world[3]:
                    if r.value is not None:
                        mask[r.acquisition_epoch, r.group] = True
                row = {"seed": seed, "method": method, "clock": ("live", "reconstructed")[clock],
                       "rmse_all": float(np.sqrt(np.mean((prediction-world[0])**2))),
                       "rmse_release_mask": float(np.sqrt(np.mean((prediction[mask]-world[0][mask])**2))),
                       "intervals": interval_metrics(world[0], lower, upper), "diagnostics": diagnostics}
                rows.append(row)
        print(json.dumps({"heldout_seed": seed, "rows": len(rows)}), flush=True)
    (args.output_dir/"results.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    summary = {"role": manifest["role"], "rows": len(rows), "runtime_seconds": time.perf_counter()-started,
               "mean_rmse": {clock: {method: float(np.mean([r["rmse_all"] for r in rows if r["clock"] == clock and r["method"] == method]))
                                      for method in ("PUBLIC", "DIRECT", "NAIVE", "CALIBRATED")} for clock in ("live", "reconstructed")}}
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

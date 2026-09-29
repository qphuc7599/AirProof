"""Delayed release-only assimilation of integrated retained protected outputs."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airproof.v5_release_twin import ReleaseCalibration, ReleaseOnlyTwin, PublicReleaseObservation, fit_release_calibration
from scripts.analyze_v5_uncertainty import world_bootstrap_indices


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(path):
    with np.load(path, allow_pickle=False) as data:
        mean = data["mean"]
        return ReleaseCalibration(mean, data["transition"], data["process_covariance"],
                                  data["initial_covariance"], len(mean)//2)


def protected_inputs(path, metadata):
    """Do not access truth, raw estimates, noiseless queries, or private counts."""
    with np.load(path, allow_pickle=False) as archive:
        baseline = np.asarray(archive["baseline"], float)
        values = np.asarray(archive["residual"], float)
        mask = np.asarray(archive["residual_mask"], bool)
    if baseline.shape != values.shape or mask.shape != baseline.shape or not np.isfinite(baseline).all():
        raise ValueError("invalid integrated protected release dimensions")
    scale = np.sqrt(float(metadata["residual"]["laplace_variance"])/2)
    if not np.isclose(scale, 7.):
        raise ValueError("registered replacement-history residual scale7 required")
    deadline = metadata["deadline"]
    if deadline != 24:
        raise ValueError("registered publication deadline24 required")
    schedule = metadata["scheduled_epochs"]
    if len(schedule) != 28 or len(set(schedule)) != 28:
        raise ValueError("registered28 unique acquisition epochs required")
    releases = []
    for acquisition in schedule:
        if not 0 <= acquisition < len(baseline):
            raise ValueError("scheduled acquisition outside public horizon")
        for group in range(baseline.shape[1]):
            value = float(values[acquisition, group]) if mask[acquisition, group] else None
            if value is not None and not np.isfinite(value):
                raise ValueError("released value must be finite")
            releases.append(PublicReleaseObservation(f"{acquisition}-{group}", acquisition,
                                                     acquisition+deadline, group, value, scale))
    return baseline, releases


def predict_release_only(baseline, releases, calibration):
    """Online interface contains only public baseline and protected releases."""
    g = calibration.groups
    if baseline.shape[1] != g:
        raise ValueError("calibration group count mismatch")
    covariance, process = calibration.initial_covariance.copy(), calibration.process_covariance.copy()
    for matrix in (covariance, process):
        matrix[g:, :] = 0; matrix[:, g:] = 0; matrix[g:, g:] = np.eye(g)*1e-8
    mean = calibration.mean.copy(); mean[g:] = 0
    transition = calibration.transition.copy(); transition[g:, :] = 0
    naive = replace(calibration, mean=mean, transition=transition, initial_covariance=covariance, process_covariance=process)
    twins = {"CALIBRATED": ReleaseOnlyTwin(calibration), "NAIVE": ReleaseOnlyTwin(naive),
             "PUBLIC_CALIBRATED": ReleaseOnlyTwin(calibration)}
    arrivals = {}
    for item in releases:
        arrivals.setdefault(item.publication_epoch, []).append(item)
    direct_live = baseline.copy(); direct_reconstructed = baseline.copy()
    correction = np.zeros(g)
    horizon = max(len(baseline), max(arrivals, default=-1)+1)
    for epoch in range(horizon):
        incoming = arrivals.get(epoch, [])
        public = baseline[min(epoch, len(baseline)-1)]
        for name, twin in twins.items():
            twin.update(epoch, public, () if name == "PUBLIC_CALIBRATED" else incoming)
        for item in incoming:
            if item.value is not None:
                correction[item.group] = item.value-baseline[item.acquisition_epoch, item.group]
                direct_reconstructed[item.acquisition_epoch, item.group] = max(item.value, 0)
        if epoch < len(baseline):
            direct_live[epoch] = np.maximum(public+correction, 0)
    predictions = {"PUBLIC": (baseline.copy(), baseline.copy()), "DIRECT": (direct_live, direct_reconstructed)}
    for name, twin in twins.items():
        predictions[name] = tuple(np.array([getattr(twin, clock)[t] for t in range(len(baseline))]) for clock in ("live", "reconstructed"))
    return predictions, {name: twin.diagnostics() for name, twin in twins.items()}


def freeze_dense(campaign, output):
    manifest = json.loads((campaign/"manifest.json").read_text())
    if manifest["stage"] != "calibration" or len(manifest["seeds"]) != 8 or len(set(manifest["seeds"])) != 8:
        raise ValueError("eight unique separate calibration worlds required")
    truth, public, query, inputs = [], [], [], {}
    for seed in manifest["seeds"]:
        for cell in manifest["cells"]:
            path = campaign/"jobs"/str(seed)/cell/"release.npz"
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata.get("dense_query_calibration_only") is not True:
                raise ValueError("explicit dense hourly calibration-query provenance required")
            with np.load(path, allow_pickle=False) as data:
                y, p, q = data["truth"], data["baseline"], data["residual_query_dense"]
                if q.shape != y.shape or not np.isfinite(q).all():
                    raise ValueError("dense finite hourly queries required; sparse24h queries forbidden")
                truth.append(y); public.append(p); query.append(q)
            inputs[str(path.resolve())] = sha(path)
    calibration = fit_release_calibration(np.stack(truth), np.stack(public), np.stack(query))
    output.mkdir(parents=True, exist_ok=False)
    frozen = output/"frozen_calibration.npz"
    np.savez_compressed(frozen, mean=calibration.mean, transition=calibration.transition,
                        process_covariance=calibration.process_covariance, initial_covariance=calibration.initial_covariance)
    (output/"manifest.json").write_text(json.dumps({"role": "integrated dense hourly public-synthetic calibration",
        "calibration_seeds": manifest["seeds"], "campaign_manifest_sha256": sha(campaign/"manifest.json"),
        "frozen_calibration_sha256": sha(frozen), "input_sha256": inputs,
        "source_hash": manifest["source_hash"], "selected": manifest["selected"],
        "hourly_AR": True, "world_boundaries_joined": False}, indent=2))


def analyze(campaign, calibration_path, calibration_manifest, output, *, transfer=False, burn=48, draws=4000):
    manifest = json.loads((campaign/"manifest.json").read_text())
    fitted = json.loads(calibration_manifest.read_text())
    if set(manifest["seeds"])&set(fitted["calibration_seeds"]):
        raise ValueError("evaluation seeds overlap calibration")
    if not transfer and (manifest["source_hash"] != fitted["source_hash"] or manifest["selected"] != fitted["selected"]):
        raise ValueError("integrated calibration identity differs; explicit transfer declaration required")
    expected = fitted.get("frozen_calibration_sha256")
    if expected is not None and sha(calibration_path) != expected:
        raise ValueError("frozen calibration hash changed")
    calibration = load_frozen(calibration_path)
    rows, inputs, diagnostics = [], {}, []
    for seed in manifest["seeds"]:
        for cell in manifest["cells"]:
            path = campaign/"jobs"/str(seed)/cell/"release.npz"
            metadata = json.loads(path.with_suffix(".json").read_text())
            baseline, releases = protected_inputs(path, metadata)
            predictions, diagnostic = predict_release_only(baseline, releases, calibration)
            # Truth is opened only after every prediction has been constructed.
            with np.load(path, allow_pickle=False) as data:
                truth = np.asarray(data["truth"], float)
            if truth.shape != baseline.shape or not np.isfinite(truth).all():
                raise ValueError("finite matched evaluation truth required")
            inputs[str(path.resolve())] = sha(path)
            diagnostics.append({"seed": seed, "cell": cell, **diagnostic})
            live_mask = np.zeros_like(truth, bool); live_mask[burn:] = True
            released_mask = np.zeros_like(truth, bool)
            for release in releases:
                if release.value is not None and release.acquisition_epoch >= burn:
                    released_mask[release.acquisition_epoch, release.group] = True
            for method, estimates in predictions.items():
                for clock, estimate in zip(("live", "reconstructed"), estimates):
                    for support, mask in (("all", live_mask), ("released_acquisitions", released_mask)):
                        if mask.any():
                            rows.append({"seed": seed, "cell": cell, "method": method, "clock": clock,
                                "support": support, "n": int(mask.sum()), "mse": float(np.mean((estimate[mask]-truth[mask])**2))})
    indices = world_bootstrap_indices(len(manifest["seeds"]), draws)
    summary = []
    for key in sorted({(r["cell"], r["method"], r["clock"], r["support"]) for r in rows}):
        subset = [r for r in rows if (r["cell"], r["method"], r["clock"], r["support"]) == key]
        values = {r["seed"]: r for r in subset}
        n = np.array([values.get(s, {}).get("n", 0) for s in manifest["seeds"]])
        total = np.array([values.get(s, {}).get("mse", 0)*values.get(s, {}).get("n", 0) for s in manifest["seeds"]])
        den = n[indices].sum(1); num = total[indices].sum(1)
        samples = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))
        bounds = np.nanquantile(samples, (.025, .975))
        summary.append({"cell": key[0], "method": key[1], "clock": key[2], "support": key[3],
                        "rmse": float(np.sqrt(total.sum()/n.sum())), "lower95": float(bounds[0]), "upper95": float(bounds[1])})
    output.mkdir(parents=True, exist_ok=False)
    (output/"summary.json").write_text(json.dumps({"role": "integrated delayed release-only evaluation",
        "calibration_transfer": transfer, "calibration_sha256": sha(calibration_path),
        "calibration_manifest_sha256": sha(calibration_manifest), "campaign_manifest_sha256": sha(campaign/"manifest.json"),
        "input_sha256": inputs, "drain_baseline": "carry final public baseline until final publication",
        "laplace_scale": 7., "variance_added_once": True, "bootstrap_unit": "whole shared world",
        "results": summary, "diagnostics": diagnostics}, indent=2, allow_nan=False))
    (output/"world_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze-dense", "analyze"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--transfer-calibration", action="store_true")
    args = parser.parse_args()
    if args.mode == "freeze-dense":
        freeze_dense(args.campaign, args.output_dir)
    else:
        if args.calibration is None or args.calibration_manifest is None:
            parser.error("analyze requires frozen calibration and its manifest")
        analyze(args.campaign, args.calibration, args.calibration_manifest, args.output_dir, transfer=args.transfer_calibration)


if __name__ == "__main__":
    main()

"""Registered consensus-Huber development and validation on exposed archive data."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import rdata

from airproof.predictor_wrapper import synthetic_citizen_replay
from airproof.v6_estimator import EstimatorConfig, estimate_public_field


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(rows, candidates):
    by_id = []
    for candidate in candidates:
        identifier = candidate["id"]
        selected_rows = [row for row in rows if row["method"] == identifier]
        quadratic_rows = [row for row in rows if row["method"] == "quadratic"]
        means = {kind: float(np.mean([row["rmse"] for row in selected_rows if row["kind"] == kind]))
                 for kind in ("clean", "adversarial_drift", "negative_bias")}
        quadratic = {kind: float(np.mean([row["rmse"] for row in quadratic_rows if row["kind"] == kind]))
                     for kind in ("clean", "adversarial_drift", "negative_bias")}
        attacks = {}
        for kind in ("adversarial_drift", "negative_bias"):
            robust_growth = means[kind] - means["clean"]
            quadratic_growth = quadratic[kind] - quadratic["clean"]
            attenuation = 1-robust_growth/quadratic_growth if quadratic_growth > 0 else None
            attacks[kind] = {"robust_growth": robust_growth, "quadratic_growth": quadratic_growth,
                             "attenuation": attenuation,
                             "passes": attenuation is not None and attenuation >= .2}
        recall = float(np.mean([row["event_recall"] for row in selected_rows if row["kind"] == "clean"]))
        quadratic_recall = float(np.mean([row["event_recall"] for row in quadratic_rows if row["kind"] == "clean"]))
        ratio = means["clean"] / quadratic["clean"]
        eligible = (ratio <= 1.05 and recall >= quadratic_recall-.05
                    and all(item["passes"] for item in attacks.values()))
        by_id.append({"id": identifier, "clean_rmse": means["clean"],
                      "quadratic_clean_rmse": quadratic["clean"], "clean_ratio": ratio,
                      "event_recall": recall, "quadratic_event_recall": quadratic_recall,
                      "attacks": attacks, "eligible": bool(eligible)})
    return by_id


def _run_partition(public, truth, coordinates, threshold, seeds, candidates, base_config,
                   output, role, score_steps):
    rows = []
    with output.open("w") as stream:
        for seed in seeds:
            for kind in ("clean", "adversarial_drift", "negative_bias"):
                records = synthetic_citizen_replay(truth, seed=seed, kind=kind)
                for candidate in candidates:
                    config = replace(base_config, loss="consensus_huber",
                                     consensus_spread_threshold=candidate["spread_threshold"])
                    result = estimate_public_field(public, coordinates, records, config,
                                                   innovation_scales=3.)
                    event = truth[:score_steps] >= threshold
                    row = {"partition": role, "seed": seed, "kind": kind,
                           "method": candidate["id"],
                           "rmse": float(np.sqrt(np.mean((result.reconstructed[:score_steps]-truth[:score_steps])**2))),
                           "event_recall": float(np.mean(result.live[:score_steps][event]>=threshold)),
                           "event_support": int(event.sum()),
                           "solver_failure_rate": result.diagnostics["solver_failure_rate"]}
                    rows.append(row); stream.write(json.dumps(row)+"\n"); stream.flush()
                quadratic = estimate_public_field(public, coordinates, records,
                    replace(base_config, loss="quadratic", output_cap=False), innovation_scales=3.)
                event = truth[:score_steps] >= threshold
                row = {"partition": role, "seed": seed, "kind": kind, "method": "quadratic",
                       "rmse": float(np.sqrt(np.mean((quadratic.reconstructed[:score_steps]-truth[:score_steps])**2))),
                       "event_recall": float(np.mean(quadratic.live[:score_steps][event]>=threshold)),
                       "event_support": int(event.sum()),
                       "solver_failure_rate": quadratic.diagnostics["solver_failure_rate"]}
                rows.append(row); stream.write(json.dumps(row)+"\n"); stream.flush()
            print(json.dumps({"partition": role, "seed": seed, "rows": len(rows)}), flush=True)
    return rows


def main():
    out = Path("reports/v6/consensus_estimator_development")
    out.mkdir(exist_ok=False)
    data_path = Path("data/external/baselines/esntnn_author/Data/APFour.RData")
    original_path = Path("reports/v4_validation/esntnn_purpleair_7800/heldout_predictions.npz")
    center_lock = Path("reports/v6/public_dynamics_development/selection_lock.json")
    center_id = json.loads(center_lock.read_text())["selected"]
    center_path = center_lock.parent / f"{center_id}_test.npz"
    calibration_path = Path("reports/v6/backbone_validation_recovery/public_validation_predictions.npz")
    candidates = [{"id": f"consensus_s{spread:g}", "spread_threshold": spread}
                  for spread in (1., 1.5, 2.)]
    base_config = EstimatorConfig(lambda_zero=.05, lambda_temporal=.05,
                                  lambda_spatial=.02, lag=1, time_step=4.,
                                  numerical_refinements=2)
    paths = [Path(__file__), Path("airproof/v6_estimator.py"),
             Path("airproof/predictor_wrapper.py"), data_path, original_path,
             center_lock, center_path, calibration_path]
    registration = {
        "role": "registered estimator development on exposed archive and synthetic citizen replays; not real-data or independent confirmation",
        "mechanism": "same-cell/acquisition dispersion gate: efficient mean+quadratic score for a tight group; median+bounded Huber otherwise",
        "scientific_reason": "individual Huber saturates on mutually consistent true-event residuals far from public center",
        "public_center": center_id, "archive_interval_hours": 4, "lag_epochs": 1,
        "known_noise_scale": 3., "cap": 8., "base_config": asdict(base_config),
        "candidate_family": candidates,
        "development_seeds": list(range(916010, 916018)),
        "validation_seeds": list(range(916100, 916112)),
        "selection_gates": {"clean_ratio": 1.05, "attack_attenuation": .2,
                            "event_recall_loss": .05, "solver_failure_rate": 0.},
        "selection_order": "eligible only; lowest worst normalized gate ratio, then id",
        "coordinated_inlier_status": "mandatory post-selection stress; gate offers no theorem when corrupted reports form a tight majority",
        "hashes": {str(path): _sha(path) for path in paths},
        "validation_read_only_if_development_selects": True,
        "confirmation_authorized": False,
    }
    (out / "registration.json").write_text(json.dumps(registration, indent=2))

    data = np.asarray(rdata.read_rda(str(data_path))["newpoll"], float).T
    with np.load(original_path) as archive:
        coordinates = archive["coordinates_lat_lon"]
    coordinates = (coordinates-coordinates.mean(0))*np.array(
        [111.2, 111.2*np.cos(np.deg2rad(coordinates[:, 0].mean()))])
    with np.load(center_path) as archive:
        public = archive["prediction"].copy()
    truth = data[775:912]
    with np.load(calibration_path) as archive:
        threshold = float(np.quantile(archive["truth"], .95))
    extended_public = np.concatenate((public, np.repeat(public[-1:], 6, axis=0)))
    # Drain epochs are present only for estimator finalization and never scored.
    start = time.perf_counter()
    development_rows = _run_partition(extended_public, truth, coordinates, threshold,
        registration["development_seeds"], candidates, base_config,
        out/"development.jsonl", "development", len(truth))
    development = _summary(development_rows, candidates)
    if any(row["solver_failure_rate"] > 0 for row in development_rows):
        for item in development:
            item["eligible"] = False
    eligible = [item for item in development if item["eligible"]]
    def selection_key(item):
        attack_ratio = max(.2/item["attacks"][kind]["attenuation"]
                           for kind in ("adversarial_drift", "negative_bias"))
        recall_gap = max(0., item["quadratic_event_recall"]-.05-item["event_recall"])
        return (max(item["clean_ratio"]/1.05, attack_ratio, recall_gap/.05), item["id"])
    selected = min(eligible, key=selection_key)["id"] if eligible else None
    lock = {"selected": selected, "development": development,
            "validation_metrics_read_for_selection": False,
            "confirmation_authorized": False}
    (out/"selection_lock.json").write_text(json.dumps(lock, indent=2))
    validation_rows = []
    validation = []
    if selected is not None:
        chosen = [candidate for candidate in candidates if candidate["id"] == selected]
        validation_rows = _run_partition(extended_public, truth, coordinates, threshold,
            registration["validation_seeds"], chosen, base_config,
            out/"validation.jsonl", "validation", len(truth))
        validation = _summary(validation_rows, chosen)
    result = {"role": registration["role"], "selected": selected,
              "development": development, "validation": validation,
              "development_rows": len(development_rows),
              "validation_rows": len(validation_rows),
              "elapsed_seconds": time.perf_counter()-start,
              "confirmation_authorized": False}
    (out/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

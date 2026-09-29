"""Registered event-aware public-center development on an exposed archive."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

from pathlib import Path
import hashlib
import json

import numpy as np
import rdata

from airproof.v6_feasibility import cap_rmse_floor
from airproof.v6_public_dynamics import fit_public_increment
from airproof.v6_public_event_center import make_high_state_hold


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path("reports/v6/public_event_center_development")
    out.mkdir(exist_ok=False)
    data_path = Path("data/external/baselines/esntnn_author/Data/APFour.RData")
    old_path = Path("reports/v4_validation/esntnn_purpleair_7800/heldout_predictions.npz")
    validation_path = Path("reports/v6/backbone_validation_recovery/public_validation_predictions.npz")
    selected_lock = Path("reports/v6/public_dynamics_development/selection_lock.json")
    source_paths = [Path(__file__), Path("airproof/v6_public_event_center.py"),
                    Path("airproof/v6_public_dynamics.py"), data_path, old_path,
                    validation_path, selected_lock]
    family = [
        {"id": f"hold_q{int(100*q)}_b{blend:g}", "threshold_quantile": q, "blend": blend}
        for q in (.90, .95) for blend in (.5, 1.)
    ]
    registration = {
        "role": "post hoc registered mechanism development on an already exposed archive; not confirmation",
        "scientific_reason": "rare high-state public underprediction dominates cap floor and event loss",
        "fit": [0, 638], "validation": [638, 775], "exposed_test": [775, 912],
        "family": family,
        "base_model": "increment ridge1 event-weight1, refit on the same closed training prefix",
        "rule": "when last public value exceeds the training quantile, lift a low base forecast toward that last value",
        "forecast_information": "public archive through t-1 only",
        "selection": [
            "finite and causal outputs",
            "validation RMSE no more than 1.05 times the selected increment baseline",
            "validation event recall no lower than the selected increment baseline",
            "validation cap8 floor no higher than the selected increment baseline",
            "among eligible candidates: highest event recall, then lowest RMSE, then id",
        ],
        "event_endpoint": "truth at or above training-prefix 95th percentile",
        "hashes": {str(path): _sha(path) for path in source_paths},
        "cap": 8.0, "confirmation_authorized": False,
    }
    (out / "registration.json").write_text(json.dumps(registration, indent=2))

    data = np.asarray(rdata.read_rda(str(data_path))["newpoll"], float).T
    with np.load(old_path) as archive:
        coordinates = archive["coordinates_lat_lon"]
        test_backbone = archive["ESN_TNN"].copy()
    distances = ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :4]
    with np.load(validation_path) as archive:
        validation_backbone = archive["prediction"].copy()
    event_threshold = float(np.quantile(data[:638], .95))

    def metrics(prediction, truth):
        event = truth >= event_threshold
        return {
            "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
            "recall": float(np.mean(prediction[event] >= event_threshold)) if event.any() else None,
            "event_support": int(event.sum()),
            "cap8_floor": cap_rmse_floor(truth, prediction, 8.0),
        }

    base = fit_public_increment(data[:638], neighbors, ridge=1.0, event_weight=1.0)
    base_validation_prediction = np.array([base.predict_next(data[:t]) for t in range(638, 775)])
    base_validation = metrics(base_validation_prediction, data[638:775])
    validation = []
    models = {}
    for candidate in family:
        model = make_high_state_hold(base, data[:638], **{k: candidate[k] for k in ("threshold_quantile", "blend")})
        prediction = np.array([model.predict_next(data[:t]) for t in range(638, 775)])
        result = metrics(prediction, data[638:775])
        eligible = (np.isfinite(prediction).all()
                    and result["rmse"] <= 1.05 * base_validation["rmse"]
                    and result["recall"] >= base_validation["recall"]
                    and result["cap8_floor"] <= base_validation["cap8_floor"] + 1e-12)
        validation.append({"id": candidate["id"], "metrics": result, "eligible": bool(eligible)})
        models[candidate["id"]] = model
        np.savez_compressed(out / f"{candidate['id']}_validation.npz", prediction=prediction,
                            threshold=model.threshold, blend=model.blend,
                            threshold_quantile=model.threshold_quantile)
    eligible = [row for row in validation if row["eligible"]]
    selected = min(eligible, key=lambda row: (-row["metrics"]["recall"], row["metrics"]["rmse"], row["id"]))["id"] if eligible else None
    lock = {"selected": selected, "base_validation": base_validation, "validation": validation,
            "event_threshold": event_threshold, "test_metrics_read_for_selection": False}
    (out / "selection_lock.json").write_text(json.dumps(lock, indent=2))

    tests = []
    for identifier, model in models.items():
        prediction = np.array([model.predict_next(data[:t]) for t in range(775, 912)])
        tests.append({"id": identifier, "metrics": metrics(prediction, data[775:])})
        np.savez_compressed(out / f"{identifier}_test.npz", prediction=prediction)
    result = {
        "role": registration["role"], "selected": selected,
        "validation": validation, "base_validation": base_validation,
        "test_all_candidates_descriptive": tests,
        "test_increment": metrics(np.array([base.predict_next(data[:t]) for t in range(775, 912)]), data[775:]),
        "test_backbone": metrics(test_backbone, data[775:]),
        "validation_backbone": metrics(validation_backbone, data[638:775]),
        "confirmation_authorized": False,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

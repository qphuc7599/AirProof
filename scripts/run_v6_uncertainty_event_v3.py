"""Run the registered event-aware v3 only when every frozen PUBLIC input exists."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from airproof.v6_uncertainty import evaluate_intervals
from airproof.v6_uncertainty_event_v3 import fit_event_aware_intervals


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs" / "v6_uncertainty_event_v3.json"
OUT = ROOT / "reports" / "v6" / "uncertainty_event_v3"


def input_path(seed, cell):
    return ROOT / "reports" / "v6" / "uncertainty_v3_public_prerequisites_v1" / f"{seed}_{cell}" / "PUBLIC.npz"


def required_paths(registration):
    return [input_path(seed, cell)
            for role in ("fit", "calibration", "evaluation")
            for seed in registration[role]["seeds"]
            for cell in registration["cells"]]


def load(path, stride):
    with np.load(path) as data:
        missing = {"live", "reconstructed", "truth", "scales", "identity", "source_hash"} - set(data.files)
        if missing:
            raise ValueError(f"{path} misses arrays {sorted(missing)}")
        selected = np.arange(0, data["truth"].shape[1], stride)
        return {key: data[key][:, selected] for key in ("live", "reconstructed", "truth", "scales")}


def combine(paths, clock, stride):
    rows = [load(path, stride) for path in paths]
    return (np.concatenate([row[clock] for row in rows]),
            np.concatenate([row["truth"] for row in rows]),
            np.concatenate([row["scales"] for row in rows]))


def main():
    registration_bytes = REGISTRATION.read_bytes()
    registration = json.loads(registration_bytes)
    if registration["status"] != "registered-code-only-required-PUBLIC-artifacts-missing":
        raise ValueError("unexpected registration status")
    paths = required_paths(registration)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"v3 remains registration-only: {len(missing)}/{len(paths)} frozen PUBLIC artifacts missing; "
            f"first missing={missing[0].relative_to(ROOT).as_posix()}"
        )
    stride = registration["spatial_stride"]
    role_paths = {role: [input_path(seed, cell)
                         for seed in registration[role]["seeds"]
                         for cell in registration["cells"]]
                  for role in ("fit", "calibration", "evaluation")}
    models, results = {}, {}
    for clock in registration["clocks"]:
        fp, ft, fs = combine(role_paths["fit"], clock, stride)
        cp, ct, cs = combine(role_paths["calibration"], clock, stride)
        model = fit_event_aware_intervals(
            fp, ft, fs, cp, ct, cs, clock=clock,
            fit_split_id=registration["fit"]["split_id"],
            calibration_split_id=registration["calibration"]["split_id"],
            evaluation_split_id=registration["evaluation"]["split_id"],
            fit_end=registration["fit"]["end"],
            calibration_end=registration["calibration"]["end"],
            evaluation_start=registration["evaluation"]["start"],
            minimum_fit_support=registration["minimum_fit_support"],
        )
        ep, et, es = combine(role_paths["evaluation"], clock, stride)
        lower, upper = model.predict(ep, es, epochs=registration["evaluation"]["start"],
                                     evaluation_split_id=registration["evaluation"]["split_id"])
        results[clock] = evaluate_intervals(
            et, lower, upper, public_strata=model.labels(ep, es),
            event_mask=model.event_mask(et),
            block_length=registration["bootstrap"]["block_length"],
            bootstrap_replicates=registration["bootstrap"]["replicates"],
            seed=registration["bootstrap"]["seed"],
        )
        models[clock] = asdict(model)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "registration.json").write_bytes(registration_bytes)
    (OUT / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    manifest = {
        "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "role": registration["role"],
        "input_hashes": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},
        "post_v2_failure_design": True,
        "evaluation_residual_updates": 0,
        "epa_test_read": False,
        "confirmation_read": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Outcome-blind semantic readiness audit for lifetime validation v1."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_lifetime_budget7_validation_v2_inputs import (
    PREDICTION_FORBIDDEN,
    load_prediction_view,
    load_scoring_view,
    verify_bundle,
)
from airproof.v6_lifetime_budget7_validation_v2_runner import (
    REGISTRATION,
    load_registration,
    planned_jobs,
    verify_source_lock,
)


def audit(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    failures: list[str] = []
    try:
        registration = load_registration(root)
        jobs = planned_jobs(registration, smoke=False)
        source = verify_source_lock(root, registration)
        source_path = root / registration["input_producer"]["source_lock"]
        input_path = root / registration["input_producer"]["input_lock"]
        input_lock = json.loads(input_path.read_text(encoding="utf-8"))
        bundle = root / registration["input_producer"]["bundle"]
        manifest = verify_bundle(
            root, bundle, expected_manifest_sha256=input_lock["manifest_sha256"])
        if input_lock.get("source_lock_sha256") != sha256_file(source_path):
            failures.append("input lock does not bind source lock")
        if input_lock.get("scientific_outcomes_before_lock") != 0:
            failures.append("input lock was created after validation outcomes")
        for seed, cell in jobs:
            view = load_prediction_view(
                root, bundle, seed, cell,
                expected_manifest_sha256=input_lock["manifest_sha256"])
            if PREDICTION_FORBIDDEN & set(view):
                failures.append(f"prediction scoring leak:{seed}:{cell}")
            epochs = np.asarray(view["prediction_epochs"])
            expected_length = (registration["worlds"]["acquisition_epochs"]
                               + registration["worlds"]["fixed_lag_epochs"]
                               - registration["worlds"]["burn_in_epochs"])
            if len(epochs) != expected_length or epochs[0] != registration["worlds"]["burn_in_epochs"]:
                failures.append(f"prediction horizon mismatch:{seed}:{cell}")
        for seed in registration["worlds"]["seeds"]:
            clean = load_scoring_view(bundle, seed, "severe_clean")
            for attack in ("severe_drift", "severe_hotspot"):
                attacked = load_scoring_view(bundle, seed, attack)
                for field in ("evaluation_truth", "event_mask", "event_threshold", "cell_groups"):
                    if not np.array_equal(clean[field], attacked[field]):
                        failures.append(f"attack scoring context differs:{seed}:{attack}:{field}")
        expected_rows = len(jobs) * len(registration["methods"])
        if expected_rows != 240:
            failures.append(f"exact row contract changed:{expected_rows}")
        identities = {
            "registration_sha256": sha256_file(root / REGISTRATION),
            "source_lock_sha256": sha256_file(source_path),
            "input_lock_sha256": sha256_file(input_path),
            "input_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "input_payload_root_sha256": manifest["payload_root_sha256"],
            "source_files": len(source["source_sha256"]),
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(str(exc))
        identities = {}
        expected_rows = 240
    return {
        "schema_version": 1,
        "role": "outcome-blind semantic readiness; no estimator scoring",
        "pass": not failures, "failures": sorted(set(failures)),
        "expected_jobs": 60, "expected_rows": expected_rows,
        "prediction_scoring_capabilities_separate": not failures,
        "scientific_outcomes_read_or_generated": 0,
        "protected_namespaces_opened": False,
        **identities,
    }


def main() -> None:
    registration = load_registration(ROOT)
    result = audit(ROOT)
    output = ROOT / registration["input_producer"]["readiness"]
    write_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

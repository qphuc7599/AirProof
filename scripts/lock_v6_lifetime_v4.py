"""Create the v4 source/control lock before any lifetime-exposure outcome."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_lifetime_registration import assert_registration
from airproof.v6_lifetime_runner import REGISTRATION, load_registration

SOURCES = [
    REGISTRATION,
    "airproof/v6_lifetime_exposure.py",
    "airproof/v6_lifetime_runner.py",
    "airproof/v6_lifetime_registration.py",
    "airproof/v6_estimator.py",
    "airproof/v6_covariance_forcing_runner.py",
    "airproof/v6_covariance_forcing_inputs.py",
    "airproof/v6_solver_refinement.py",
    "airproof/metrics.py",
    "airproof/records.py",
    "airproof/meteorology.py",
    "airproof/field.py",
    "scripts/run_v6_lifetime_v4.py",
    "scripts/analyze_v6_lifetime_v4.py",
    "scripts/audit_v6_lifetime_v4_readiness.py",
    "scripts/lock_v6_lifetime_v4.py",
    "tests/test_v6_lifetime_exposure.py",
    "tests/test_v6_lifetime_v4_contract.py",
]


def main() -> None:
    registration = load_registration(ROOT)
    assert_registration(registration)
    destination = ROOT / registration["artifacts"]["source_lock"]
    if destination.exists():
        raise SystemExit("v4 lifetime source lock already exists; overwrite prohibited")
    outcome = ROOT / registration["artifacts"]["outcomes"]
    existing_outcomes = list(outcome.rglob("*")) if outcome.exists() else []
    if any(path.is_file() for path in existing_outcomes):
        raise SystemExit("v4 lifetime outcome artifact exists before source lock")
    source_hashes = {}
    for relative in SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        source_hashes[relative] = sha256_file(path)
    reused = {}
    root = ROOT / registration["controls"]["source_campaign"] / "jobs"
    candidate = registration["controls"]["source_candidate"]
    for seed in registration["inputs"]["development_seeds"]:
        for cell in registration["inputs"]["cells"]:
            result = root / str(seed) / cell / "result.json"
            reused[str(result.relative_to(ROOT)).replace("\\", "/")] = sha256_file(result)
            for method in registration["controls"]["reused_methods"]:
                arm = root / str(seed) / cell / candidate / method
                for name in ("prediction.npz", "prediction_manifest.json"):
                    path = arm / name
                    reused[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    v2_analysis = ROOT / registration["development_history"]["v2_analysis"]
    v3_analysis = ROOT / registration["development_history"]["v3_analysis"]
    v3_diagnosis = ROOT / registration["development_history"]["v3_diagnosis"]
    v2_lock = ROOT / registration["inputs"]["input_lock"]
    payload = {
        "schema_version": 1,
        "role": "immutable pre-v4-lifetime-outcome source and reused-control lock",
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "source_sha256": source_hashes,
        "v2_analysis_sha256": sha256_file(v2_analysis),
        "v3_analysis_sha256": sha256_file(v3_analysis),
        "v3_diagnosis_sha256": sha256_file(v3_diagnosis),
        "v2_input_lock_sha256": sha256_file(v2_lock),
        "reused_control_sha256": reused,
        "reused_control_files": len(reused),
        "new_outcomes_before_lock": 0,
        "historical_v4_rerun": False,
        "protected_namespaces_opened": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, payload)
    print(json.dumps({"locked": True,
        "source_lock_sha256": sha256_file(destination),
        "source_files": len(source_hashes),
        "reused_control_files": len(reused),
        "new_outcomes_before_lock": 0}, indent=2))


if __name__ == "__main__":
    main()

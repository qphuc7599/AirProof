"""Lock repair-v5 sources, frozen validation inputs, and reused controls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_lifetime_budget_repair_registration import assert_registration
from airproof.v6_lifetime_budget_repair_runner import REGISTRATION, load_registration

SOURCES = [
    REGISTRATION,
    "airproof/v6_lifetime_budget_repair_registration.py",
    "airproof/v6_lifetime_budget_repair_runner.py",
    "airproof/v6_lifetime_exposure.py",
    "airproof/v6_lifetime_validation_inputs.py",
    "airproof/v6_covariance_forcing_inputs.py",
    "airproof/v6_covariance_forcing_runner.py",
    "airproof/v6_estimator.py", "airproof/v6_solver_refinement.py",
    "airproof/metrics.py", "airproof/records.py",
    "scripts/lock_v6_lifetime_budget_repair_v5.py",
    "scripts/audit_v6_lifetime_budget_repair_v5_readiness.py",
    "scripts/run_v6_lifetime_budget_repair_v5.py",
    "scripts/analyze_v6_lifetime_budget_repair_v5.py",
    "tests/test_v6_lifetime_budget_repair_v5.py",
]


def main() -> None:
    registration = load_registration(ROOT)
    assert_registration(registration)
    destination = ROOT / registration["artifacts"]["source_lock"]
    if destination.exists():
        raise SystemExit("repair source lock already exists; overwrite prohibited")
    outcome = ROOT / registration["artifacts"]["outcomes"]
    if outcome.exists() and any(path.is_file() for path in outcome.rglob("*")):
        raise SystemExit("repair candidate outcome exists before source lock")
    source_hashes = {}
    for relative in SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        source_hashes[relative] = sha256_file(path)
    reuse = registration["input_reuse"]
    upstream_paths = [
        reuse["input_lock"], reuse["input_manifest"],
        registration["failure_lineage"]["parent_development"],
        registration["failure_lineage"]["parent_validation"],
        registration["failure_lineage"]["statistical_audit"],
        registration["failure_lineage"]["mechanism_diagnosis"],
        registration["fixed_estimator"]["innovation_calibration"],
    ]
    upstream = {relative: sha256_file(ROOT / relative) for relative in upstream_paths}
    controls = {}
    parent = ROOT / reuse["source_validation_outcomes"] / "jobs"
    for seed in reuse["seeds"]:
        for cell in reuse["cells"]:
            result = parent / str(seed) / cell / "result.json"
            relative = result.relative_to(ROOT).as_posix()
            controls[relative] = sha256_file(result)
            payload = json.loads(result.read_text(encoding="utf-8"))
            if (payload.get("status") != "complete"
                    or {row.get("method") for row in payload.get("rows", [])}
                    != set(registration["controls"]["reused_methods"])):
                raise ValueError(f"incomplete reused controls: {relative}")
            for method in registration["controls"]["reused_methods"]:
                for name in ("prediction.npz", "prediction_manifest.json"):
                    path = result.parent / method / name
                    controls[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    upstream.update(controls)
    payload = {"schema_version": 1,
        "role": "immutable pre-candidate repair source, reused-input, and reused-control lock",
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "source_sha256": source_hashes, "upstream_sha256": upstream,
        "upstream_control_files": len(controls),
        "candidate_outcomes_before_lock": 0, "controls_recomputed": False,
        "new_world_or_transport_generation": False,
        "validation_v1_rescued": False, "protected_namespaces_opened": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, payload)
    print(json.dumps({"locked": True, "source_lock_sha256": sha256_file(destination),
        "source_files": len(source_hashes), "upstream_control_files": len(controls),
        "candidate_outcomes_before_lock": 0}, indent=2))


if __name__ == "__main__":
    main()

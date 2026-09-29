"""Outcome-blind readiness audit for exposed lifetime-budget repair v5."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_lifetime_validation_inputs import PREDICTION_FORBIDDEN, load_prediction_view, verify_bundle
from airproof.v6_lifetime_budget_repair_runner import load_registration, planned_jobs, verify_source_lock


def audit(root: str | Path = ROOT) -> dict:
    root = Path(root)
    failures = []
    try:
        registration = load_registration(root)
        jobs = planned_jobs(registration, smoke=False)
        source = verify_source_lock(root, registration)
        source_path = root / registration["artifacts"]["source_lock"]
        input_path = root / registration["input_reuse"]["input_lock"]
        input_lock = json.loads(input_path.read_text(encoding="utf-8"))
        bundle = root / registration["input_reuse"]["bundle"]
        manifest = verify_bundle(root, bundle,
            expected_manifest_sha256=input_lock["manifest_sha256"])
        if sha256_file(bundle / "manifest.json") != sha256_file(
                root / registration["input_reuse"]["input_manifest"]):
            failures.append("registered reused input manifest differs")
        for seed, cell in jobs:
            view = load_prediction_view(root, bundle, seed, cell,
                expected_manifest_sha256=input_lock["manifest_sha256"])
            if PREDICTION_FORBIDDEN & set(view):
                failures.append(f"prediction scoring leak:{seed}:{cell}")
            parent = (root / registration["input_reuse"]["source_validation_outcomes"]
                      / "jobs" / str(seed) / cell / "result.json")
            result = json.loads(parent.read_text(encoding="utf-8"))
            if ({row.get("method") for row in result.get("rows", [])}
                    != set(registration["controls"]["reused_methods"])):
                failures.append(f"reused controls incomplete:{seed}:{cell}")
        outcome = root / registration["artifacts"]["outcomes"]
        existing = list(outcome.rglob("*")) if outcome.exists() else []
        if any(path.is_file() for path in existing):
            failures.append("candidate outcomes exist before readiness")
        identities = {"registration_sha256": sha256_file(root / "configs/v6/lifetime_budget_repair_development_v5.json"),
            "source_lock_sha256": sha256_file(source_path),
            "input_lock_sha256": sha256_file(input_path),
            "input_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "input_payload_root_sha256": manifest["payload_root_sha256"],
            "source_files": len(source["source_sha256"]),
            "upstream_control_files": source["upstream_control_files"]}
    except Exception as exc:  # noqa: BLE001 - preserve every preflight failure
        failures.append(f"{type(exc).__name__}: {exc}")
        identities = {}
    return {"schema_version": 1,
        "role": "repair-v5 outcome-blind readiness on frozen validation-v1 inputs",
        "pass": not failures, "failures": sorted(set(failures)),
        "expected_jobs": 60, "expected_candidate_rows": 120,
        "reused_control_rows": 240, "comparison_rows": 600,
        "scientific_outcomes_read_or_generated": 0,
        "controls_recomputed": False, "new_world_or_transport_generation": False,
        "validation_v1_rescued": False, "protected_namespaces_opened": False,
        **identities}


def main() -> None:
    registration = load_registration(ROOT)
    result = audit(ROOT)
    output = ROOT / registration["artifacts"]["readiness"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

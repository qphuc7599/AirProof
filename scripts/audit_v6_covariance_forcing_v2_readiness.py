"""Outcome-blind readiness audit for the frozen multi-world v2 protocol."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import (
    PREDICTION_FORBIDDEN,
    REGISTRATION,
    load_csr,
    sha256_file,
    verify_campaign_bundle,
    write_json,
)
from airproof.v6_covariance_forcing_registration import (
    assert_exact_family,
    assert_fixed_contract,
    assert_seed_registry,
    registered_seeds,
)


def audit() -> dict:
    registration = json.loads((ROOT / REGISTRATION).read_text(encoding="utf-8"))
    checks: dict[str, dict] = {}

    def check(name, operation):
        try:
            checks[name] = {"pass": True, "detail": operation()}
        except Exception as exc:  # noqa: BLE001 - every readiness failure must be recorded
            checks[name] = {"pass": False, "detail": f"{type(exc).__name__}: {exc}"}

    check("registration", lambda: "frozen schema3 zero-outcome protocol" if (
        registration["schema_version"] == 3
        and registration["status"] == "prospective_protocol_frozen_before_input_generation_and_outcomes"
        and "mechanistic-synthetic" in registration["role"]
        and registration["historical_reuse"]["rerun"] is False)
        else (_ for _ in ()).throw(ValueError("registration identity/scope changed")))
    check("fixed_estimator", lambda: (assert_exact_family(registration),
        assert_fixed_contract(registration), "exact2x2 cap8 delta1.345 lag6 original gates")[2])
    checks["same_information_contract"] = dict(checks["fixed_estimator"])
    registry = json.loads((ROOT / registration["seed_namespace"]["registry"]).read_text())
    check("seed_namespace", lambda: (assert_seed_registry(registration, registry),
        "eight disjoint calibration and eight development worlds reserved")[1])

    input_dir = ROOT / registration["producer"]["campaign_bundle"]
    lock_path = ROOT / registration["producer"]["input_lock"]
    lock: dict = {}
    manifest: dict = {}
    def input_lock():
        nonlocal lock
        lock = json.loads(lock_path.read_text())
        if (lock["registration_sha256"] != sha256_file(ROOT / REGISTRATION)
                or lock["scientific_outcomes_before_lock"] != 0
                or lock["protected_namespaces_opened"] is not False):
            raise ValueError("input lock does not bind the zero-outcome registration")
        return {"sha256": sha256_file(lock_path),
                "input_manifest_sha256": lock["input_manifest_sha256"]}
    check("input_lock", input_lock)

    def bundle():
        nonlocal manifest
        manifest = verify_campaign_bundle(ROOT, input_dir,
            expected_manifest_sha256=lock["input_manifest_sha256"])
        return {"payloads": len(manifest["payload_sha256"]),
                "payload_root_sha256": manifest["payload_root_sha256"]}
    check("producer_hashes", bundle)

    def source_snapshot():
        mismatches = []
        for relative, expected in manifest["source_files"].items():
            current = ROOT / relative
            snapshot = input_dir / "source_snapshot" / relative
            if (not current.is_file() or not snapshot.is_file()
                    or sha256_file(current) != expected or sha256_file(snapshot) != expected):
                mismatches.append(relative)
        if mismatches:
            raise ValueError(f"source/snapshot mismatches: {mismatches}")
        return {"files": len(manifest["source_files"]), "source_digest": manifest["source_digest"]}
    check("source_snapshot", source_snapshot)

    def global_calibration():
        value = json.loads((input_dir / "global_calibration.json").read_text())
        calibration, _development = registered_seeds(registration)
        if (set(value["calibration_worlds"]) != calibration
                or value["development_worlds_read"] is not False
                or value["fold_unit"] != "world" or value["fold_count"] != 8
                or set(value["innovation_scales"]) != set(registration["family_axes"]["innovation_covariance"])
                or not all(np.isfinite(list(value["innovation_scales"].values())))
                or min(value["innovation_scales"].values()) <= 0):
            raise ValueError("global calibration role, folds, or scales invalid")
        for fold in value["world_cross_fitted_folds"]:
            if (fold["held_world"] in fold["training_worlds"]
                    or set(fold["training_worlds"]) != calibration - {fold["held_world"]}):
                raise ValueError("world-level fold leakage")
        return {"worlds": 8, "folds": 8, "scales": value["innovation_scales"]}
    check("global_calibration", global_calibration)
    checks["world_level_folds"] = dict(checks["global_calibration"])

    def sparse_operator():
        operator = load_csr(input_dir / "mechanism" / "physical_operator.npz")
        if operator.shape != (1024, 1024) or operator.nnz > 5 * 1024:
            raise ValueError("physical operator is not the registered sparse grid operator")
        return {"shape": list(operator.shape), "nnz": int(operator.nnz)}
    check("sparse_operator", sparse_operator)

    def causal_prediction_inputs():
        for seed in registration["world_roles"]["development"]["seeds"]:
            directory = input_dir / "prediction" / str(seed)
            with np.load(directory / "context.npz", allow_pickle=False) as arrays:
                if PREDICTION_FORBIDDEN & set(arrays.files):
                    raise ValueError("truth/scoring field in prediction context")
                epochs = arrays["prediction_epochs"]
                if (epochs[0] != 48 or epochs[-1] != 677
                        or np.any(arrays["public_center_available_at"] > epochs)
                        or np.any(arrays["physical_forcing_available_at"] > epochs)
                        or not np.isfinite(arrays["residual_forcing"]).all()):
                    raise ValueError("noncausal or incomplete prediction context")
            for cell in registration["world_roles"]["development"]["cells"]:
                with np.load(directory / f"observations_{cell}.npz", allow_pickle=False) as rows:
                    if PREDICTION_FORBIDDEN & set(rows.files) or np.any(rows["arrival"] < rows["epoch"]):
                        raise ValueError("prediction observations contain labels or precognition")
            scoring = input_dir / "scoring" / str(seed) / "scoring_only.npz"
            if not scoring.is_file() or scoring.parent == directory:
                raise ValueError("scoring capability is absent or colocated with prediction")
        return "all eight prediction bundles causal; scoring stored separately"
    check("producer_causality", causal_prediction_inputs)
    checks["prediction_truth_isolation"] = dict(checks["producer_causality"])

    def runner_adapter():
        from airproof.v6_covariance_forcing_runner import execute_registered_plan
        return "callable without invocation" if callable(execute_registered_plan) else \
            (_ for _ in ()).throw(ValueError("runner adapter is not callable"))
    check("runner_adapter", runner_adapter)
    required = set(registration["readiness"]["required_checks"])
    missing = sorted(required - set(checks))
    if missing:
        checks["required_check_inventory"] = {"pass": False, "detail": missing}
    else:
        checks["required_check_inventory"] = {"pass": True, "detail": sorted(required)}
    outcome_dir = ROOT / registration["output"]["directory"]
    existing_outcomes = list(outcome_dir.rglob("result.json")) if outcome_dir.exists() else []
    if existing_outcomes:
        checks["outcome_blind"] = {"pass": False,
            "detail": "readiness cannot be regenerated after development outcomes exist"}
    else:
        checks["outcome_blind"] = {"pass": True, "detail": "zero result artifacts"}
    passed = all(row["pass"] for row in checks.values())
    return {"role": "outcome-blind v2 readiness only", "registration": REGISTRATION,
        "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "input_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "readiness_pass": passed,
        "decision": "ready_for_registered_runtime_smoke" if passed else "blocked_do_not_run",
        "checks": checks, "scientific_outcomes_read_or_generated": 0,
        "epa_validation_test_primary_confirmation_opened": False}


def main() -> None:
    registration = json.loads((ROOT / REGISTRATION).read_text())
    destination = ROOT / registration["readiness"]["artifact"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    write_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

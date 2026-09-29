"""Create the immutable lifetime-validation source lock before inputs or outcomes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_lifetime_validation_registration import (
    BOOTSTRAP_NAMESPACE,
    BOOTSTRAP_SEED,
    CELLS,
    METHODS,
    SEEDS,
    assert_registration,
    assert_seed_registry,
)

REGISTRATION = "configs/v6/causal_lifetime_exposure_validation_v1.json"
SOURCES = [
    REGISTRATION,
    "configs/v6/protocol.yaml",
    "reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml",
    "airproof/audit.py",
    "airproof/config.py",
    "airproof/continual_release.py",
    "airproof/dtn.py",
    "airproof/experiment.py",
    "airproof/field.py",
    "airproof/integrity.py",
    "airproof/intersectional.py",
    "airproof/meteorology.py",
    "airproof/metrics.py",
    "airproof/privacy.py",
    "airproof/records.py",
    "airproof/reference.py",
    "airproof/rng.py",
    "airproof/scheduler.py",
    "airproof/simulator.py",
    "airproof/twin.py",
    "airproof/v5_estimator.py",
    "airproof/v5_experiment.py",
    "airproof/v5_fairness.py",
    "airproof/v5_transport.py",
    "airproof/v5_wire_control.py",
    "airproof/v6_audit_return.py",
    "airproof/v6_covariance_forcing_inputs.py",
    "airproof/v6_covariance_forcing_runner.py",
    "airproof/v6_estimator.py",
    "airproof/v6_experiment.py",
    "airproof/v6_fairness.py",
    "airproof/v6_lifetime_exposure.py",
    "airproof/v6_lifetime_validation_inputs.py",
    "airproof/v6_lifetime_validation_registration.py",
    "airproof/v6_lifetime_validation_runner.py",
    "airproof/v6_mobility.py",
    "airproof/v6_public_checkpoint.py",
    "airproof/v6_receipts.py",
    "airproof/v6_residual_dynamics.py",
    "airproof/v6_resources.py",
    "airproof/v6_solver_refinement.py",
    "airproof/v6_transport.py",
    "scripts/analyze_v6_lifetime_validation_v1.py",
    "scripts/audit_v6_lifetime_validation_v1_readiness.py",
    "scripts/build_v6_lifetime_validation_v1_inputs.py",
    "scripts/lock_v6_lifetime_validation_v1.py",
    "scripts/run_v6_lifetime_validation_v1.py",
    "tests/test_v6_lifetime_validation_input_registration_contract.py",
]
SOURCES = sorted(set(SOURCES) | {
    path.relative_to(ROOT).as_posix()
    for path in (ROOT / "airproof").glob("*.py")
} | {
    "configs/v6/seed_registry.json",
    "configs/v6/v4_reuse_hash_lock.json",
    "configs/v6/causal_lifetime_exposure_development_v4.json",
    "reports/v6/causal_lifetime_exposure_development_v4/outcomes/analysis.json",
    "reports/v6/innovation_covariance_forcing_development_v2/input_lock.json",
    "reports/v6/innovation_covariance_forcing_development_v2/inputs/global_calibration.json",
})


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _evidence_path(registration: dict[str, Any], relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    try:
        normalized = candidate.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path leaves repository: {relative}") from exc
    protected = [value.casefold() for value in registration["protected_namespaces"]]
    folded = normalized.casefold()
    if any(folded == prefix or folded.startswith(f"{prefix}/") for prefix in protected):
        raise ValueError(f"evidence path enters protected namespace: {relative}")
    return candidate


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} is missing or changed: {_relative(path)}")


def _assert_empty_validation_namespace(registration: dict[str, Any]) -> None:
    producer = registration["input_producer"]
    forbidden = [
        ROOT / producer["bundle"],
        ROOT / f"{producer['bundle']}.building",
        ROOT / producer["input_lock"],
        ROOT / producer["readiness"],
        ROOT / registration["artifacts"]["outcomes"],
    ]
    present = [path for path in forbidden if path.exists()]
    if present:
        names = [_relative(path) for path in present]
        raise SystemExit(f"validation input, staging, readiness, or outcome exists: {names}")
    artifact_root = ROOT / registration["artifacts"]["directory"]
    destination = ROOT / producer["source_lock"]
    files = ([path for path in artifact_root.rglob("*") if path.is_file()]
             if artifact_root.exists() else [])
    unexpected = [path for path in files if path != destination]
    if unexpected:
        raise SystemExit(
            f"validation namespace is not zero-artifact: {[_relative(p) for p in unexpected]}"
        )


def _verify_development_nomination(
    registration: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    nomination = registration["nomination"]
    development_path = _evidence_path(registration, nomination["development_registration"])
    analysis_path = _evidence_path(registration, nomination["development_analysis"])
    development = _load(development_path)
    analysis = _load(analysis_path)
    selected = nomination["selected_candidate"]
    candidate = next(
        (item for item in development.get("candidates", []) if item.get("id") == selected),
        None,
    )
    matrix = analysis.get("matrix_audit", {})
    gates = analysis.get("gate_evaluation", {})
    if (analysis.get("registration") != nomination["development_registration"]
            or analysis.get("registration_sha256") != _sha256(development_path)
            or matrix.get("pass") is not True
            or matrix.get("v2_controls_recomputed") is not False
            or gates.get("selected_candidate") != selected
            or gates.get("family_closed") is not False
            or gates.get("extension_authorized") is not False
            or gates.get("scope")
            != "sequential exposed mechanistic-synthetic development; no p-values"
            or analysis.get("historical_v4_rerun") is not False
            or candidate != {"id": "lifetime_budget28", "per_user_exposure_budget": 28.0}):
        raise ValueError("development nomination is incomplete, changed, or ineligible")

    development_lock_path = _evidence_path(
        registration, development["artifacts"]["source_lock"]
    )
    development_lock = _load(development_lock_path)
    if (development_lock.get("registration_sha256") != _sha256(development_path)
            or development_lock.get("new_outcomes_before_lock") != 0
            or development_lock.get("historical_v4_rerun") is not False
            or matrix.get("source_lock_sha256") != _sha256(development_lock_path)):
        raise ValueError("development source lock or analysis provenance changed")
    evidence_paths = {
        "v2_analysis_sha256": _evidence_path(
            registration, development["development_history"]["v2_analysis"]
        ),
        "v3_analysis_sha256": _evidence_path(
            registration, development["development_history"]["v3_analysis"]
        ),
        "v3_diagnosis_sha256": _evidence_path(
            registration, development["development_history"]["v3_diagnosis"]
        ),
        "v2_input_lock_sha256": _evidence_path(
            registration, development["inputs"]["input_lock"]
        ),
    }
    for field, path in evidence_paths.items():
        _require_hash(path, development_lock[field], field)
    for relative, expected in development_lock.get("reused_control_sha256", {}).items():
        _require_hash(
            _evidence_path(registration, relative), expected, "reused development control"
        )
    return ({
        "development_registration_sha256": _sha256(development_path),
        "development_analysis_sha256": _sha256(analysis_path),
        "development_source_lock_sha256": _sha256(development_lock_path),
        **{field: _sha256(path) for field, path in evidence_paths.items()},
    }, analysis)


def _verify_historical_reuse(registration: dict[str, Any]) -> dict[str, Any]:
    reuse = registration["historical_reuse"]
    lock_path = _evidence_path(registration, reuse["v4_reuse_lock"])
    lock = _load(lock_path)
    if (lock.get("schema_version") != 1
            or lock.get("role") != (
                "immutable expected hashes for retained historical-v4 evidence; "
                "verification only, never outcome regeneration"
            )):
        raise ValueError("historical v4 reuse lock identity changed")
    files = lock.get("files")
    if not isinstance(files, dict) or reuse["v4_primary"] not in files:
        raise ValueError("registered historical v4 evidence is absent from reuse lock")
    for relative, expected in files.items():
        _require_hash(
            _evidence_path(registration, relative), expected, "historical v4 evidence"
        )
    return {
        "v4_reuse_lock_sha256": _sha256(lock_path),
        "v4_primary_sha256": files[reuse["v4_primary"]],
        "verified_historical_files": len(files),
        "historical_v4_rerun": False,
    }


def _assert_sources_outside_protected(registration: dict[str, Any]) -> None:
    protected = tuple(f"{value.rstrip('/')}" for value in registration["protected_namespaces"])
    offenders = [
        source for source in SOURCES
        if any(source == prefix or source.startswith(f"{prefix}/") for prefix in protected)
    ]
    if offenders:
        raise ValueError(f"source inventory enters a protected namespace: {offenders}")


def main() -> None:
    registration_path = ROOT / REGISTRATION
    registration = _load(registration_path)
    assert_registration(registration)
    destination = ROOT / registration["input_producer"]["source_lock"]
    if destination.exists():
        raise SystemExit("validation source lock already exists; overwrite prohibited")
    _assert_empty_validation_namespace(registration)

    registry_path = ROOT / registration["worlds"]["seed_registry"]
    registry = _load(registry_path)
    assert_seed_registry(registration, registry)
    nomination_hashes, _ = _verify_development_nomination(registration)
    historical = _verify_historical_reuse(registration)
    _assert_sources_outside_protected(registration)

    source_hashes: dict[str, str] = {}
    for relative in SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        source_hashes[relative] = _sha256(path)
    calibration = ROOT / registration["fixed_estimator"]["innovation_calibration"]
    if not calibration.is_file():
        raise FileNotFoundError(_relative(calibration))

    payload = {
        "schema_version": 1,
        "role": (
            "immutable pre-validation-input and pre-validation-outcome source, nomination, "
            "and historical-reuse lock"
        ),
        "registration": REGISTRATION,
        "registration_sha256": _sha256(registration_path),
        "source_sha256": source_hashes,
        **nomination_hashes,
        "innovation_calibration_sha256": _sha256(calibration),
        "seed_registry_sha256": _sha256(registry_path),
        "seed_contract": {
            "world_namespace": registration["worlds"]["seed_namespace"],
            "world_seeds": SEEDS,
            "bootstrap_namespace": BOOTSTRAP_NAMESPACE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "disjoint": True,
        },
        "matrix_contract": {
            "worlds": len(SEEDS),
            "cells": CELLS,
            "methods": METHODS,
            "jobs": len(SEEDS) * len(CELLS),
            "rows": len(SEEDS) * len(CELLS) * len(METHODS),
        },
        "evaluation_clock": registration["evaluation_clock"],
        "attack_contract": registration["attack_contract"],
        "scientific_gates": registration["scientific_gates"],
        "historical_reuse": historical,
        "validation_inputs_before_lock": 0,
        "validation_outcomes_before_lock": 0,
        "protected_namespaces_opened": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    print(json.dumps({
        "locked": True,
        "source_lock_sha256": _sha256(destination),
        "source_files": len(source_hashes),
        "historical_files_verified": historical["verified_historical_files"],
        "validation_inputs_before_lock": 0,
        "validation_outcomes_before_lock": 0,
    }, indent=2))


if __name__ == "__main__":
    main()

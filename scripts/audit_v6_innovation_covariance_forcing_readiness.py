"""Audit exact cached inputs for the registered 2x2 family; never run outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from airproof.v6_mechanistic_provenance import (
    assert_prediction_schema,
    assert_producer_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6/innovation_covariance_forcing_development_v1.json"
OUTPUT = ROOT / "reports/v6/innovation_covariance_forcing_development_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz_inventory(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False, "keys": []}
    with np.load(path, allow_pickle=False) as arrays:
        return {
            "exists": True,
            "keys": sorted(arrays.files),
            "shapes": {key: list(arrays[key].shape) for key in arrays.files},
            "sha256": sha256(path),
        }


def audit() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        (scale, dynamics)
        for scale in cfg["family_axes"]["innovation_scale"]
        for dynamics in cfg["family_axes"]["residual_dynamics"]
    }
    actual = {(row["innovation_scale"], row["residual_dynamics"]) for row in cfg["candidates"]}
    family_exact = len(cfg["candidates"]) == 4 and actual == expected
    schema_error = None
    try:
        assert_prediction_schema(cfg["prediction_input_fields"])
    except ValueError as exc:
        schema_error = str(exc)

    source = cfg["locked_source"]
    inventories = {
        "public_validation": npz_inventory(ROOT / source["public_validation_artifact"]),
        "public_evaluation": npz_inventory(ROOT / source["public_evaluation_artifact"]),
        "truth_evaluation": npz_inventory(ROOT / source["truth_evaluation_artifact"]),
    }
    exact_files = {
        key: {
            "path": value,
            "exists": (ROOT / value).is_file(),
            "sha256": sha256(ROOT / value) if (ROOT / value).is_file() else None,
        }
        for key, value in cfg["readiness"]["required_exact_files"].items()
    }

    manifest_error = "producer manifest absent"
    manifest_path = ROOT / cfg["readiness"]["required_exact_files"]["producer_manifest"]
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            required_producers = set(cfg["producer_contracts"])
            supplied = set(manifest.get("producers", {}))
            if supplied != required_producers:
                raise ValueError(
                    f"producer set mismatch: required={sorted(required_producers)} "
                    f"supplied={sorted(supplied)}"
                )
            for producer in manifest["producers"].values():
                assert_producer_manifest(producer)
            manifest_error = None
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            manifest_error = str(exc)

    blockers = []
    if not family_exact:
        blockers.append("registered candidates are not the exact Cartesian 2x2 family")
    if schema_error:
        blockers.append(schema_error)
    if not inventories["public_validation"]["exists"]:
        blockers.append("frozen public calibration prediction artifact is absent")
    elif not {"prediction", "truth", "indices"}.issubset(inventories["public_validation"]["keys"]):
        blockers.append("public calibration artifact lacks prediction/truth/index labels")
    else:
        blockers.append(
            "available public calibration arrays are public-selection residuals, not a "
            "paired citizen-public covariance calibration with per-row availability metadata"
        )
    if not inventories["public_evaluation"]["exists"]:
        blockers.append("frozen public evaluation artifact is absent")
    if not inventories["truth_evaluation"]["exists"]:
        blockers.append("exposed scoring truth artifact is absent")
    for key, item in exact_files.items():
        if not item["exists"]:
            blockers.append(f"required exact input absent: {key}")
    if manifest_error:
        blockers.append(f"producer provenance invalid: {manifest_error}")

    # Existing archive runners support negative_bias, not the registered severe_hotspot
    # endpoint. Never relabel it as the original hotspot mechanism.
    blockers.append(
        "existing archive replay has negative_bias, not the required severe_hotspot endpoint"
    )
    blockers.append(
        "stored archive outcome NPZ files contain predictions only; exact citizen records, "
        "arrival map, sensor/public error pairs, transition operators, and physical forcing "
        "cannot be reconstructed from them"
    )

    ready = not blockers
    return {
        "role": "readiness and provenance audit only; zero scientific outcomes",
        "registration": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "registration_sha256": sha256(CONFIG),
        "family_exact_2x2": family_exact,
        "locked_splits": {
            key: source[key]
            for key in ("public_fit_half_open", "mechanism_calibration_half_open", "exposed_evaluation_half_open")
        },
        "locked_seeds": cfg["locked_seeds"],
        "artifact_inventories": inventories,
        "required_exact_files": exact_files,
        "prediction_schema_error": schema_error,
        "producer_manifest_error": manifest_error,
        "readiness_pass": ready,
        "decision": "ready_to_run_exposed_development" if ready else "blocked_do_not_run",
        "blockers": blockers,
        "scientific_outcomes_read_or_generated": 0,
        "protected_namespaces_read": False,
        "runtime_estimate": None if not ready else "benchmark after input-bundle integrity preflight",
    }


def main() -> None:
    result = audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    readiness = OUTPUT / "readiness.json"
    readiness.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["readiness_pass"]:
        blocker = {
            "status": "blocked",
            "reason": "Exact causal input provenance is insufficient; no outcome runner is authorized.",
            "blockers": result["blockers"],
            "registration_sha256": result["registration_sha256"],
            "readiness_sha256": sha256(readiness),
            "scientific_outcomes": 0,
            "epa_validation_test_primary_confirmation_opened": False,
        }
        (OUTPUT / "blocker.json").write_text(
            json.dumps(blocker, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

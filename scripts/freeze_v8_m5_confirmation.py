#!/usr/bin/env python3
"""Create the immutable V8 M5 source/config lock before confirmation outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v8/m5_release_reference_stress_confirmation_v1.json"
OUTPUT = ROOT / "reports/v8/m5_release_reference_stress_confirmation_v1"

SOURCES = (
    "configs/v8/m5_release_reference_stress_confirmation_v1.json",
    "configs/v8/seed_registry.json",
    "configs/v8/m5_release_assimilation_development_v1.json",
    "reports/v8/m5_release_assimilation_development_v1/result.json",
    "configs/v7/reviewer_shared_resource_core_confirmation_v3.json",
    "configs/v7/seed_registry.json",
    "configs/v6/protocol.yaml",
    "reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml",
    "airproof/v8_release_assimilation.py",
    "airproof/v7_privacy_controls.py",
    "airproof/v7_shared_execution.py",
    "scripts/run_v8_m5_release_confirmation.py",
    "scripts/run_v7_shared_resource_confirmation.py",
    "airproof/v5_experiment.py",
    "airproof/continual_release.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute() -> dict:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    development = json.loads(
        (ROOT / registration["development_result"]).read_text(encoding="utf-8")
    )
    registry = json.loads((ROOT / registration["seed_registry"]).read_text(encoding="utf-8"))
    v7_registry = json.loads((ROOT / "configs/v7/seed_registry.json").read_text(encoding="utf-8"))
    v7_seeds = {seed for values in v7_registry["namespaces"].values() for seed in values}
    seeds = registration["seeds"]
    if (
        "frozen" not in registration["status"]
        or development["selected_candidate"] != registration["selected_estimator"]["name"]
        or seeds != registry["namespaces"][registration["seed_namespace"]]
        or len(seeds) != registration["worlds"]
        or len(set(seeds)) != len(seeds)
        or v7_seeds.intersection(seeds)
    ):
        raise ValueError("confirmation registration, development choice or seed isolation changed")
    paths = [ROOT / value for value in SOURCES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source lock inputs missing: {missing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outcome_files = list((OUTPUT / "jobs").rglob("result.json")) if (OUTPUT / "jobs").exists() else []
    if outcome_files:
        raise RuntimeError("cannot create a source lock after confirmation outcomes")
    inventory = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in paths
    }
    identity = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lock = {
        "schema_version": 1,
        "role": "immutable source/config/seed/gate lock created before independent outcomes",
        "outcomes_before_lock": 0,
        "registration_sha256": _sha(REGISTRATION),
        "source_identity": identity,
        "source_inventory": inventory,
    }
    target = OUTPUT / "source_lock.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != lock:
            raise ValueError("existing source lock differs")
    else:
        target.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))
    return lock


if __name__ == "__main__":
    execute()

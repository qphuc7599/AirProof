"""Build eight canonical PUBLIC-only uncertainty-v3.1 inputs; never runs an estimator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from airproof.config import config_hash, load_config
from airproof.simulator import generate_world
from airproof.v5_experiment import cell_configuration
from airproof.v6_numerical_experiment import prepare_public
from airproof.v6_uncertainty_public_prereq import (
    ARRAY_KEYS, array_digest, artifact_identity, canonical_hash,
    prove_historical_cell_invariance, require_unambiguous_publish_state,
    sha256, target_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs" / "v6_uncertainty_event_v3_1.json"
FINAL = ROOT / "reports" / "v6" / "uncertainty_v3_1_public_prerequisites_v1"
STAGING = FINAL.with_name(FINAL.name + ".staging")
HISTORICAL = ROOT / "reports" / "v6" / "development_v1"
BASE = ROOT / "reports" / "v4_primary" / "core_final_7600_7629_20260903" / "base_configuration.yaml"
PROTOCOL = ROOT / "configs" / "v6" / "protocol.yaml"
CELLS = ("anchor_clean", "outage_clean", "severe_clean", "severe_drift", "severe_hotspot")
PROOF_SEEDS = tuple(range(6201000, 6201008))
DEPENDENCIES = (
    "airproof/simulator.py", "airproof/field.py", "airproof/reference.py",
    "airproof/meteorology.py", "airproof/v6_numerical_experiment.py",
    "airproof/v6_calibration.py", "airproof/v5_experiment.py", "airproof/experiment.py",
    "airproof/rng.py", "airproof/records.py", "airproof/config.py",
)
ORCHESTRATION = (
    "configs/v6_uncertainty_event_v3_1.json",
    "airproof/v6_uncertainty_public_prereq.py",
    "scripts/build_v6_uncertainty_v3_1_public.py",
)


def source_proof():
    hashes, snapshot_hashes = {}, {}
    for relative in DEPENDENCIES:
        current = ROOT / relative
        snapshot = HISTORICAL / "source_snapshot" / relative
        hashes[relative] = sha256(current)
        snapshot_hashes[relative] = sha256(snapshot)
        if hashes[relative] != snapshot_hashes[relative]:
            raise RuntimeError(f"PUBLIC dependency changed since invariance evidence: {relative}")
    return {"current": hashes, "development_v1_snapshot": snapshot_hashes,
            "all_equal": True, "digest": canonical_hash(hashes)}


def registration():
    payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if payload.get("status") != "registered-v3.1-code-only-eight-canonical-PUBLIC-artifacts-missing":
        raise RuntimeError("unexpected v3.1 registration status")
    seeds = tuple(payload[role]["seeds"] for role in ("fit", "calibration", "evaluation"))
    flat = tuple(seed for group in seeds for seed in group)
    if flat != tuple(range(6202000, 6202008)) or payload["canonical_public_cell"] != "anchor_clean":
        raise RuntimeError("unexpected uncertainty-v3.1 seed/cell registration")
    return payload, flat


def preflight():
    payload, seeds = registration()
    sources = source_proof()
    invariance = prove_historical_cell_invariance(HISTORICAL, PROOF_SEEDS, CELLS)
    expected = [FINAL / str(seed) / "PUBLIC.npz" for seed in seeds]
    present = [path for path in expected if path.exists()]
    require_unambiguous_publish_state(expected, present, STAGING)
    return payload, seeds, sources, invariance, expected, present


def build():
    payload, seeds, sources, invariance, expected, present = preflight()
    if present:
        raise RuntimeError("complete final prerequisite set already exists; overwrite forbidden")
    STAGING.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        base = load_config(BASE)
        protocol = load_config(PROTOCOL)
        artifacts = []
        for seed in seeds:
            cfg = cell_configuration(base, "anchor_clean")
            cfg["transport"] = protocol["transport"] | {"drain_epochs": 24}
            world = generate_world(cfg, seed)
            public, _, scales, provenance = prepare_public(world, cfg)
            burn, steps = cfg["world"]["burn_in_steps"], len(world.truth)
            arrays = {"live": public[burn:steps], "reconstructed": public[burn:steps],
                      "truth": world.truth[burn:], "public": public[burn:steps],
                      "cell_groups": world.cell_groups, "scales": scales[burn:steps]}
            digest = array_digest(arrays)
            cell = "anchor_clean"
            cfg_hash = config_hash(cfg)
            identity = artifact_identity(seed=seed, cell="canonical-public",
                source_digest=sources["digest"], configuration_hash=cfg_hash,
                array_digest=digest)
            directory = STAGING / str(seed)
            directory.mkdir()
            path = directory / "PUBLIC.npz"
            np.savez_compressed(path, **arrays, identity=identity,
                source_hash=sources["digest"], seed=seed, cell=cell,
                config_hash=cfg_hash, array_digest=digest)
            record = {"schema_version": 1, "role": "uncertainty-v3.1-public-prerequisite",
                "seed": seed, "cell": cell, "identity": identity,
                "canonical_computation_cell": "anchor_clean",
                "configuration_hash": cfg_hash, "source_digest": sources["digest"],
                "array_digest": digest, "artifact_sha256": sha256(path),
                "public_provenance": provenance,
                "physical_cell_reuse_basis": "five-way equality is provenance only; this artifact is scored once",
                "estimator_run": False, "transport_run": False,
                "confirmation_read": False, "epa_read": False}
            (directory / "PUBLIC.json").write_text(json.dumps(
                record, indent=2,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value)
            ) + "\n")
            artifacts.append(record)
        manifest = {"schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "registration_sha256": sha256(REGISTRATION),
            "base_configuration_sha256": sha256(BASE),
            "protocol_sha256": sha256(PROTOCOL), "sources": sources,
            "orchestration_hashes": {relative: sha256(ROOT / relative)
                                      for relative in ORCHESTRATION},
            "historical_physical_cell_invariance": invariance,
            "matrix": [{"seed": s, "cell": "anchor_clean", "scored_once": True} for s in seeds],
            "expected": 8, "completed": len(artifacts), "artifacts": artifacts,
            "runtime_seconds": time.perf_counter() - started,
            "scope": "PUBLIC/truth prerequisite only; no estimator, EPA test or confirmation read"}
        (STAGING / "manifest.json").write_text(json.dumps(
            manifest, indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value)
        ) + "\n")
        if len(artifacts) != 8 or any(not path.exists() for path in
                [STAGING / str(s) / "PUBLIC.npz" for s in seeds]):
            raise RuntimeError("staging completeness check failed")
        os.replace(STAGING, FINAL)
    except Exception:
        # Preserve staging for diagnosis.  A future invocation refuses to proceed
        # until a human resolves it; no partial final directory is published.
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
        help="generate and atomically publish the complete prerequisite directory")
    args = parser.parse_args()
    payload, seeds, sources, invariance, expected, present = preflight()
    summary = {"mode": "execute" if args.execute else "preflight",
        "seeds": list(seeds), "canonical_cell": "anchor_clean", "expected": len(expected),
        "present": len(present), "source_snapshot_equal": sources["all_equal"],
        "historical_physical_cell_arrays_equal": invariance["exact_equal"],
        "estimator_run": False, "epa_read": False, "confirmation_read": False}
    print(json.dumps(summary, indent=2))
    if args.execute:
        build()


if __name__ == "__main__":
    main()

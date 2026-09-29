"""Build registered five-cell validation inputs without estimator outcomes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_lifetime_validation_inputs import (
    _expected_payloads,
    build_validation_seed,
    finalize_bundle,
    load_registration,
    verify_bundle,
)


def _task(root: str, stage: str, seed: int):
    return build_validation_seed(root, stage, seed)


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale atomic-write temporary exists: {temporary}")
    write_json(temporary, value)
    os.replace(temporary, path)


def _verify_source_lock(root: Path, registration: dict, source_lock: Path) -> dict:
    lock = json.loads(source_lock.read_text(encoding="utf-8"))
    registration_path = root / "configs/v6/causal_lifetime_exposure_validation_v1.json"
    if (lock.get("registration_sha256") != sha256_file(registration_path)
            or lock.get("validation_outcomes_before_lock") != 0
            or not isinstance(lock.get("source_sha256"), dict)
            or not lock["source_sha256"]):
        raise RuntimeError("validation source lock schema or zero-outcome claim invalid")
    mismatches = [
        relative for relative, expected in lock["source_sha256"].items()
        if not (root / relative).is_file() or sha256_file(root / relative) != expected
    ]
    required_sources = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in (root / "airproof").glob("*.py")
    }
    required_sources.update({
        "scripts/build_v6_lifetime_validation_v1_inputs.py",
        "scripts/audit_v6_lifetime_validation_v1_readiness.py",
        "scripts/analyze_v6_lifetime_validation_v1.py",
        "configs/v6/causal_lifetime_exposure_validation_v1.json",
        registration["worlds"]["seed_registry"],
        registration["nomination"]["development_registration"],
        registration["nomination"]["development_analysis"],
        registration["historical_reuse"]["v4_reuse_lock"],
        registration["fixed_estimator"]["innovation_calibration"],
        registration["executable_sources"]["base_configuration"],
        registration["executable_sources"]["protocol"],
        "reports/v6/innovation_covariance_forcing_development_v2/input_lock.json",
    })
    missing_inventory = required_sources - set(lock["source_sha256"])
    if missing_inventory:
        raise RuntimeError(
            f"validation source lock omits required sources: {sorted(missing_inventory)}"
        )
    checks = {
        "development_analysis_sha256": registration["nomination"]["development_analysis"],
        "innovation_calibration_sha256": registration["fixed_estimator"][
            "innovation_calibration"],
        "seed_registry_sha256": registration["worlds"]["seed_registry"],
    }
    for field, relative in checks.items():
        if lock.get(field) != sha256_file(root / relative):
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"validation source lock mismatch: {sorted(set(mismatches))}")
    return lock


def _seed_payloads(registration: dict, seed: int) -> set[str]:
    prefix = f"prediction/{seed}"
    score = f"scoring/{seed}"
    expected = {f"{prefix}/summary.json"}
    expected.update(f"{prefix}/observations_{cell}.npz"
                    for cell in registration["worlds"]["cells"])
    expected.update(f"{prefix}/context_{cell}.npz"
                    for cell in registration["worlds"]["clean_cells"])
    expected.update(f"{score}/scoring_{cell}.npz"
                    for cell in registration["worlds"]["clean_cells"])
    return expected


def _seed_complete(stage: Path, registration: dict, seed: int) -> bool:
    expected = _seed_payloads(registration, seed)
    actual = {
        str(path.relative_to(stage)).replace("\\", "/")
        for parent in (stage / "prediction" / str(seed), stage / "scoring" / str(seed))
        if parent.exists() for path in parent.rglob("*") if path.is_file()
    }
    if not actual:
        return False
    if actual != expected:
        raise RuntimeError(
            f"stale or partial validation seed staging payload: {seed}"
        )
    return True


def _input_lock_value(root: Path, registration: dict, source_lock: Path,
                      manifest: dict, manifest_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "role": "immutable validation input lock; zero estimator outcomes",
        "registration_sha256": sha256_file(
            root / "configs/v6/causal_lifetime_exposure_validation_v1.json"),
        "source_lock_sha256": sha256_file(source_lock),
        "manifest_sha256": manifest_sha256,
        "payload_root_sha256": manifest["payload_root_sha256"],
        "scientific_outcomes_before_lock": 0,
        "protected_namespaces_opened": False,
    }


def _recover_publish(root: Path, registration: dict, bundle: Path, stage: Path,
                     source_lock: Path, input_lock_path: Path,
                     transaction_path: Path) -> dict | None:
    staged_lock = input_lock_path.with_name(input_lock_path.name + ".building")
    transaction_temporary = transaction_path.with_name(transaction_path.name + ".tmp")
    if not transaction_path.exists():
        if not staged_lock.exists():
            if transaction_temporary.exists():
                raise RuntimeError("orphaned validation transaction temporary")
            return None
        if not stage.exists() or bundle.exists() or not (stage / "manifest.json").is_file():
            raise RuntimeError("orphaned validation staged input lock")
        staged_value = json.loads(staged_lock.read_text(encoding="utf-8"))
        manifest_sha = staged_value.get("manifest_sha256")
        manifest = verify_bundle(root, stage, expected_manifest_sha256=manifest_sha)
        expected_lock = _input_lock_value(
            root, registration, source_lock, manifest, manifest_sha)
        if staged_value != expected_lock:
            raise RuntimeError("orphaned validation staged input lock is stale")
        transaction_value = {
            "schema_version": 1,
            "registration_sha256": expected_lock["registration_sha256"],
            "source_lock_sha256": expected_lock["source_lock_sha256"],
            "manifest_sha256": manifest_sha,
            "payload_root_sha256": manifest["payload_root_sha256"],
        }
        if transaction_temporary.exists():
            if json.loads(transaction_temporary.read_text(encoding="utf-8")) != transaction_value:
                raise RuntimeError("validation transaction temporary is stale")
            os.replace(transaction_temporary, transaction_path)
        else:
            _atomic_write_json(transaction_path, transaction_value)
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    if (transaction.get("schema_version") != 1
            or transaction.get("registration_sha256") != sha256_file(
                root / "configs/v6/causal_lifetime_exposure_validation_v1.json")
            or transaction.get("source_lock_sha256") != sha256_file(source_lock)):
        raise RuntimeError("stale validation publish transaction")
    manifest_sha = transaction["manifest_sha256"]
    if bundle.exists() and stage.exists():
        raise RuntimeError("validation publish has both staged and final bundles")
    target = bundle if bundle.exists() else stage
    if not target.exists():
        raise RuntimeError("validation publish transaction lost its bundle")
    manifest = verify_bundle(root, target, expected_manifest_sha256=manifest_sha)
    expected_lock = _input_lock_value(root, registration, source_lock, manifest, manifest_sha)
    if input_lock_path.exists():
        if json.loads(input_lock_path.read_text(encoding="utf-8")) != expected_lock:
            raise RuntimeError("validation publish input lock mismatch")
    else:
        if (not staged_lock.exists()
                or json.loads(staged_lock.read_text(encoding="utf-8")) != expected_lock):
            raise RuntimeError("validation publish staged input lock missing or stale")
        os.replace(staged_lock, input_lock_path)
    if not bundle.exists():
        stage.rename(bundle)
    verify_bundle(root, bundle, expected_manifest_sha256=manifest_sha)
    transaction_path.unlink()
    if staged_lock.exists():
        staged_lock.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    registration = load_registration(ROOT)
    bundle = ROOT / registration["input_producer"]["bundle"]
    source_lock = ROOT / registration["input_producer"]["source_lock"]
    input_lock_path = ROOT / registration["input_producer"]["input_lock"]
    stage = Path(str(bundle) + ".building")
    transaction_path = input_lock_path.with_name("input_publish_transaction.json")
    if not source_lock.is_file():
        raise SystemExit("validation source lock required before input generation")
    _verify_source_lock(ROOT, registration, source_lock)
    recovered = _recover_publish(
        ROOT, registration, bundle, stage, source_lock, input_lock_path,
        transaction_path,
    )
    if recovered is not None:
        print(json.dumps({"recovered": True,
                          "payloads": len(recovered["payload_sha256"]),
                          "manifest_sha256": sha256_file(bundle / "manifest.json"),
                          "input_lock_sha256": sha256_file(input_lock_path),
                          "scientific_outcomes": 0}, indent=2))
        return
    if args.verify:
        input_lock = json.loads(input_lock_path.read_text(encoding="utf-8"))
        manifest = verify_bundle(
            ROOT, bundle, expected_manifest_sha256=input_lock["manifest_sha256"])
        expected_lock = _input_lock_value(
            ROOT, registration, source_lock, manifest, input_lock["manifest_sha256"])
        if input_lock != expected_lock:
            raise SystemExit("validation input lock semantics mismatch")
        print(json.dumps({"verified": True, "payloads": len(manifest["payload_sha256"]),
                          "manifest_sha256": sha256_file(bundle / "manifest.json")},
                         indent=2))
        return
    if bundle.exists():
        raise SystemExit("validation input namespace exists; use --verify")
    if input_lock_path.exists():
        raise SystemExit("validation input lock exists without a publish transaction")
    outcomes = ROOT / registration["artifacts"]["outcomes"]
    if outcomes.exists() and any(outcomes.rglob("*")):
        raise SystemExit("validation outcome namespace is nonempty before input lock")
    if not 1 <= args.workers <= registration["runtime"]["workers"]:
        raise SystemExit("workers outside registered range")
    if not stage.exists():
        stage.mkdir(parents=True)
        write_json(stage / "build_state.json", {
            "registration_sha256": sha256_file(ROOT / "configs/v6/causal_lifetime_exposure_validation_v1.json"),
            "source_lock_sha256": sha256_file(source_lock),
            "scientific_outcomes": 0,
        })
    state = json.loads((stage / "build_state.json").read_text(encoding="utf-8"))
    if (state["registration_sha256"]
            != sha256_file(ROOT / "configs/v6/causal_lifetime_exposure_validation_v1.json")
            or state["source_lock_sha256"] != sha256_file(source_lock)):
        raise SystemExit("stale validation input staging state")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    pending = [int(seed) for seed in registration["worlds"]["seeds"]
               if not _seed_complete(stage, registration, int(seed))]
    completed = 0
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_task, str(ROOT), str(stage), seed)
                       for seed in pending]
            for future in as_completed(futures):
                summary = future.result()
                completed += 1
                write_json(stage / "progress.json", {
                    "newly_completed": completed, "submitted": len(pending),
                    "last_seed": summary["seed"], "scientific_outcomes": 0,
                })
    summaries = list((stage / "prediction").glob("*/summary.json"))
    if (len(summaries) != len(registration["worlds"]["seeds"])
            or not all(_seed_complete(stage, registration, int(seed))
                       for seed in registration["worlds"]["seeds"])):
        raise RuntimeError("validation input matrix incomplete")
    # Exclude transient state and progress from the immutable payload inventory.
    (stage / "build_state.json").unlink()
    progress = stage / "progress.json"
    if progress.exists():
        progress.unlink()
    manifest = finalize_bundle(ROOT, stage, sha256_file(source_lock))
    manifest_sha = sha256_file(stage / "manifest.json")
    verified = verify_bundle(ROOT, stage, expected_manifest_sha256=manifest_sha)
    if set(verified["payload_sha256"]) != _expected_payloads(registration):
        raise RuntimeError("validation final payload inventory changed")
    lock_value = _input_lock_value(
        ROOT, registration, source_lock, verified, manifest_sha)
    staged_lock = input_lock_path.with_name(input_lock_path.name + ".building")
    if staged_lock.exists() or transaction_path.exists():
        raise SystemExit("stale validation publish state exists")
    _atomic_write_json(staged_lock, lock_value)
    _atomic_write_json(transaction_path, {
        "schema_version": 1,
        "registration_sha256": lock_value["registration_sha256"],
        "source_lock_sha256": lock_value["source_lock_sha256"],
        "manifest_sha256": manifest_sha,
        "payload_root_sha256": verified["payload_root_sha256"],
    })
    os.replace(staged_lock, input_lock_path)
    stage.rename(bundle)
    verify_bundle(ROOT, bundle, expected_manifest_sha256=manifest_sha)
    transaction_path.unlink()
    print(json.dumps({
        "completed": True, "worlds": len(summaries),
        "payloads": len(manifest["payload_sha256"]),
        "manifest_sha256": manifest_sha,
        "input_lock_sha256": sha256_file(input_lock_path),
        "scientific_outcomes": 0,
    }, indent=2))


if __name__ == "__main__":
    main()

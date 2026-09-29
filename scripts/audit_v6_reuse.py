"""Verify retained v4 artifacts against a frozen hash lock; never rerun them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASH_LOCK = ROOT / "configs/v6/v4_reuse_hash_lock.json"
OUTPUT = ROOT / "reports/v6/reuse_verified_manifest.json"

ITEMS = [
    ("v4_primary", "reports/v4_primary/core_final_7600_7629_20260903",
     "historical complete primary under v4 execution; not v6 confirmation"),
    ("beijing_transfer", "reports/v4_validation/beijing_seasonal_20260903",
     "unchanged v4 estimator seven-month transfer"),
    ("dtn_confirmation", "reports/v4_validation/dtn_deadline_confirm_7950_7979",
     "standalone v4 routing contract"),
    ("concurrency", "reports/v4_validation/ingestion_concurrency_20260903",
     "unchanged atomic ingestion implementation and tested process model"),
    ("sepolia", "reports/v4_validation/sepolia_30_anchors_20260903",
     "historical public-network anchors; no new transactions"),
    ("backbone", "reports/v4_validation/esntnn_purpleair_7800",
     "saved weights/predictions and original archive exposure; no retraining"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lock = json.loads(HASH_LOCK.read_text(encoding="utf-8"))
    expected = lock["files"]
    rows = []
    all_missing: list[str] = []
    all_mismatches: list[str] = []
    seen: set[str] = set()
    for identity, relative, scope in ITEMS:
        folder = ROOT / relative
        actual: dict[str, str] = {}
        missing: list[str] = []
        mismatches: list[str] = []
        group_expected = sorted(path for path in expected
                                if path == relative or path.startswith(f"{relative}/"))
        for name in group_expected:
            path = ROOT / name
            seen.add(name)
            if not path.is_file():
                missing.append(name)
                continue
            digest = sha256_file(path)
            actual[name] = digest
            if digest != expected[name]:
                mismatches.append(name)
        all_missing.extend(missing)
        all_mismatches.extend(mismatches)
        rows.append({
            "id": identity,
            "directory": relative,
            "scope": scope,
            "decision": "retain; no reproduction run",
            "files": actual,
            "expected_sha256": {name: expected[name] for name in group_expected},
            "verification": {
                "files_checked": len(actual),
                "missing": missing,
                "hash_mismatches": mismatches,
                "pass": not missing and not mismatches and folder.is_dir(),
            },
            "v6_changed_mechanism_confirmation": False,
        })
    unassigned = sorted(set(expected) - seen)
    preserved = json.loads((ROOT / "reports/v6/bootstrap/preserved_manifest.json")
                           .read_text(encoding="utf-8"))
    checked = {}
    for relative, digest in preserved.items():
        path = ROOT / relative
        if path.suffix == ".pdf" or (relative.startswith("airproof")
                                     and "v6_" not in relative):
            actual = sha256_file(path) if path.is_file() else None
            checked[relative] = {
                "expected_sha256": digest,
                "actual_sha256": actual,
                "unchanged_since_bootstrap": actual == digest,
            }
    current_source_changes = sorted(name for name, value in checked.items()
                                    if not value["unchanged_since_bootstrap"])
    passed = not all_missing and not all_mismatches and not unassigned
    result = {
        "schema_version": 2,
        "policy": "docs/V6_V4_REUSE_AND_CLAIM_POLICY.md",
        "hash_lock": str(HASH_LOCK.relative_to(ROOT)).replace("\\", "/"),
        "hash_lock_sha256": sha256_file(HASH_LOCK),
        "artifacts": rows,
        "artifact_verification": {
            "pass": passed,
            "files_expected": len(expected),
            "files_checked": sum(len(row["files"]) for row in rows),
            "missing": sorted(all_missing),
            "hash_mismatches": sorted(all_mismatches),
            "unassigned_lock_entries": unassigned,
        },
        "preservation_checks": checked,
        "current_source_changes_since_v6_bootstrap": current_source_changes,
        "historical_source_rule": (
            "Current source changes do not mutate or reconfirm v4. Historical claims remain "
            "bound to each retained run's source snapshot, runner, configuration and result hashes."
        ),
        "limits": (
            "Artifact integrity and declared scope audit, not a new statistical validation. "
            "Theorems retain their original assumptions."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "groups": len(rows),
        "expected_files": len(expected),
        "checked_files": result["artifact_verification"]["files_checked"],
        "artifact_verification_pass": passed,
        "missing": all_missing,
        "hash_mismatches": all_mismatches,
        "current_source_changes": current_source_changes,
    }, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

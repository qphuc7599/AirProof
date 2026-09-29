"""Materialize small registered source/config dependencies without copying data."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "epa_2025_rolling_controls", "epa_2025_public_tail_diagnostic",
    "epa_2025_public_rolling_hgb", "epa_2025_public_spatial",
    "epa_2025_public_spatiotemporal", "epa_2025_public_spatiotemporal_hgb",
    "epa_2025_public_peer_vector", "epa_2025_spatiotemporal_controls",
    "epa_2025_projected_output_cap", "epa_2025_public_rich_forest",
)
SMALL_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    aggregate = []
    for name in ARTIFACTS:
        artifact = ROOT / "reports" / "v6" / name
        registration_path = artifact / "registration.json"
        if not registration_path.exists():
            aggregate.append({"artifact": name, "status": "missing_registration"})
            continue
        registration = json.loads(registration_path.read_text())
        entries = []
        for raw, expected in registration.get("hashes", {}).items():
            source = Path(raw)
            if not source.is_absolute():
                source = ROOT / source
            try:
                relative = source.relative_to(ROOT)
            except ValueError:
                entries.append({"path": raw, "status": "external_path", "sha256": expected})
                continue
            if source.suffix.lower() not in SMALL_SUFFIXES:
                entries.append({"path": relative.as_posix(), "status": "hash_only_large_dependency",
                                "sha256": expected})
                continue
            target = artifact / "source_snapshot" / relative
            if target.exists() and sha(target) == expected:
                entries.append({"path": relative.as_posix(), "status": "existing_exact_snapshot",
                                "sha256": expected})
                continue
            if not source.exists() or sha(source) != expected:
                entries.append({"path": relative.as_posix(), "status": "working_source_changed_no_exact_snapshot",
                                "sha256": expected})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entries.append({"path": relative.as_posix(), "status": "copied_exact_snapshot",
                            "sha256": expected})
        manifest = {"artifact": name, "registration_sha256": sha(registration_path),
                    "entries": entries,
                    "complete_for_small_registered_dependencies": all(
                        item["status"] not in {"missing_registration", "working_source_changed_no_exact_snapshot"}
                        for item in entries)}
        (artifact / "source_snapshot_manifest.json").write_text(json.dumps(manifest, indent=2))
        aggregate.append(manifest)
    output = ROOT / "reports" / "v6" / "epa_2025_registered_source_snapshots.json"
    output.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps({"artifacts": len(aggregate),
                      "complete": sum(bool(item.get("complete_for_small_registered_dependencies"))
                                      for item in aggregate)}, indent=2))


if __name__ == "__main__":
    main()

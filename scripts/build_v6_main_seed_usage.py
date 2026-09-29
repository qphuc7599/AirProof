"""Audit execution use of the prospectively registered v6 world namespaces."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "v6" / "seed_registry.json"
OUTPUT = ROOT / "reports" / "v6" / "main_seed_usage_audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    registry = load(REGISTRY)
    namespaces = {name: set(map(int, values)) for name, values in registry["namespaces"].items()}

    development_matrix = ROOT / "reports" / "v6" / "development_v1" / "complete_matrix.json"
    development = load(development_matrix)
    used_development = {
        int(row["seed"]) for row in development.get("jobs", []) if row.get("status") == "complete"
    }

    fairness_matrix = ROOT / "reports" / "v6" / "fairness_validation_v1" / "complete_matrix.json"
    fairness = load(fairness_matrix)
    fairness_worlds = ROOT / "reports" / "v6" / "fairness_validation_v1" / "worlds"
    used_validation = {
        int(path.parent.name)
        for path in fairness_worlds.glob("*/result.json")
        if load(path).get("status") == "complete"
    }

    prerequisite = ROOT / "reports" / "v6" / "uncertainty_v3_public_prerequisites_v1"
    prerequisite_v31 = ROOT / "reports" / "v6" / "uncertainty_v3_1_public_prerequisites_v1"
    public_counts = {
        seed: (len(list(prerequisite.glob(f"{seed}_*/PUBLIC.npz")))
               + int((prerequisite_v31 / str(seed) / "PUBLIC.npz").exists()))
        for seed in sorted(namespaces["calibration"])
    }
    used_calibration = {seed for seed, count in public_counts.items() if count > 0}
    uncertainty_v31 = ROOT / "reports" / "v6" / "uncertainty_event_v3_1" / "manifest.json"
    uncertainty_v31_complete = uncertainty_v31.exists() and load(uncertainty_v31).get("evaluation_residual_updates") == 0

    expected_roles = {
        "development": used_development,
        "calibration": used_calibration,
        "validation": used_validation,
        "confirmation_reserved": set(),
        "descriptive_stress": set(),
    }
    cross_namespace = []
    owner = {}
    for role, values in namespaces.items():
        for value in values:
            if value in owner:
                cross_namespace.append({"seed": value, "roles": [owner[value], role]})
            owner[value] = role

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registry": REGISTRY.relative_to(ROOT).as_posix(),
        "registry_sha256": sha(REGISTRY),
        "nature": "post-execution usage audit; the registry remains the prospective allocation record",
        "namespace_overlap": cross_namespace,
        "roles": {},
        "auxiliary_rng": {
            "uncertainty_bootstrap_6607001": "consumed by adverse v2 development evaluation",
            "uncertainty_bootstrap_6607002": (
                "consumed by adverse post-v2 v3.1 exposed-development evaluation"
                if uncertainty_v31_complete else
                "registered for v3.1; unread until a source-bound result exists"
            ),
            "engineering_fixture_6006001": "exposed and reused only for deterministic engineering smokes",
        },
        "rules": [
            "A seed used by any component or diagnostic is exposed for that world namespace.",
            "Fairness validation exposure prevents 6203000..6203011 from later serving as independent estimator validation.",
            "PUBLIC-only generation consumes a world seed even when citizen estimation is skipped.",
            "Confirmation seeds remain unread; no primary or confirmation campaign is authorized.",
        ],
        "input_hashes": {},
    }
    for role, registered in namespaces.items():
        used = expected_roles[role]
        out["roles"][role] = {
            "registered": sorted(registered),
            "consumed": sorted(used),
            "unread": sorted(registered - used),
            "outside_registered_namespace": sorted(used - registered),
        }
    for path in (development_matrix, fairness_matrix, uncertainty_v31):
        if path.exists():
            out["input_hashes"][path.relative_to(ROOT).as_posix()] = sha(path)
    out["calibration_public_artifact_counts"] = public_counts
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "development_consumed": len(used_development),
        "calibration_consumed": len(used_calibration),
        "validation_consumed": len(used_validation),
        "confirmation_consumed": 0,
        "namespace_overlap": len(cross_namespace),
    }, indent=2))


if __name__ == "__main__":
    main()

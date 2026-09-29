"""Build the post-execution EPA auxiliary seed-use audit.

This script does not alter, relabel, or rerun an evaluated artifact.  It records
which registered namespaces were actually read and reports integer-token reuse
against the v5 registry, even when the two executions use different generators.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "v6"
V5_REGISTRY = ROOT / "configs" / "v5" / "seed_usage.json"
V6_REGISTRY = ROOT / "configs" / "v6" / "seed_registry.json"
OUTPUT = ROOT / "configs" / "v6" / "epa_seed_usage.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(entry: dict[str, Any]) -> set[int]:
    values = {int(x) for x in entry.get("seeds", [])}
    if "range" in entry:
        lo, hi = (int(x) for x in entry["range"])
        values.update(range(lo, hi + 1))
    return values


def historical_entries(registry: dict[str, Any]) -> Iterable[tuple[str, str, set[int]]]:
    for state in ("consumed", "active", "reserved"):
        for row in registry.get(state, []):
            yield state, str(row.get("generator", "unknown")), expand(row)


def registered_seeds(registration: dict[str, Any]) -> tuple[set[int], set[int]]:
    development: set[int] = set()
    validation: set[int] = set()
    for key, value in registration.items():
        if not isinstance(value, (list, int)) or "seed" not in key.lower():
            continue
        numbers = {int(value)} if isinstance(value, int) else {int(x) for x in value}
        if "validation" in key.lower():
            validation.update(numbers)
        else:
            development.update(numbers)
    return development, validation


def positive_rows(result: dict[str, Any], key: str) -> bool:
    value = result.get(key, 0)
    if isinstance(value, list):
        return len(value) > 0
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def main() -> None:
    v5 = json.loads(V5_REGISTRY.read_text(encoding="utf-8"))
    v6 = json.loads(V6_REGISTRY.read_text(encoding="utf-8"))
    historical = list(historical_entries(v5))
    main_v6 = {
        int(seed)
        for seeds in v6.get("namespaces", {}).values()
        for seed in seeds
    }

    artifacts: list[dict[str, Any]] = []
    consumed: set[int] = set()
    reserved_unread: set[int] = set()
    intersections: list[dict[str, Any]] = []

    for registration_path in sorted(REPORT_ROOT.glob("epa_*/registration.json")):
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        development, validation = registered_seeds(registration)
        if not development and not validation:
            continue
        result_path = registration_path.with_name("result.json")
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        development_read = result_path.exists() and (
            positive_rows(result, "development_rows")
            or positive_rows(result, "rows")
            or registration_path.parent.name.endswith(("rolling_controls", "spatiotemporal_controls", "projected_output_cap"))
        )
        validation_read = result_path.exists() and positive_rows(result, "validation_rows")
        if development_read:
            consumed.update(development)
        if validation_read:
            consumed.update(validation)
        else:
            reserved_unread.update(validation)

        row = {
            "artifact": registration_path.parent.name,
            "registration": registration_path.relative_to(ROOT).as_posix(),
            "registration_sha256": sha256(registration_path),
            "result": result_path.relative_to(ROOT).as_posix() if result_path.exists() else None,
            "result_sha256": sha256(result_path) if result_path.exists() else None,
            "development_seeds": sorted(development),
            "validation_seeds": sorted(validation),
            "development_read": development_read,
            "validation_read": validation_read,
        }
        artifacts.append(row)

        for seed in sorted(development | validation):
            for state, generator, old_seeds in historical:
                if seed in old_seeds:
                    intersections.append(
                        {
                            "seed": seed,
                            "epa_artifact": registration_path.parent.name,
                            "epa_role": "validation" if seed in validation else "development_or_diagnostic",
                            "v5_state": state,
                            "v5_generator": generator,
                            "same_generator": generator == "EPA synthetic citizen channel",
                        }
                    )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Post-execution audit of EPA auxiliary development seeds; not a prospective registration.",
        "rules": {
            "development_result_present": "A positive result row count marks registered development seeds as read.",
            "validation_result_present": "Only positive validation_rows marks validation seeds as read.",
            "failed_scientific_results": "Seeds are not replaced and failed families are not rerun under new seeds.",
            "freshness": "A new citizen-channel seed does not create a new pollution history.",
        },
        "consumed_development_or_diagnostic": sorted(consumed),
        "reserved_validation_unread": sorted(reserved_unread - consumed),
        "intersection_with_v6_main_namespace": sorted(consumed & main_v6),
        "v5_integer_token_intersections": intersections,
        "interpretation": {
            "main_v6_collision": len(consumed & main_v6) > 0,
            "historical_integer_reuse_present": len(intersections) > 0,
            "consequence": (
                "Historical integer reuse is a provenance defect even where generators differ; "
                "the affected EPA artifacts remain exposed development evidence and cannot be relabeled as confirmation."
            ),
        },
        "artifacts": artifacts,
        "input_hashes": {
            V5_REGISTRY.relative_to(ROOT).as_posix(): sha256(V5_REGISTRY),
            V6_REGISTRY.relative_to(ROOT).as_posix(): sha256(V6_REGISTRY),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "artifacts": len(artifacts),
        "consumed": len(consumed),
        "reserved_validation_unread": len(reserved_unread - consumed),
        "v5_integer_intersections": len(intersections),
        "v6_main_intersections": len(consumed & main_v6),
    }, indent=2))


if __name__ == "__main__":
    main()

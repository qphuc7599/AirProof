from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_v6_lifetime_validation_v1 import collect

from airproof.records import canonical_json
from airproof.v6_covariance_forcing_inputs import sha256_file
from airproof.v6_lifetime_validation_registration import evaluate
from airproof.v6_lifetime_validation_runner import (
    REGISTRATION,
    load_registration,
)

OUTPUT = ROOT / "reports/v6/causal_lifetime_exposure_validation_v1/outcomes/analysis.json"
FAILURE = (
    ROOT
    / "reports/v6/causal_lifetime_exposure_validation_v1/outcomes/analysis_serialization_failure.json"
)
FROZEN_ANALYZER = ROOT / "scripts/analyze_v6_lifetime_validation_v1.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sanitize(value: Any, path: str = "") -> tuple[Any, list[dict[str, str]]]:
    replacements: list[dict[str, str]] = []
    if isinstance(value, float) and not math.isfinite(value):
        semantic = "negative_infinity" if value < 0 else "positive_infinity"
        return None, [{"path": path, "semantic_value": semantic}]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            child, changed = _sanitize(item, f"{path}.{key}" if path else str(key))
            result[key] = child
            replacements.extend(changed)
        return result, replacements
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            child, changed = _sanitize(item, f"{path}[{index}]")
            result.append(child)
            replacements.extend(changed)
        return result, replacements
    return value, replacements


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("analysis already exists; recovery overwrite prohibited")
    registration = load_registration(ROOT)
    rows, matrix = collect(registration, smoke=False)
    if matrix.get("pass") is not True or matrix.get("complete_rows") != 240:
        raise RuntimeError("cannot recover an incomplete or failed matrix audit")
    gate = evaluate(registration, rows)
    clean_gate, replacements = _sanitize(gate)
    expected = [{
        "path": "tests.attack:severe_drift.lower",
        "semantic_value": "negative_infinity",
    }]
    if replacements != expected:
        raise RuntimeError(f"unexpected nonfinite gate fields: {replacements}")
    drift = gate["tests"]["attack:severe_drift"]
    if (
        drift["pass"] is not False
        or drift["positive_denominator_draws"] != 19999
        or gate["pass"] is not False
        or gate["all_invariants_pass"] is not True
    ):
        raise RuntimeError("serialization recovery would change the scientific decision")

    source_lock = ROOT / registration["input_producer"]["source_lock"]
    lock = json.loads(source_lock.read_text(encoding="utf-8"))
    analyzer_relative = FROZEN_ANALYZER.relative_to(ROOT).as_posix()
    if lock["source_sha256"].get(analyzer_relative) != _sha256(FROZEN_ANALYZER):
        raise RuntimeError("failed analyzer does not match the validation source lock")
    recovery = {
        "schema_version": 1,
        "role": "strict-JSON serialization recovery; no metric or gate recomputation change",
        "frozen_analyzer": analyzer_relative,
        "frozen_analyzer_sha256": _sha256(FROZEN_ANALYZER),
        "source_lock_sha256": _sha256(source_lock),
        "original_exception": (
            "ValueError: Out of range float values are not JSON compliant: -inf"
        ),
        "semantic_replacements": replacements,
        "decision_preserved": {
            "gate_pass": False,
            "all_invariants_pass": True,
            "failed_gate": "attack:severe_drift",
            "positive_denominator_draws": 19999,
            "bootstrap_draws": 20000,
        },
    }
    FAILURE.parent.mkdir(parents=True, exist_ok=True)
    FAILURE.write_bytes(canonical_json(recovery) + b"\n")
    result = {
        "schema_version": 1,
        "scope": "independent 12-world five-cell synthetic mechanism validation",
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "analysis_script_sha256": _sha256(FROZEN_ANALYZER),
        "recovery_script_sha256": _sha256(Path(__file__)),
        "matrix_audit": matrix,
        "gate_evaluation": clean_gate,
        "serialization_recovery": {
            **recovery,
            "artifact": FAILURE.relative_to(ROOT).as_posix(),
            "artifact_sha256": _sha256(FAILURE),
        },
        "historical_v4_rerun": False,
        "epa_test_primary_confirmation_opened": False,
    }
    OUTPUT.write_bytes(canonical_json(result) + b"\n")
    print(json.dumps({
        "artifact": OUTPUT.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha256(OUTPUT),
        "matrix_rows": matrix["complete_rows"],
        "gate_pass": clean_gate["pass"],
        "failed_gate": "attack:severe_drift",
        "semantic_replacements": replacements,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Outcome-blind readiness audit for causal lifetime-exposure v4."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_lifetime_registration import assert_registration
from airproof.v6_lifetime_runner import (
    REGISTRATION,
    load_registration,
    verify_source_lock,
)


def audit() -> dict:
    registration = load_registration(ROOT)
    checks = {}

    def check(name, operation):
        try:
            checks[name] = {"pass": True, "detail": operation()}
        except Exception as exc:  # noqa: BLE001 - persist every readiness failure
            checks[name] = {"pass": False, "detail": f"{type(exc).__name__}: {exc}"}

    check("registration", lambda: (assert_registration(registration),
          "lifetime budgets7/14/28, cap8, delta1.345, lag6, original gates")[1])
    check("source_lock", lambda: {
        "sha256": sha256_file(ROOT / registration["artifacts"]["source_lock"]),
        "reused_files": len(verify_source_lock(ROOT, registration)["reused_control_sha256"]),
    })

    def v3_disposition():
        value = json.loads((ROOT / registration["development_history"]["v3_analysis"])
                           .read_text(encoding="utf-8"))
        gate = value["gate_evaluation"]
        if (gate["family_closed"] is not True or gate["selected_candidate"] is not None
                or gate["eligible"]):
            raise ValueError("v3 is not a closed no-nomination family")
        return "v3 closed; all three fixed-window budgets failed drift"
    check("v3_disposition", v3_disposition)

    def diagnosis():
        value = json.loads((ROOT / registration["development_history"]["v3_diagnosis"])
                           .read_text(encoding="utf-8"))
        if (value["family_extension_authorized"] is not False
                or value["validation_epa_test_primary_confirmation_opened"] is not False
                or value["gate_evaluation"]["family_closed"] is not True):
            raise ValueError("v3 diagnosis is not closed and protected")
        return value["lifetime_exposure"]
    check("v3_diagnosis", diagnosis)
    outcome = ROOT / registration["artifacts"]["outcomes"]
    existing = ([path for path in outcome.rglob("*") if path.is_file()]
                if outcome.exists() else [])
    checks["outcome_blind"] = {"pass": not existing,
        "detail": f"{len(existing)} v4 lifetime outcome artifacts before readiness"}
    passed = all(row["pass"] for row in checks.values())
    return {
        "schema_version": 1,
        "role": "v4 causal lifetime-exposure outcome-blind readiness",
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "source_lock_sha256": sha256_file(ROOT / registration["artifacts"]["source_lock"]),
        "pass": passed,
        "decision": "ready_for_runtime_smoke" if passed else "blocked_do_not_run",
        "checks": checks,
        "new_outcomes_read_or_generated": 0,
        "v2_controls_recomputed": False,
        "epa_validation_test_primary_confirmation_opened": False,
    }


def main() -> None:
    registration = load_registration(ROOT)
    destination = ROOT / registration["artifacts"]["readiness"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    write_json(destination, result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

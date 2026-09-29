"""Outcome-blind readiness audit for history-normalized Huber v3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_history_normalized_registration import assert_registration
from airproof.v6_history_normalized_runner import (
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
          "budgets1/2/4, cap8, delta1.345, lag6, original gates")[1])
    check("source_lock", lambda: {
        "sha256": sha256_file(ROOT / registration["artifacts"]["source_lock"]),
        "reused_files": len(verify_source_lock(ROOT, registration)["reused_control_sha256"]),
    })

    def v2_disposition():
        value = json.loads((ROOT / registration["development_history"]["v2_analysis"])
                           .read_text(encoding="utf-8"))
        gate = value["gate_evaluation"]
        if (gate["family_closed"] is not True or gate["selected_candidate"] is not None
                or gate["eligible"]):
            raise ValueError("v2 is not a closed no-nomination family")
        return "v2 closed; base choice is explicitly development-informed"
    check("v2_disposition", v2_disposition)

    def feasibility():
        value = json.loads((ROOT / registration["development_history"]["input_feasibility_audit"])
                           .read_text(encoding="utf-8"))
        if (value["budgets_inherited_without_change"] != [1.0, 2.0, 4.0]
                or value["development_scoring_read"] is not False
                or value["estimator_executed"] is not False):
            raise ValueError("history-budget input audit is not outcome-free or exact")
        return {key: row["binding_fraction"] for key, row in value["results"].items()}
    check("input_feasibility", feasibility)
    outcome = ROOT / registration["artifacts"]["outcomes"]
    existing = ([path for path in outcome.rglob("*") if path.is_file()]
                if outcome.exists() else [])
    checks["outcome_blind"] = {"pass": not existing,
        "detail": f"{len(existing)} v3 outcome artifacts before readiness"}
    passed = all(row["pass"] for row in checks.values())
    return {
        "schema_version": 1,
        "role": "v3 outcome-blind readiness",
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

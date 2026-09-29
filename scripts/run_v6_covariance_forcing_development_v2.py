"""Fail-closed controller for the frozen v2 mechanistic development campaign."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import REGISTRATION, sha256_file


def require_ready(registration: dict) -> dict:
    path = ROOT / registration["readiness"]["artifact"]
    if not path.is_file():
        raise RuntimeError("v2 readiness artifact absent; execution prohibited")
    status = json.loads(path.read_text(encoding="utf-8"))
    lock_path = ROOT / registration["producer"]["input_lock"]
    if (status.get("readiness_pass") is not True
            or status.get("registration_sha256") != sha256_file(ROOT / REGISTRATION)
            or not lock_path.is_file()
            or status.get("input_lock_sha256") != sha256_file(lock_path)
            or status.get("scientific_outcomes_read_or_generated") != 0):
        raise RuntimeError("v2 readiness, registration, or immutable input lock mismatch")
    return status


def planned_runs(registration: dict, seeds: list[int]) -> list[dict]:
    roles = registration.get("world_roles", {}).get("development")
    if roles is None:  # Tiny outcome-free parity fixtures used by the producer audit.
        cells = registration["scenarios"]
    else:
        registered = set(roles["seeds"])
        if not set(seeds) <= registered:
            raise ValueError("unregistered development seed")
        cells = roles["cells"]
    controls = registration["same_information_controls"]["names"]
    return [{"seed": int(seed), "scenario": cell, "candidate": candidate["id"],
             "methods": [candidate["id"], *controls]}
            for seed in seeds
            for cell in cells
            for candidate in registration["candidates"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--runtime-smoke", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    registration = json.loads((ROOT / REGISTRATION).read_text(encoding="utf-8"))
    if args.preflight_only:
        path = ROOT / registration["readiness"]["artifact"]
        status = json.loads(path.read_text()) if path.is_file() else None
        print(json.dumps({"ready": bool(status and status.get("readiness_pass") is True),
                          "readiness_artifact_exists": path.is_file()}, indent=2))
        return
    require_ready(registration)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    seeds = (registration["world_roles"]["runtime_smoke"]["seeds"]
             if args.runtime_smoke else registration["world_roles"]["development"]["seeds"])
    plan = planned_runs(registration, list(map(int, seeds)))
    from airproof.v6_covariance_forcing_runner import execute_registered_plan
    result = execute_registered_plan(registration, plan, runtime_smoke=args.runtime_smoke)
    if args.runtime_smoke:
        full = len(registration["world_roles"]["development"]["seeds"])
        smoke = len(registration["world_roles"]["runtime_smoke"]["seeds"])
        projected = result["wall_seconds"] / smoke * full * registration["runtime"]["smoke_projection_multiplier"]
        result["projected_full_wall_seconds"] = projected
        if projected > registration["runtime"]["smoke_projection_deadline_seconds"]:
            raise RuntimeError("registered full-run runtime projection exceeds one day")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

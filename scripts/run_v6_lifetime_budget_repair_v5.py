"""Fail-closed controller for exposed lifetime-budget repair v5."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file
from airproof.v6_lifetime_budget_repair_runner import execute, load_registration


def require_ready(root: str | Path = ROOT) -> dict:
    root = Path(root)
    registration = load_registration(root)
    readiness = json.loads((root / registration["artifacts"]["readiness"])
                           .read_text(encoding="utf-8"))
    if (readiness.get("pass") is not True or readiness.get("expected_jobs") != 60
            or readiness.get("expected_candidate_rows") != 120
            or readiness.get("reused_control_rows") != 240
            or readiness.get("scientific_outcomes_read_or_generated") != 0
            or readiness.get("controls_recomputed") is not False
            or readiness.get("source_lock_sha256") != sha256_file(
                root / registration["artifacts"]["source_lock"])
            or readiness.get("input_lock_sha256") != sha256_file(
                root / registration["input_reuse"]["input_lock"])):
        raise RuntimeError("repair readiness is absent, stale, or failed")
    return readiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--runtime-smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    ready = require_ready(ROOT)
    if args.preflight_only:
        print(json.dumps({"ready": True,
            "expected_candidate_rows": ready["expected_candidate_rows"]}, indent=2))
        return
    print(json.dumps(execute(smoke=args.runtime_smoke), indent=2))


if __name__ == "__main__":
    main()

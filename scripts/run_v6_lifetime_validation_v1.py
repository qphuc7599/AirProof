"""Fail-closed controller for the dedicated lifetime validation runner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file
from airproof.v6_lifetime_validation_runner import execute, load_registration


def require_ready(root: str | Path = ROOT) -> dict:
    root = Path(root)
    registration = load_registration(root)
    readiness_path = root / registration["input_producer"]["readiness"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    source = root / registration["input_producer"]["source_lock"]
    inputs = root / registration["input_producer"]["input_lock"]
    if (readiness.get("pass") is not True
            or readiness.get("expected_jobs") != 60
            or readiness.get("expected_rows") != 240
            or readiness.get("scientific_outcomes_read_or_generated") != 0
            or readiness.get("source_lock_sha256") != sha256_file(source)
            or readiness.get("input_lock_sha256") != sha256_file(inputs)):
        raise RuntimeError("lifetime validation readiness is absent, stale, or failed")
    return readiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--runtime-smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    readiness = require_ready(ROOT)
    if args.preflight_only:
        print(json.dumps({"ready": True, "expected_rows": readiness["expected_rows"]}, indent=2))
        return
    print(json.dumps(execute(smoke=args.runtime_smoke), indent=2))


if __name__ == "__main__":
    main()

"""Fail-closed runner skeleton for the registered covariance/forcing family."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "reports/v6/innovation_covariance_forcing_development_v1/readiness.json"


def require_ready() -> dict:
    if not READINESS.is_file():
        raise RuntimeError(
            "readiness artifact absent; run audit_v6_innovation_covariance_forcing_readiness.py"
        )
    status = json.loads(READINESS.read_text(encoding="utf-8"))
    if status.get("readiness_pass") is not True:
        raise RuntimeError("readiness failed; exact causal inputs are missing; outcome run prohibited")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.execute:
        raise SystemExit("choose exactly one of --preflight-only or --execute")
    status = require_ready()
    if args.preflight_only:
        print(json.dumps({"ready": True, "registration": status["registration"]}, indent=2))
        return
    # The outcome body is intentionally absent until a hash-bound exact input bundle
    # passes readiness. This prevents a developer from filling gaps by regenerating
    # citizen records from scoring truth or substituting identity physics.
    raise RuntimeError("runner skeleton has no authorized outcome executor")


if __name__ == "__main__":
    main()


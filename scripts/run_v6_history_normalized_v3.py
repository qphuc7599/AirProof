"""Fail-closed controller for history-normalized Huber v3."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import sha256_file
from airproof.v6_history_normalized_runner import (
    REGISTRATION,
    execute,
    load_registration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runtime-smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    registration = load_registration(ROOT)
    readiness = json.loads((ROOT / registration["artifacts"]["readiness"])
                           .read_text(encoding="utf-8"))
    lock_path = ROOT / registration["artifacts"]["source_lock"]
    if (readiness.get("pass") is not True
            or readiness.get("registration_sha256") != sha256_file(ROOT / REGISTRATION)
            or readiness.get("source_lock_sha256") != sha256_file(lock_path)
            or readiness.get("new_outcomes_read_or_generated") != 0):
        raise RuntimeError("v3 readiness or source lock mismatch")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    print(json.dumps(execute(smoke=args.runtime_smoke), indent=2))


if __name__ == "__main__":
    main()

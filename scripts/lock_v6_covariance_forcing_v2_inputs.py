"""Create the immutable v2 input lock after input generation and before outcomes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import (
    REGISTRATION,
    load_registration,
    sha256_file,
    verify_campaign_bundle,
    write_json,
)


def main() -> None:
    registration = load_registration(ROOT)
    input_dir = ROOT / registration["producer"]["campaign_bundle"]
    manifest = verify_campaign_bundle(ROOT, input_dir)
    lock_path = ROOT / registration["producer"]["input_lock"]
    if lock_path.exists():
        raise SystemExit("input lock already exists; overwrite prohibited")
    lock = {"schema_version": 1, "role": "immutable pre-outcome input lock",
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "input_manifest": registration["producer"]["campaign_manifest"],
        "input_manifest_sha256": sha256_file(input_dir / "manifest.json"),
        "payload_root_sha256": manifest["payload_root_sha256"],
        "global_calibration_sha256": sha256_file(input_dir / "global_calibration.json"),
        "operator_sha256": sha256_file(input_dir / "mechanism" / "physical_operator.npz"),
        "calibration_worlds": manifest["calibration_seeds"],
        "development_worlds": manifest["development_seeds"],
        "development_cells": manifest["development_cells"],
        "scientific_outcomes_before_lock": 0,
        "protected_namespaces_opened": False}
    write_json(lock_path, lock)
    print(json.dumps({"locked": True, "input_lock_sha256": sha256_file(lock_path),
                      "scientific_outcomes_before_lock": 0}, indent=2))


if __name__ == "__main__":
    main()

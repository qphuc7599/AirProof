"""Freeze public LOSO scale artifacts from explicit rows; never launch a campaign.

Input JSON contains records (PublicLOSOResidual fields) and metadata with training_end,
training_split_id and public_model_id. Optional age_edges/geometry_edges must already be
locked by the caller. Output directory must be new to preserve prior artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from airproof.v6_calibration import PublicLOSOResidual, fit_public_context_calibrator
from airproof.v6_estimator import bounded_development_candidates


def run(source, destination):
    source, destination = Path(source), Path(destination)
    raw = source.read_bytes()
    payload = json.loads(raw)
    if payload.get("role") not in ("component_diagnostic", "development_public_calibration"):
        raise ValueError("explicit non-confirmatory role required")
    rows = [PublicLOSOResidual(**{**row, "predictor_station_ids": tuple(row["predictor_station_ids"])})
            for row in payload["records"]]
    calibrated = fit_public_context_calibrator(rows, **payload["metadata"])
    module_root = Path(__file__).resolve().parents[1]
    sources = [module_root / "airproof" / name for name in ("v6_calibration.py", "v6_estimator.py")]
    report = {
        "role": payload["role"], "scientific_confirmation": False,
        "input_path": str(source.resolve()), "input_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {str(path.relative_to(module_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "calibrator": asdict(calibrated),
        "bounded_candidate_family": {name: asdict(cfg) for name, cfg in bounded_development_candidates().items()},
        "selection_performed": False,
        "remaining": ["independent post-selection interval calibration", "public-to-citizen covariance validation",
                      "paired development all-control outcomes", "unchanged target gates and prospective power"],
    }
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "public_calibration.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args.input, args.output))

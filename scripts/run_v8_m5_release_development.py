#!/usr/bin/env python3
"""Evaluate the predeclared V8 M5 family on exposed shared-resource outputs."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v8_release_assimilation import (  # noqa: E402
    ReleaseAssimilationConfig,
    persistent_reference_stress,
    run_protected_assimilation,
)

REGISTRATION = ROOT / "configs/v8/m5_release_assimilation_development_v1.json"
OUTPUT = ROOT / "reports/v8/m5_release_assimilation_development_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rmse(prediction: np.ndarray, truth: np.ndarray, burn: int) -> float:
    return float(np.sqrt(np.mean((prediction[burn:] - truth[burn:]) ** 2)))


def execute() -> dict:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if "predeclared" not in registration["status"] or len(registration["candidate_family"]) != 3:
        raise ValueError("development registration is not the frozen small family")
    source = ROOT / registration["source"]
    output_exists = OUTPUT.exists()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = {}
    burn, stop = registration["shared_parameters"]["score_half_open"]
    bias = np.asarray(registration["stress_regime"]["group_bias"], dtype=float)
    for seed in registration["seeds"]:
        job = source / "jobs" / str(seed) / registration["cells"][0]
        release_path = job / "protected_releases.npz"
        result_path = job / "result.json"
        hashes[str(release_path.relative_to(ROOT)).replace("\\", "/")] = _sha(release_path)
        hashes[str(result_path.relative_to(ROOT)).replace("\\", "/")] = _sha(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        nominal_scale = float(result["public_provenance"]["scale"]["pooled_scale"])
        with np.load(release_path, allow_pickle=False) as arrays:
            public = arrays["baseline"][:stop]
            truth = arrays["truth"][:stop]
            residual = arrays["residual"][:stop]
            residual_mask = arrays["residual_mask"][:stop]
            raw = arrays["raw_nonnegative"][:stop]
            raw_mask = arrays["raw_nonnegative_mask"][:stop]
        nominal_uncertainty = np.full_like(public, nominal_scale)
        stress_public, stress_uncertainty = persistent_reference_stress(
            public, nominal_uncertainty, bias
        )
        schedule = tuple(int(value) for value in result["privacy_release"]["scheduled_epochs"])
        public_scores = {
            "main_public_rmse": _rmse(public, truth, burn),
            "stress_public_rmse": _rmse(stress_public, truth, burn),
        }
        for candidate in registration["candidate_family"]:
            common = dict(
                groups=public.shape[1],
                scheduled_acquisition_epochs=schedule,
                deadline_epochs=registration["privacy_contract"]["deadline_epochs"],
                epsilon_history=registration["privacy_contract"]["epsilon_history"],
                k_min=registration["privacy_contract"]["k_min"],
                prior_variance=candidate["prior_variance"],
                public_uncertainty_gate=registration["shared_parameters"]["public_uncertainty_gate"],
                innovation_clip=registration["shared_parameters"]["innovation_clip"],
                correction_cap=registration["shared_parameters"]["correction_cap"],
            )
            residual_cfg = ReleaseAssimilationConfig(
                **common,
                laplace_scale=registration["privacy_contract"]["residual_laplace_scale"],
            )
            raw_cfg = ReleaseAssimilationConfig(
                **common,
                laplace_scale=registration["privacy_contract"]["nonnegative_raw_laplace_scale"],
            )
            main, main_diag = run_protected_assimilation(
                public, nominal_uncertainty, residual, residual_mask, residual_cfg
            )
            protected, protected_diag = run_protected_assimilation(
                stress_public, stress_uncertainty, residual, residual_mask, residual_cfg
            )
            raw_prediction, raw_diag = run_protected_assimilation(
                stress_public, stress_uncertainty, raw, raw_mask, raw_cfg
            )
            rows.append(
                {
                    "seed": seed,
                    "candidate": candidate["name"],
                    **public_scores,
                    "main_protected_rmse": _rmse(main, truth, burn),
                    "stress_protected_rmse": _rmse(protected, truth, burn),
                    "stress_nonnegative_raw_rmse": _rmse(raw_prediction, truth, burn),
                    "main_diagnostics": main_diag,
                    "stress_diagnostics": protected_diag,
                    "raw_diagnostics": raw_diag,
                    "config": asdict(residual_cfg),
                }
            )
    summary = {}
    for candidate in registration["candidate_family"]:
        selected = [row for row in rows if row["candidate"] == candidate["name"]]
        protected = np.asarray([row["stress_protected_rmse"] for row in selected])
        public = np.asarray([row["stress_public_rmse"] for row in selected])
        raw = np.asarray([row["stress_nonnegative_raw_rmse"] for row in selected])
        main = np.asarray([row["main_protected_rmse"] for row in selected])
        main_public = np.asarray([row["main_public_rmse"] for row in selected])
        summary[candidate["name"]] = {
            "mean_stress_rmse": float(protected.mean()),
            "mean_stress_public_rmse": float(public.mean()),
            "mean_stress_nonnegative_raw_rmse": float(raw.mean()),
            "stress_protected_minus_public": float(np.mean(protected - public)),
            "stress_protected_minus_nonnegative_raw": float(np.mean(protected - raw)),
            "protected_better_than_public_worlds": int(np.sum(protected < public)),
            "protected_better_than_nonnegative_raw_worlds": int(np.sum(protected < raw)),
            "main_protected_minus_public": float(np.mean(main - main_public)),
        }
    eligible = [
        item
        for item in registration["candidate_family"]
        if summary[item["name"]]["protected_better_than_public_worlds"] >= 9
        and summary[item["name"]]["protected_better_than_nonnegative_raw_worlds"] >= 9
    ]
    selected = min(
        eligible,
        key=lambda item: (summary[item["name"]]["mean_stress_rmse"], item["prior_variance"]),
    )["name"] if eligible else None
    result = {
        "schema_version": 1,
        "role": "exposed development only",
        "registration_sha256": _sha(REGISTRATION),
        "source_hashes": hashes,
        "preexisting_output_directory": output_exists,
        "all_candidate_rows_retained": True,
        "summary": summary,
        "selected_candidate": selected,
        "selection_passed": selected is not None,
        "confirmation_authorized": False,
    }
    (OUTPUT / "world_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    execute()

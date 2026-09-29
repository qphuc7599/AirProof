#!/usr/bin/env python3
"""Run the registered noncausal equal-total-exposure attribution control."""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from airproof.metrics import prediction_metrics
from airproof.v6_covariance_forcing_inputs import load_csr, restore_records, sha256_file, write_json
from airproof.v6_covariance_forcing_runner import _predict_one
from airproof.v6_lifetime_budget7_validation_v2_inputs import load_scoring_view, verify_bundle
from airproof.v6_lifetime_budget7_validation_v2_runner import fixed_estimator
from airproof.v7_lifetime_review import uniform_total_exposure_oracle

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = Path("configs/v7/lifetime_uniform_exposure_control_v1.json")


def _job(spec: tuple[str, dict, dict, int, str, str]) -> dict:
    output_raw, registration, source_registration, seed, cell, identity = spec
    output = Path(output_raw)
    target = output / "jobs" / str(seed) / cell
    result_path = target / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("stale uniform-exposure control result")
        return result
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bundle = ROOT / registration["input_bundle"]
    base = "severe_clean"
    with np.load(
        bundle / "prediction" / str(seed) / f"context_{base}.npz",
        allow_pickle=False,
    ) as arrays:
        public = arrays["public_center"].copy()
        forcing = arrays["residual_forcing"].copy()
    observations, arrivals = restore_records(
        bundle / "prediction" / str(seed) / f"observations_{cell}.npz"
    )
    effective, exposure = uniform_total_exposure_oracle(
        observations,
        arrivals,
        lag=int(registration["lag"]),
        per_user_exposure_budget=float(registration["per_user_exposure_budget"]),
    )
    estimator = fixed_estimator(source_registration)
    calibration = json.loads((bundle / "global_calibration.json").read_text(encoding="utf-8"))
    scale = float(
        calibration["innovation_scales"][
            source_registration["fixed_estimator"]["innovation_covariance"]
        ]
    )
    operator = load_csr(bundle / "mechanism" / "physical_operator.npz")
    live, reconstructed, diagnostics = _predict_one(
        public,
        effective,
        arrivals,
        operator,
        forcing,
        scale,
        estimator,
        "CANDIDATE",
    )
    prediction = target / "prediction.npz"
    np.savez_compressed(prediction, live=live, reconstructed=reconstructed)
    prediction_hash = sha256_file(prediction)
    scoring = load_scoring_view(bundle, seed, cell)
    truth = scoring["evaluation_truth"]
    events = scoring["event_mask"]
    threshold = float(scoring["event_threshold"])
    causal_path = (
        ROOT
        / registration["retained_outcomes"]
        / "jobs"
        / str(seed)
        / cell
        / registration["causal_comparator"]
        / "prediction.npz"
    )
    causal_manifest = json.loads(
        causal_path.with_name("prediction_manifest.json").read_text(encoding="utf-8")
    )
    if causal_manifest["prediction_sha256"] != sha256_file(causal_path):
        raise ValueError("retained causal prediction hash mismatch")
    with np.load(causal_path, allow_pickle=False) as arrays:
        causal = arrays["reconstructed"][: len(truth)].copy()
    reconstructed = reconstructed[: len(truth)]
    live = live[: len(truth)]
    late = slice(416, len(truth))
    metrics = prediction_metrics(truth, reconstructed, scoring["cell_groups"])
    result = {
        "schema_version": 1,
        "identity": identity,
        "seed": seed,
        "cell": cell,
        "candidate": registration["candidate"],
        "prediction_sha256": prediction_hash,
        "causal_prediction_sha256": causal_manifest["prediction_sha256"],
        "metrics": {
            **metrics,
            "live_rmse": float(np.sqrt(np.mean((live - truth) ** 2))),
            "event_recall": float(np.mean(live[events] >= threshold)),
            "late_rmse": float(np.sqrt(np.mean((reconstructed[late] - truth[late]) ** 2))),
            "causal_rmse": float(np.sqrt(np.mean((causal - truth) ** 2))),
            "causal_late_rmse": float(np.sqrt(np.mean((causal[late] - truth[late]) ** 2))),
        },
        "exposure": exposure,
        "solver_failure_rate": float(diagnostics["solver_failure_rate"]),
        "projected_gradient_inf": max(
            (
                float(row["projected_gradient_inf"])
                for row in diagnostics.get("epoch_terms", [])
            ),
            default=0.0,
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    return result


def _analyze(registration: dict, rows: list[dict], output: Path) -> dict:
    margin = float(registration["evaluation"]["noninferiority_margin_ratio"])
    overall = np.asarray(
        [row["metrics"]["causal_rmse"] - margin * row["metrics"]["rmse"] for row in rows]
    )
    late = np.asarray(
        [
            row["metrics"]["causal_late_rmse"] - margin * row["metrics"]["late_rmse"]
            for row in rows
        ]
    )
    rng = np.random.default_rng(registration["evaluation"]["bootstrap_seed"])
    replicates = int(registration["evaluation"]["bootstrap_replicates"])
    indices = rng.integers(0, len(rows), size=(replicates, len(rows)))
    summary = {}
    for name, values in (("overall", overall), ("late_third", late)):
        distribution = values[indices].mean(axis=1)
        summary[name] = {
            "mean_causal_minus_1.05_oracle": float(values.mean()),
            "upper_95": float(np.quantile(distribution, 0.95)),
            "causal_noninferior": float(np.quantile(distribution, 0.95)) <= 0,
        }
    analysis = {
        "schema_version": 1,
        "role": registration["role"],
        "jobs": len(rows),
        "worlds": len(registration["seeds"]),
        "cells": registration["cells"],
        "paired_noninferiority": summary,
        "all_solver_failures_zero": all(row["solver_failure_rate"] == 0 for row in rows),
        "all_equal_total_exposure": all(
            row["exposure"]["maximum_user_lifetime_exposure"]
            <= registration["per_user_exposure_budget"] + 1e-10
            for row in rows
        ),
        "scope": registration["oracle_scope"],
    }
    write_json(output / "analysis.json", analysis)
    return analysis


def execute(output: Path, workers: int) -> dict:
    registration_file = ROOT / REGISTRATION
    registration = json.loads(registration_file.read_text(encoding="utf-8"))
    source_registration = json.loads(
        (ROOT / registration["source_validation"]).read_text(encoding="utf-8")
    )
    input_lock = json.loads((ROOT / registration["input_lock"]).read_text(encoding="utf-8"))
    verify_bundle(
        ROOT,
        ROOT / registration["input_bundle"],
        expected_manifest_sha256=input_lock["manifest_sha256"],
    )
    source_hashes = {
        REGISTRATION.as_posix(): sha256_file(registration_file),
        "scripts/run_v7_lifetime_uniform_exposure_control.py": sha256_file(Path(__file__)),
        "airproof/v7_lifetime_review.py": sha256_file(ROOT / "airproof/v7_lifetime_review.py"),
        registration["source_validation"]: sha256_file(ROOT / registration["source_validation"]),
        registration["input_lock"]: sha256_file(ROOT / registration["input_lock"]),
    }
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if output.exists():
        lock = json.loads((output / "source_lock.json").read_text(encoding="utf-8"))
        if lock.get("identity") != identity:
            raise ValueError("uniform-control source lock mismatch")
    else:
        output.mkdir(parents=True)
        write_json(
            output / "source_lock.json",
            {
                "identity": identity,
                "source_hashes": source_hashes,
                "outcomes_before_lock": 0,
            },
        )
    specs = [
        (str(output), registration, source_registration, seed, cell, identity)
        for seed in registration["seeds"]
        for cell in registration["cells"]
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_job, spec) for spec in specs]):
            rows.append(future.result())
            write_json(
                output / "progress.json",
                {"completed": len(rows), "expected": len(specs), "failures": 0},
            )
    analysis = _analyze(registration, rows, output)
    return {
        "output": str(output),
        "jobs": len(rows),
        "analysis_sha256": sha256_file(output / "analysis.json"),
        "paired_noninferiority": analysis["paired_noninferiority"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output or Path(
        "reports/v7/reviewer_revision/lifetime_uniform_exposure_control_v1"
    )
    if not output.is_absolute():
        output = ROOT / output
    print(json.dumps(execute(output, args.workers), indent=2))


if __name__ == "__main__":
    main()

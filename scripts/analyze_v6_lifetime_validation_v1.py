"""Recompute and analyze the complete independent lifetime validation matrix."""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.metrics import prediction_metrics
from airproof.v6_covariance_forcing_inputs import sha256_file, write_json
from airproof.v6_lifetime_validation_inputs import load_scoring_view, verify_bundle
from airproof.v6_lifetime_validation_registration import evaluate
from airproof.v6_lifetime_validation_runner import (
    REGISTRATION,
    load_registration,
    verify_source_lock,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(left, right) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _scoring_path(bundle: Path, seed: int, cell: str) -> Path:
    base = "severe_clean" if cell in ("severe_drift", "severe_hotspot") else cell
    return bundle / "scoring" / str(seed) / f"scoring_{base}.npz"


def collect(registration: dict, *, smoke: bool) -> tuple[list[dict], dict]:
    verify_source_lock(ROOT, registration)
    source_lock = ROOT / registration["input_producer"]["source_lock"]
    input_lock_path = ROOT / registration["input_producer"]["input_lock"]
    input_lock = _load(input_lock_path)
    source_lock_sha, input_lock_sha = sha256_file(source_lock), sha256_file(input_lock_path)
    bundle = ROOT / registration["input_producer"]["bundle"]
    manifest = verify_bundle(
        ROOT, bundle, expected_manifest_sha256=input_lock["manifest_sha256"])
    seeds = (registration["runtime"]["runtime_smoke_seeds"] if smoke
             else registration["worlds"]["seeds"])
    methods = registration["methods"]
    rows, failures, result_hashes = [], [], {}
    outcome = ROOT / registration["artifacts"]["outcomes"]
    for seed, cell in product(map(int, seeds), registration["worlds"]["cells"]):
        result_path = outcome / "jobs" / str(seed) / cell / "result.json"
        relative = str(result_path.relative_to(ROOT)).replace("\\", "/")
        if not result_path.is_file():
            failures.append(f"missing:{relative}")
            continue
        result = _load(result_path)
        result_hashes[relative] = sha256_file(result_path)
        if (result.get("status") != "complete" or result.get("seed") != seed
                or result.get("cell") != cell
                or result.get("source_lock_sha256") != source_lock_sha
                or result.get("input_lock_sha256") != input_lock_sha
                or result.get("row_count") != len(methods)):
            failures.append(f"identity:{relative}")
        group = result.get("rows", [])
        if len(group) != len(methods) or {row.get("method") for row in group} != set(methods):
            failures.append(f"matrix:{relative}")
            continue
        scoring = load_scoring_view(bundle, seed, cell)
        truth, events = scoring["evaluation_truth"], scoring["event_mask"]
        threshold, groups = float(scoring["event_threshold"]), scoring["cell_groups"]
        scoring_path = _scoring_path(bundle, seed, cell)
        scoring_sha = sha256_file(scoring_path)
        summary = _load(bundle / "prediction" / str(seed) / "summary.json")
        inherited_resource = summary["cell_summary"][cell]["resource_invariants_pass"] is True
        for row in group:
            method = row["method"]
            arm = result_path.parent / method
            prediction, manifest_path = arm / "prediction.npz", arm / "prediction_manifest.json"
            if not prediction.is_file() or not manifest_path.is_file():
                failures.append(f"prediction_missing:{relative}:{method}")
                continue
            prediction_manifest = _load(manifest_path)
            if (prediction_manifest.get("scoring_loaded_during_prediction") is not False
                    or prediction_manifest.get("prediction_sha256") != sha256_file(prediction)
                    or row.get("prediction_sha256")
                    != prediction_manifest.get("prediction_sha256")):
                failures.append(f"prediction_integrity:{relative}:{method}")
            with np.load(prediction, allow_pickle=False) as arrays:
                live = arrays["live"][:len(truth)]
                reconstructed = arrays["reconstructed"][:len(truth)]
            metrics = prediction_metrics(truth, reconstructed, groups)
            live_metrics = prediction_metrics(truth, live, groups)
            recomputed = {
                "seed": seed, "scenario": cell, "method": method,
                "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
                "worst_group_rmse": float(metrics["worst_group_rmse"]),
                "live_rmse": float(live_metrics["rmse"]),
                "event_recall": float(np.mean(live[events] >= threshold)),
                "event_support": int(events.sum()), "event_threshold": threshold,
                "prediction_horizon_half_open": [
                    registration["worlds"]["burn_in_epochs"],
                    registration["worlds"]["acquisition_epochs"]
                    + registration["worlds"]["fixed_lag_epochs"]],
                "scoring_horizon_half_open": [
                    registration["worlds"]["burn_in_epochs"],
                    registration["worlds"]["acquisition_epochs"]],
                "rmse_clock": "fixed_lag_reconstructed_after_registered_lag",
                "live_rmse_clock": "immutable_prediction_at_epoch",
                "event_recall_clock": "immutable_live_prediction_at_epoch",
                "information_fingerprint": prediction_manifest["information_fingerprint"],
                "solver_failure_rate": prediction_manifest["solver_failure_rate"],
                "projected_gradient_inf": prediction_manifest["projected_gradient_inf"],
                "maximum_absolute_correction": prediction_manifest[
                    "maximum_absolute_correction"],
                "maximum_user_lifetime_exposure": prediction_manifest[
                    "maximum_user_lifetime_exposure"],
                "wall_seconds": prediction_manifest["wall_seconds"],
                "process_cpu_seconds": prediction_manifest["process_cpu_seconds"],
                "peak_rss_bytes": prediction_manifest["peak_rss_bytes"],
                "prediction_identity": prediction_manifest["identity"],
                "prediction_sha256": prediction_manifest["prediction_sha256"],
                "scoring_sha256": scoring_sha,
                "resource_invariants_pass": inherited_resource,
                "finite_outputs": bool(np.isfinite(live).all()
                                       and np.isfinite(reconstructed).all()),
            }
            mismatched = [field for field, value in recomputed.items()
                          if field not in row or not _same(row[field], value)]
            if mismatched:
                failures.append(
                    f"recomputed_row:{relative}:{method}:{','.join(mismatched)}")
            verified = dict(row)
            verified.update(recomputed)
            rows.append(verified)
    expected = len(seeds) * len(registration["worlds"]["cells"]) * len(methods)
    if len(rows) != expected:
        failures.append(f"row_count:{len(rows)}!={expected}")
    finite = all(row.get("finite_outputs") is True and all(math.isfinite(float(row[field]))
        for field in ("rmse", "event_recall", "wall_seconds", "process_cpu_seconds",
                      "peak_rss_bytes")) for row in rows)
    if not finite:
        failures.append("nonfinite")
    if not smoke and expected != 240:
        failures.append(f"registered_full_matrix_not_240:{expected}")
    return rows, {
        "pass": not failures, "failures": sorted(set(failures)),
        "expected_rows": expected, "complete_rows": len(rows),
        "source_lock_sha256": source_lock_sha, "input_lock_sha256": input_lock_sha,
        "input_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "input_payload_root_sha256": manifest["payload_root_sha256"],
        "result_sha256": result_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    registration = load_registration(ROOT)
    rows, audit = collect(registration, smoke=args.smoke)
    gate = evaluate(registration, rows) if audit["pass"] and not args.smoke else None
    result = {
        "schema_version": 1,
        "scope": ("one-world runtime smoke; descriptive only" if args.smoke else
                  "independent 12-world five-cell synthetic mechanism validation"),
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "matrix_audit": audit, "gate_evaluation": gate,
        "historical_v4_rerun": False,
        "epa_test_primary_confirmation_opened": False,
    }
    output = ROOT / registration["artifacts"]["outcomes"] / (
        "runtime_smoke_audit.json" if args.smoke else "analysis.json")
    write_json(output, result)
    print(json.dumps({"artifact": str(output.relative_to(ROOT)),
        "matrix_pass": audit["pass"], "gate_evaluation": gate}, indent=2))
    if not audit["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

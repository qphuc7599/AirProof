"""Recompute candidate metrics and join hash-locked validation-v1 controls."""
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
from airproof.v6_lifetime_budget_repair_registration import evaluate
from airproof.v6_lifetime_budget_repair_runner import REGISTRATION, load_registration, verify_source_lock
from airproof.v6_lifetime_validation_inputs import load_scoring_view, verify_bundle


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(left, right) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12) if isinstance(left, (int, float)) and isinstance(right, (int, float)) else left == right


def collect(registration: dict, *, smoke: bool) -> tuple[list[dict], dict]:
    lock = verify_source_lock(ROOT, registration)
    source_path = ROOT / registration["artifacts"]["source_lock"]
    input_path = ROOT / registration["input_reuse"]["input_lock"]
    source_sha, input_sha = sha256_file(source_path), sha256_file(input_path)
    input_lock = _load(input_path)
    bundle = ROOT / registration["input_reuse"]["bundle"]
    manifest = verify_bundle(ROOT, bundle,
        expected_manifest_sha256=input_lock["manifest_sha256"])
    seeds = (registration["runtime"]["runtime_smoke_seeds"] if smoke
             else registration["input_reuse"]["seeds"])
    rows, failures, result_hashes = [], [], {}
    outcome = ROOT / registration["artifacts"]["outcomes"]
    for seed, cell in product(map(int, seeds), registration["input_reuse"]["cells"]):
        result_path = outcome / "jobs" / str(seed) / cell / "result.json"
        relative = result_path.relative_to(ROOT).as_posix()
        if not result_path.is_file():
            failures.append(f"missing:{relative}")
            continue
        result, result_hashes[relative] = _load(result_path), sha256_file(result_path)
        if (result.get("status") != "complete" or result.get("seed") != seed
                or result.get("cell") != cell or result.get("source_lock_sha256") != source_sha
                or result.get("input_lock_sha256") != input_sha
                or result.get("row_count") != len(registration["candidates"])
                or result.get("controls_recomputed") is not False):
            failures.append(f"identity:{relative}")
        scoring = load_scoring_view(bundle, seed, cell)
        truth, events = scoring["evaluation_truth"], scoring["event_mask"]
        threshold, groups = float(scoring["event_threshold"]), scoring["cell_groups"]
        candidate_rows = result.get("rows", [])
        if {row.get("candidate") for row in candidate_rows} != {
                candidate["id"] for candidate in registration["candidates"]}:
            failures.append(f"candidate_matrix:{relative}")
            continue
        parent_path = (ROOT / registration["input_reuse"]["source_validation_outcomes"]
                       / "jobs" / str(seed) / cell / "result.json")
        parent = _load(parent_path)
        if sha256_file(parent_path) != lock["upstream_sha256"][parent_path.relative_to(ROOT).as_posix()]:
            failures.append(f"parent_hash:{relative}")
        for candidate_row in candidate_rows:
            candidate = candidate_row["candidate"]
            arm = result_path.parent / candidate
            prediction, manifest_path = arm / "prediction.npz", arm / "prediction_manifest.json"
            prediction_manifest = _load(manifest_path)
            if (prediction_manifest.get("scoring_loaded_during_prediction") is not False
                    or prediction_manifest.get("prediction_sha256") != sha256_file(prediction)):
                failures.append(f"prediction_integrity:{relative}:{candidate}")
                continue
            with np.load(prediction, allow_pickle=False) as arrays:
                live, reconstructed = arrays["live"][:len(truth)], arrays["reconstructed"][:len(truth)]
            metrics, live_metrics = prediction_metrics(truth, reconstructed, groups), prediction_metrics(truth, live, groups)
            recomputed = {"rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
                "worst_group_rmse": float(metrics["worst_group_rmse"]),
                "live_rmse": float(live_metrics["rmse"]),
                "event_recall": float(np.mean(live[events] >= threshold)),
                "event_support": int(events.sum()), "event_threshold": threshold,
                "information_fingerprint": prediction_manifest["information_fingerprint"],
                "solver_failure_rate": prediction_manifest["solver_failure_rate"],
                "projected_gradient_inf": prediction_manifest["projected_gradient_inf"],
                "maximum_absolute_correction": prediction_manifest["maximum_absolute_correction"],
                "maximum_user_lifetime_exposure": prediction_manifest["maximum_user_lifetime_exposure"],
                "wall_seconds": prediction_manifest["wall_seconds"],
                "process_cpu_seconds": prediction_manifest["process_cpu_seconds"],
                "peak_rss_bytes": prediction_manifest["peak_rss_bytes"],
                "prediction_identity": prediction_manifest["identity"],
                "prediction_sha256": prediction_manifest["prediction_sha256"],
                "resource_invariants_pass": all(row.get("resource_invariants_pass") is True for row in parent["rows"]),
                "finite_outputs": bool(np.isfinite(live).all() and np.isfinite(reconstructed).all())}
            mismatch = [key for key, value in recomputed.items()
                        if key not in candidate_row or not _same(candidate_row[key], value)]
            if mismatch:
                failures.append(f"recomputed_row:{relative}:{candidate}:{','.join(mismatch)}")
            verified = dict(candidate_row); verified.update(recomputed); rows.append(verified)
            for control in parent["rows"]:
                copied = dict(control); copied["candidate"] = candidate
                rows.append(copied)
    expected = len(seeds) * 5 * len(registration["candidates"]) * 5
    if len(rows) != expected:
        failures.append(f"row_count:{len(rows)}!={expected}")
    return rows, {"pass": not failures, "failures": sorted(set(failures)),
        "expected_comparison_rows": expected, "complete_comparison_rows": len(rows),
        "new_candidate_rows": len(seeds) * 5 * 2, "reused_control_rows": len(seeds) * 5 * 4,
        "controls_recomputed": False, "source_lock_sha256": source_sha,
        "input_lock_sha256": input_sha, "input_payload_root_sha256": manifest["payload_root_sha256"],
        "result_sha256": result_hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    registration = load_registration(ROOT)
    rows, audit = collect(registration, smoke=args.smoke)
    gate = evaluate(registration, rows) if audit["pass"] and not args.smoke else None
    result = {"schema_version": 1,
        "scope": "one-world exposed runtime smoke" if args.smoke else "twelve-world exposed repair development",
        "registration": REGISTRATION, "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "matrix_audit": audit, "gate_evaluation": gate,
        "validation_v1_rescued": False, "historical_v4_rerun": False,
        "epa_primary_confirmation_opened": False}
    output = ROOT / registration["artifacts"]["outcomes"] / (
        "runtime_smoke_audit.json" if args.smoke else "analysis.json")
    write_json(output, result)
    print(json.dumps({"artifact": output.relative_to(ROOT).as_posix(),
        "matrix_pass": audit["pass"], "gate_evaluation": gate}, indent=2))
    if not audit["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

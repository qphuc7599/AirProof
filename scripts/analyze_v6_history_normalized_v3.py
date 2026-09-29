"""Verify and analyze the complete v3 history-normalized development matrix."""
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
from airproof.v6_history_normalized_registration import evaluate
from airproof.v6_history_normalized_runner import (
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


def collect(registration: dict, *, smoke: bool) -> tuple[list[dict], dict, list[dict]]:
    lock = verify_source_lock(ROOT, registration)
    lock_path = ROOT / registration["artifacts"]["source_lock"]
    lock_sha = sha256_file(lock_path)
    outcome = ROOT / registration["artifacts"]["outcomes"]
    seeds = (registration["runtime"]["runtime_smoke_seeds"] if smoke
             else registration["inputs"]["development_seeds"])
    cells = registration["inputs"]["cells"]
    candidates = [row["id"] for row in registration["candidates"]]
    new_rows, comparison, base_rows = [], [], []
    failures = []
    result_hashes = {}
    v2_root = ROOT / registration["controls"]["source_campaign"] / "jobs"
    v2_candidate = registration["controls"]["source_candidate"]
    for seed, cell in product(map(int, seeds), cells):
        result_path = outcome / "jobs" / str(seed) / cell / "result.json"
        relative = str(result_path.relative_to(ROOT)).replace("\\", "/")
        if not result_path.is_file():
            failures.append(f"missing:{relative}")
            continue
        result = _load(result_path)
        result_hashes[relative] = sha256_file(result_path)
        if (result.get("status") != "complete" or result.get("seed") != seed
                or result.get("cell") != cell
                or result.get("v3_source_lock_sha256") != lock_sha
                or result.get("row_count") != len(candidates)):
            failures.append(f"identity:{relative}")
        group = result.get("rows", [])
        if (len(group) != len(candidates)
                or {row.get("candidate") for row in group} != set(candidates)):
            failures.append(f"new_matrix:{relative}")
        v2_result_path = v2_root / str(seed) / cell / "result.json"
        v2 = _load(v2_result_path)
        controls = {row["method"]: row for row in v2["rows"]
                    if row["candidate"] == v2_candidate}
        if set(controls) != {"CANDIDATE", "PUBLIC", "SQ", "HUBER"}:
            failures.append(f"v2_controls:{relative}")
            continue
        scoring_path = (ROOT / registration["inputs"]["campaign_bundle"] / "scoring"
                        / str(seed) / "scoring_only.npz")
        with np.load(scoring_path, allow_pickle=False) as scoring:
            truth = scoring["evaluation_truth"]
            events = scoring["event_mask"]
            threshold = float(scoring["event_threshold"])
            groups = scoring["cell_groups"]
        scoring_sha = sha256_file(scoring_path)
        inherited_resource = all(row.get("resource_invariants_pass") is True
                                 for row in controls.values())
        for row in group:
            candidate = row["candidate"]
            arm = outcome / "jobs" / str(seed) / cell / candidate
            prediction = arm / "prediction.npz"
            manifest_path = arm / "prediction_manifest.json"
            if not prediction.is_file() or not manifest_path.is_file():
                failures.append(f"prediction_missing:{relative}:{candidate}")
                continue
            manifest = _load(manifest_path)
            if (manifest.get("scoring_loaded_during_prediction") is not False
                    or manifest.get("prediction_sha256") != sha256_file(prediction)
                    or row.get("prediction_sha256") != manifest.get("prediction_sha256")):
                failures.append(f"prediction_integrity:{relative}:{candidate}")
            with np.load(prediction, allow_pickle=False) as arrays:
                live = arrays["live"][:len(truth)]
                reconstructed = arrays["reconstructed"][:len(truth)]
            metrics = prediction_metrics(truth, reconstructed, groups)
            live_metrics = prediction_metrics(truth, live, groups)
            recall = float(np.mean(live[events] >= threshold)) if events.any() else None
            recomputed = {
                "seed": seed,
                "scenario": cell,
                "candidate": candidate,
                "method": "CANDIDATE",
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "worst_group_rmse": float(metrics["worst_group_rmse"]),
                "live_rmse": float(live_metrics["rmse"]),
                "event_recall": recall,
                "event_support": int(events.sum()),
                "event_threshold": threshold,
                "information_fingerprint": manifest["information_fingerprint"],
                "solver_failure_rate": manifest["solver_failure_rate"],
                "projected_gradient_inf": manifest["projected_gradient_inf"],
                "maximum_absolute_correction": manifest["maximum_absolute_correction"],
                "maximum_user_window_weight": manifest["maximum_user_window_weight"],
                "per_user_weight_budget": manifest["per_user_weight_budget"],
                "wall_seconds": manifest["wall_seconds"],
                "process_cpu_seconds": manifest["process_cpu_seconds"],
                "peak_rss_bytes": manifest["peak_rss_bytes"],
                "prediction_identity": manifest["identity"],
                "prediction_sha256": manifest["prediction_sha256"],
                "scoring_sha256": scoring_sha,
                "resource_invariants_pass": inherited_resource,
                "finite_outputs": bool(np.isfinite(live).all()
                                       and np.isfinite(reconstructed).all()),
            }
            mismatched = [field for field, value in recomputed.items()
                          if field not in row or not _same(row[field], value)]
            if mismatched:
                failures.append(
                    f"recomputed_row:{relative}:{candidate}:{','.join(mismatched)}")
            verified = dict(row)
            verified.update(recomputed)
            new_rows.append(verified)
            comparison.append(verified)
            for method in ("PUBLIC", "SQ", "HUBER"):
                reused = dict(controls[method])
                reused["candidate"] = candidate
                reused["reused_from_v2"] = True
                comparison.append(reused)
            base = dict(controls["CANDIDATE"])
            base["candidate"] = candidate
            base["method"] = "BASE_BOUNDED_HUBER"
            base["reused_from_v2"] = True
            base_rows.append(base)
    expected_new = len(seeds) * len(cells) * len(candidates)
    expected_comparison = expected_new * 4
    if len(new_rows) != expected_new:
        failures.append(f"new_row_count:{len(new_rows)}!={expected_new}")
    if len(comparison) != expected_comparison:
        failures.append(f"comparison_row_count:{len(comparison)}!={expected_comparison}")
    for relative, expected in lock["reused_control_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            failures.append(f"reused_hash:{relative}")
    finite = all(row.get("finite_outputs") is True and all(math.isfinite(float(row[field]))
        for field in ("rmse", "event_recall", "wall_seconds", "process_cpu_seconds",
                      "peak_rss_bytes")) for row in new_rows)
    if not finite:
        failures.append("nonfinite_new_row")
    return comparison, {
        "pass": not failures, "failures": sorted(set(failures)),
        "expected_new_rows": expected_new, "complete_new_rows": len(new_rows),
        "expected_comparison_rows": expected_comparison,
        "complete_comparison_rows": len(comparison),
        "result_sha256": result_hashes, "source_lock_sha256": lock_sha,
        "v2_controls_recomputed": False,
    }, base_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    registration = load_registration(ROOT)
    rows, audit, base = collect(registration, smoke=args.smoke)
    gate = evaluate(registration, rows) if audit["pass"] and not args.smoke else None
    result = {
        "schema_version": 1,
        "scope": ("one-world runtime smoke; descriptive only" if args.smoke else
                  "sequential exposed development on reused v2 worlds"),
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(ROOT / REGISTRATION),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "matrix_audit": audit,
        "gate_evaluation": gate,
        "base_bounded_huber_rows": base,
        "historical_v4_rerun": False,
        "v2_public_sq_huber_recomputed": False,
        "epa_validation_test_primary_confirmation_opened": False,
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

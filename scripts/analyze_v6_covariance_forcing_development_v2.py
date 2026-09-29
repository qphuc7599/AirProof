"""Audit and analyze the frozen v2 covariance/forcing outcome matrix."""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.v6_covariance_forcing_inputs import (
    REGISTRATION,
    sha256_file,
    write_json,
)
from airproof.v6_covariance_forcing_registration import (
    evaluate_joint_gates,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_matrix(registration: dict, *, smoke: bool) -> tuple[list[dict], dict]:
    output = ROOT / registration["output"]["directory"]
    lock_path = ROOT / registration["producer"]["input_lock"]
    lock = _load_json(lock_path)
    lock_sha = sha256_file(lock_path)
    input_dir = ROOT / registration["producer"]["campaign_bundle"]
    readiness_path = ROOT / registration["readiness"]["artifact"]
    readiness = _load_json(readiness_path)
    input_manifest = _load_json(input_dir / "manifest.json")
    seeds = (registration["world_roles"]["runtime_smoke"]["seeds"] if smoke
             else registration["world_roles"]["development"]["seeds"])
    cells = registration["world_roles"]["development"]["cells"]
    candidates = [row["id"] for row in registration["candidates"]]
    methods = ("CANDIDATE", "PUBLIC", "SQ", "HUBER")
    rows: list[dict] = []
    result_hashes: dict[str, str] = {}
    failures: list[str] = []
    if (readiness.get("readiness_pass") is not True
            or readiness.get("input_lock_sha256") != lock_sha):
        failures.append("readiness_lock_mismatch")
    if (lock.get("registration_sha256") != sha256_file(ROOT / REGISTRATION)
            or lock.get("input_manifest_sha256")
            != sha256_file(input_dir / "manifest.json")):
        failures.append("input_lock_binding_mismatch")
    gate_source = "airproof/v6_covariance_forcing_registration.py"
    if (input_manifest.get("source_files", {}).get(gate_source)
            != sha256_file(ROOT / gate_source)
            or input_manifest.get("source_files", {}).get(gate_source)
            != sha256_file(input_dir / "source_snapshot" / gate_source)):
        failures.append("locked_gate_algorithm_mismatch")
    for seed, cell in product(map(int, seeds), cells):
        result_path = output / "jobs" / str(seed) / cell / "result.json"
        relative = str(result_path.relative_to(ROOT)).replace("\\", "/")
        if not result_path.is_file():
            failures.append(f"missing:{relative}")
            continue
        result = _load_json(result_path)
        result_hashes[relative] = sha256_file(result_path)
        if (result.get("status") != "complete" or result.get("seed") != seed
                or result.get("cell") != cell
                or result.get("input_lock_sha256") != lock_sha
                or result.get("input_manifest_sha256") != lock["input_manifest_sha256"]):
            failures.append(f"identity:{relative}")
        group = result.get("rows", [])
        expected = set(product(candidates, methods))
        actual = {(row.get("candidate"), row.get("method")) for row in group}
        if len(group) != len(expected) or actual != expected:
            failures.append(f"matrix:{relative}")
        for row in group:
            candidate, method = row.get("candidate"), row.get("method")
            if row.get("seed") != seed or row.get("scenario") != cell:
                failures.append(f"row_identity:{relative}:{candidate}:{method}")
                continue
            arm = output / "jobs" / str(seed) / cell / str(candidate) / str(method)
            prediction = arm / "prediction.npz"
            manifest_path = arm / "prediction_manifest.json"
            if not prediction.is_file() or not manifest_path.is_file():
                failures.append(f"prediction_missing:{relative}:{candidate}:{method}")
                continue
            manifest = _load_json(manifest_path)
            if (manifest.get("scoring_loaded_during_prediction") is not False
                    or manifest.get("prediction_sha256") != sha256_file(prediction)
                    or row.get("prediction_sha256") != manifest.get("prediction_sha256")
                    or row.get("prediction_identity") != manifest.get("identity")):
                failures.append(f"prediction_integrity:{relative}:{candidate}:{method}")
            scoring_path = input_dir / "scoring" / str(seed) / "scoring_only.npz"
            if row.get("scoring_sha256") != sha256_file(scoring_path):
                failures.append(f"scoring_integrity:{relative}:{candidate}:{method}")
        rows.extend(group)
    expected_rows = len(seeds) * len(cells) * len(candidates) * len(methods)
    duplicate_keys = len(rows) != len({(row.get("seed"), row.get("scenario"),
        row.get("candidate"), row.get("method")) for row in rows})
    if len(rows) != expected_rows:
        failures.append(f"row_count:{len(rows)}!={expected_rows}")
    if duplicate_keys:
        failures.append("duplicate_row_keys")
    audit = {
        "pass": not failures,
        "failures": sorted(set(failures)),
        "expected_jobs": len(seeds) * len(cells),
        "complete_jobs": len(result_hashes),
        "expected_rows": expected_rows,
        "complete_rows": len(rows),
        "result_sha256": result_hashes,
        "input_lock_sha256": lock_sha,
        "input_manifest_sha256": lock["input_manifest_sha256"],
        "readiness_sha256": sha256_file(readiness_path),
        "locked_gate_algorithm_sha256": input_manifest["source_files"][gate_source],
        "predictions_precede_scoring": not any(
            item.startswith("prediction_integrity") for item in failures),
    }
    return rows, audit


def invariant_audit(registration: dict, rows: list[dict]) -> dict:
    gates = registration["joint_gates"]
    failures: list[str] = []
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["seed"], row["scenario"], row["candidate"])
        grouped.setdefault(key, []).append(row)
        numeric = ("rmse", "event_recall", "wall_seconds", "process_cpu_seconds",
                   "peak_rss_bytes")
        if (row.get("finite_outputs") is not True
                or not all(math.isfinite(float(row[name])) for name in numeric)):
            failures.append(f"finite:{key}:{row['method']}")
        if row.get("resource_invariants_pass") is not True:
            failures.append(f"resource:{key}:{row['method']}")
        if row["method"] != "PUBLIC" and (
                float(row["solver_failure_rate"]) > gates["all_solver_failure_rates_max"]
                or float(row["projected_gradient_inf"])
                > gates["all_projected_gradient_inf_max"]):
            failures.append(f"solver:{key}:{row['method']}")
        if (row["method"] == "CANDIDATE"
                and float(row["maximum_absolute_correction"])
                > gates["all_candidate_absolute_corrections_max"]):
            failures.append(f"cap:{key}")
    for key, group in grouped.items():
        if len(group) != 4 or len({row["information_fingerprint"] for row in group}) != 1:
            failures.append(f"fingerprint:{key}")
    return {"pass": not failures, "failures": sorted(set(failures)),
            "comparison_groups": len(grouped)}


def descriptive_means(rows: list[dict]) -> list[dict]:
    keys = sorted({(row["scenario"], row["candidate"], row["method"])
                   for row in rows})
    output = []
    for scenario, candidate, method in keys:
        group = [row for row in rows if row["scenario"] == scenario
                 and row["candidate"] == candidate and row["method"] == method]
        output.append({"scenario": scenario, "candidate": candidate, "method": method,
            "worlds": len(group), "mean_rmse": fmean(float(row["rmse"]) for row in group),
            "mean_event_recall": fmean(float(row["event_recall"]) for row in group)})
    return output


def render_markdown(result: dict) -> str:
    lines = ["# V6 covariance × forcing development v2", "",
        f"- Scope: `{result['scope']}`",
        (f"- Matrix integrity: **{'PASS' if result['matrix_audit']['pass'] else 'FAIL'}** "
         f"({result['matrix_audit']['complete_rows']}/{result['matrix_audit']['expected_rows']} rows)"),
        f"- Correctness/resource invariants: **{'PASS' if result['invariant_audit']['pass'] else 'FAIL'}**",
        "- Historical v4 evidence was reused without rerunning it.", ""]
    if result.get("runtime_projection") is not None:
        projection = result["runtime_projection"]
        lines += [f"- Registered full-run projection: {projection['projected_full_wall_seconds'] / 3600:.2f} h",
                  f"- Projection gate: **{'PASS' if projection['pass'] else 'FAIL'}**", ""]
    if result.get("gate_evaluation") is not None:
        gate = result["gate_evaluation"]
        lines += [f"- Eligible candidates: `{gate['eligible']}`",
                  f"- Selected candidate: `{gate['selected_candidate']}`",
                  f"- Family closed without nomination: `{gate['family_closed']}`", "",
                  "| Candidate | Pass | Clean/SQ | Drift attenuation | Hotspot attenuation | Drift event loss (pp) | Hotspot event loss (pp) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in gate["candidates"]:
            metric = row["metrics"]
            lines.append("| {candidate} | {passed} | {clean:.4f} | {drift:.4f} | "
                "{hotspot:.4f} | {drift_event:.3f} | {hotspot_event:.3f} |".format(
                    candidate=row["candidate"], passed=row["pass"],
                    clean=metric["clean_ratio"],
                    drift=metric["attenuation"]["severe_drift"],
                    hotspot=metric["attenuation"]["severe_hotspot"],
                    drift_event=metric["event_loss_percentage_points"]["severe_drift"],
                    hotspot_event=metric["event_loss_percentage_points"]["severe_hotspot"]))
    if result["matrix_audit"]["failures"] or result["invariant_audit"]["failures"]:
        lines += ["", "## Failures", ""]
        lines.extend(f"- `{item}`" for item in (
            result["matrix_audit"]["failures"] + result["invariant_audit"]["failures"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Audit the registered one-world runtime smoke only.")
    args = parser.parse_args()
    registration_path = ROOT / REGISTRATION
    registration = _load_json(registration_path)
    rows, matrix = collect_matrix(registration, smoke=args.smoke)
    invariants = invariant_audit(registration, rows) if rows else {
        "pass": False, "failures": ["no_rows"], "comparison_groups": 0}
    result = {
        "schema_version": 1,
        "scope": ("registered one-world runtime smoke; descriptive and retained, never "
                  "candidate selection" if args.smoke else
                  "complete eight-world exposed mechanistic-synthetic development"),
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(registration_path),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "historical_v4_rerun": False,
        "matrix_audit": matrix,
        "invariant_audit": invariants,
        "descriptive_means": descriptive_means(rows),
        "gate_evaluation": None,
        "runtime_projection": None,
    }
    if args.smoke and matrix["pass"]:
        smoke_summary = _load_json(ROOT / registration["output"]["directory"]
                                   / "runtime_smoke.json")
        factor = (len(registration["world_roles"]["development"]["seeds"])
                  / len(registration["world_roles"]["runtime_smoke"]["seeds"])
                  * registration["runtime"]["smoke_projection_multiplier"])
        projected = float(smoke_summary["wall_seconds"]) * factor
        result["runtime_projection"] = {
            "smoke_wall_seconds": smoke_summary["wall_seconds"],
            "projected_full_wall_seconds": projected,
            "deadline_seconds": registration["runtime"]["smoke_projection_deadline_seconds"],
            "pass": projected <= registration["runtime"]["smoke_projection_deadline_seconds"],
        }
    elif not args.smoke and matrix["pass"] and invariants["pass"]:
        calibration = _load_json(ROOT / registration["producer"]["global_calibration"])
        result["gate_evaluation"] = evaluate_joint_gates(
            registration, rows, calibration["innovation_scales"])
    destination = ROOT / registration["output"]["directory"] / (
        "runtime_smoke_audit.json" if args.smoke else "analysis.json")
    write_json(destination, result)
    markdown = destination.with_suffix(".md")
    markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"artifact": str(destination.relative_to(ROOT)),
        "matrix_pass": matrix["pass"], "invariants_pass": invariants["pass"],
        "gate_evaluation": result["gate_evaluation"],
        "runtime_projection": result["runtime_projection"]}, indent=2))
    if not matrix["pass"] or not invariants["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

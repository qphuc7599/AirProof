"""Analyze the complete registered v6 fairness validation without primary inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, float); mean = float(values.mean())
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "descriptive_95_ci": None}
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    radius = float(t.ppf(.975, len(values)-1)*se)
    return {"n": len(values), "mean": mean,
            "descriptive_95_ci": [mean-radius, mean+radius]}


def flatten_metrics(row):
    metrics = row["metrics"]
    return {"coverage_gap": metrics["coverage_gap"],
        "worst_group_rmse": metrics["worst_group_rmse"], "rmse": metrics["rmse"],
        "minimum_24epoch_capped_service": metrics["minimum_24epoch_capped_service"],
        "worst_available_but_unserved_streak": metrics["worst_available_but_unserved_streak"],
        "total_distinct_user_debt": sum(metrics["distinct_user_debt_by_group"].values()),
        "selected_records": metrics["selected_records"],
        "mean_max_opportunity_deficit": metrics["mean_max_opportunity_deficit"],
        "mean_max_allocation_deficit": metrics["mean_max_allocation_deficit"]}


def write_csv(path, rows):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="reports/v6/fairness_validation_v1")
    args = parser.parse_args(); root = Path(args.input)
    registration = json.loads((ROOT/"configs/v6/fairness_validation.json").read_text())
    outcomes_path = root/"outcomes.json"; outcomes = json.loads(outcomes_path.read_text())
    if not outcomes["complete"] or len(outcomes["rows"]) != registration["matrix"]["total_evaluations"]:
        raise ValueError("complete registered 144-row matrix required")
    rows = outcomes["rows"]
    keys = {(r["seed"], r["allocator"], r["relay"], r["robustification"]) for r in rows}
    expected = {(seed, allocator, relay, robust)
        for seed in registration["seeds"]
        for allocator in registration["allocator_levels"]
        for relay in (False, True) for robust in (False, True)}
    if keys != expected or len(keys) != len(rows):
        raise ValueError("missing, duplicate, or extra evaluation identity")
    index = {(r["seed"], r["allocator"], r["relay"], r["robustification"]): r for r in rows}
    per_world = []
    for row in rows:
        per_world.append({"seed": row["seed"], "allocator": row["allocator"],
            "relay": int(row["relay"]), "robustification": int(row["robustification"]),
            **flatten_metrics(row), "byte_violations": row["metrics"]["byte_violations"],
            "floor_violations": row["metrics"]["feasible_floor_violations"],
            "solver_failure_rate": row["metrics"]["solver_failure_rate"],
            "strata_count": len(row["strata"])})
    write_csv(root/"per_world.csv", per_world)

    arm_means = {}
    for allocator in registration["allocator_levels"]:
        for relay in (False, True):
            for robust in (False, True):
                name = f"{allocator}_R{int(relay)}_B{int(robust)}"
                arm_rows = [index[seed, allocator, relay, robust] for seed in registration["seeds"]]
                arm_means[name] = {metric: summary([flatten_metrics(row)[metric] for row in arm_rows])
                                   for metric in flatten_metrics(arm_rows[0])}

    paired_rows = []; paired_summary = {}
    for baseline in ("utility_only", "v4_fallback"):
        paired_summary[baseline] = {}
        for relay in (False, True):
            for robust in (False, True):
                cell = f"R{int(relay)}_B{int(robust)}"; paired_summary[baseline][cell] = {}
                for metric in flatten_metrics(rows[0]):
                    values = []
                    for seed in registration["seeds"]:
                        value = flatten_metrics(index[seed,"v6_minimax",relay,robust])[metric] - flatten_metrics(index[seed,baseline,relay,robust])[metric]
                        values.append(value); paired_rows.append({"seed": seed, "baseline": baseline,
                            "relay": int(relay), "robustification": int(robust), "metric": metric,
                            "v6_minus_baseline": value})
                    paired_summary[baseline][cell][metric] = summary(values)
    write_csv(root/"paired_outcomes.csv", paired_rows)

    h6 = [index[s,"v6_minimax",True,True]["metrics"]["coverage_gap"] -
          .8*index[s,"utility_only",True,True]["metrics"]["coverage_gap"] for s in registration["seeds"]]
    h7 = [index[s,"v6_minimax",True,True]["metrics"]["worst_group_rmse"] -
          .9*index[s,"utility_only",True,True]["metrics"]["worst_group_rmse"] for s in registration["seeds"]]
    hypotheses = {"H6": {**summary(h6), "threshold": "gap_v6 - 0.8*gap_utility < 0",
                          "descriptive_threshold_met": float(np.mean(h6)) < 0},
                  "H7": {**summary(h7), "threshold": "worst_group_v6 - 0.9*worst_group_utility < 0",
                          "descriptive_threshold_met": float(np.mean(h7)) < 0}}

    effects = {}; effect_rows = []
    codes = {"F": lambda f,r,b: 1 if f else -1, "R": lambda f,r,b: 1 if r else -1,
             "B": lambda f,r,b: 1 if b else -1,
             "FR": lambda f,r,b: (1 if f else -1)*(1 if r else -1),
             "FB": lambda f,r,b: (1 if f else -1)*(1 if b else -1),
             "RB": lambda f,r,b: (1 if r else -1)*(1 if b else -1),
             "FRB": lambda f,r,b: (1 if f else -1)*(1 if r else -1)*(1 if b else -1)}
    for effect, code in codes.items():
        effects[effect] = {}
        for metric in flatten_metrics(rows[0]):
            values = []
            for seed in registration["seeds"]:
                value = sum(code(f,r,b)*flatten_metrics(index[seed,"v6_minimax" if f else "utility_only",r,b])[metric]
                            for f in (False,True) for r in (False,True) for b in (False,True))/4
                values.append(value); effect_rows.append({"seed":seed,"effect":effect,"metric":metric,"effect_value":value})
            effects[effect][metric] = summary(values)
    write_csv(root/"factorial_effects.csv", effect_rows)

    strata_rows = []; strata_summary = {}
    for baseline in ("utility_only", "v4_fallback"):
        strata_summary[baseline] = {}
        for name in index[registration["seeds"][0],"v6_minimax",True,True]["strata"]:
            differences = []; supported = 0; service_differences = []
            for seed in registration["seeds"]:
                left = index[seed,"v6_minimax",True,True]["strata"][name]
                right = index[seed,baseline,True,True]["strata"][name]
                both = left["supported"] and right["supported"] and left["rmse"] is not None and right["rmse"] is not None
                difference = left["rmse"]-right["rmse"] if both else None
                service = left["selected_distinct_contributors"]-right["selected_distinct_contributors"]
                supported += int(both); service_differences.append(service)
                if difference is not None: differences.append(difference)
                strata_rows.append({"seed":seed,"baseline":baseline,"stratum":name,
                    "cells":left["cells"],"both_supported":int(both),"rmse_difference":difference,
                    "selected_distinct_difference":service})
            strata_summary[baseline][name] = {"supported_pairs":supported,
                "rmse_difference": summary(differences) if differences else None,
                "selected_distinct_difference": summary(service_differences)}
    write_csv(root/"strata_paired.csv", strata_rows)

    world_results = [json.loads((root/"worlds"/str(seed)/"result.json").read_text()) for seed in registration["seeds"]]
    resource_audit = {"all_transport_checks_pass": all(route["all_checks_pass"] for world in world_results for route in world["transport_by_relay"].values()),
        "arrival_hashes_present": all(route["raw_arrivals_sha256"] for world in world_results for route in world["transport_by_relay"].values()),
        "byte_violations": sum(row["metrics"]["byte_violations"] for row in rows),
        "feasible_floor_violations": sum(row["metrics"]["feasible_floor_violations"] for row in rows if row["allocator"] != "utility_only"),
        "solver_failure_rate_max": max(row["metrics"]["solver_failure_rate"] for row in rows),
        "all_rows_have_39_strata": all(len(row["strata"]) == 39 for row in rows)}
    numerical_failures = [{"seed":row["seed"],"method":row["method"],
        "solver_failure_rate":row["metrics"]["solver_failure_rate"]}
        for row in rows if row["metrics"]["solver_failure_rate"] > 0]
    trace_pairing = [{"seed":world["seed"],"trace_hash":world["trace_hash"],
        "direct_arrivals_sha256":world["transport_by_relay"]["0"]["raw_arrivals_sha256"],
        "relay_arrivals_sha256":world["transport_by_relay"]["1"]["raw_arrivals_sha256"]}
        for world in world_results]

    v4_path = ROOT/"reports/v4_primary/core_final_7600_7629_20260903/analysis.json"
    v5_path = ROOT/"reports/v5/validation_export/integrated_fairness_diagnosis.json"
    v4 = json.loads(v4_path.read_text()); v5 = json.loads(v5_path.read_text())
    historical = {"v4_primary": {"decision":"reuse without rerun", "sha256":sha(v4_path),
        "H6":v4["primary_hypotheses"]["H6"], "H7":v4["primary_hypotheses"]["H7"],
        "fairness_reductions":v4["fairness_reductions_vs_no_fairness"]},
        "v5_validation": {"decision":"retain adverse changed-pipeline evidence", "sha256":sha(v5_path),
            "means":v5["means"]}}
    result = {"role":registration["role"], "complete_matrix":True, "worlds":len(registration["seeds"]),
        "evaluations":len(rows), "q":30, "hypotheses":hypotheses, "arm_means":arm_means,
        "paired_outcomes":paired_summary, "factorial_effects":effects,
        "strata_39":strata_summary, "resource_audit":resource_audit,
        "numerical_failures":numerical_failures, "trace_pairing":trace_pairing,
        "historical_evidence_audit":historical,
        "limitations":["descriptive validation, not primary inference", "fixed reg0.1 estimator member failed development selection", "receipt return excluded for allocator attribution", "deadline6 and direct share exogenous trace but have different policy-induced arrival maps", "39 strata are overlapping outcome audits, not independent samples"]}
    analysis_path = root/"analysis.json"; write = lambda p,v: Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write(analysis_path,result)
    artifacts = [outcomes_path, root/"per_world.csv",root/"paired_outcomes.csv",root/"factorial_effects.csv",root/"strata_paired.csv",analysis_path]
    manifest = {"role":registration["role"],"source_outcomes_sha256":sha(outcomes_path),
        "analysis_script_sha256":sha(Path(__file__)),"artifacts_sha256":{p.name:sha(p) for p in artifacts},
        "results_document_sha256":sha(ROOT/"docs/V6_FAIRNESS_VALIDATION_RESULTS.md"),
        "run_manifest_sha256":sha(root/"manifest.json"),
        "world_results_sha256":{str(seed):sha(root/"worlds"/str(seed)/"result.json") for seed in registration["seeds"]},
        "primary_authorized":False,"remaining_for_primary":"eligible frozen estimator, independent calibration, prospective power/sample-size lock, and untouched confirmation seeds"}
    write(root/"analysis_manifest.json",manifest)
    print(json.dumps({"hypotheses":hypotheses,"resource_audit":resource_audit},indent=2))


if __name__ == "__main__":
    main()

"""Registered bounded selection and paired-world inference, including failures."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t, norm
from airproof.v5_estimator import enumerate_candidates
from airproof.v5_experiment import development_specs, method_specs
from airproof.v5_estimator import EstimatorConfig


def read_complete(path):
    data = json.loads(Path(path).read_text())
    if not data["complete"] or len(data["rows"]) != data["expected_rows"]:
        raise ValueError("complete registered matrix required; no favorable-row filtering")
    rows = data["rows"]; index = {(r["seed"], r["cell"], r["method"]): r for r in rows}
    if len(index) != len(rows): raise ValueError("duplicate evaluation identities")
    stages = {r["stage"] for r in rows}
    if len(stages) != 1: raise ValueError("exactly one evidence stage required")
    stage = next(iter(stages)); seeds = sorted({r["seed"] for r in rows})
    fixed = {"development": list(range(911000,911008)), "validation": list(range(913000,913012)),
             "calibration": list(range(912000,912008))}
    if stage in fixed and seeds != fixed[stage]: raise ValueError("registered stage seed matrix missing or changed")
    if stage == "confirmation" and (not 30 <= len(seeds) <= 100 or seeds != list(range(914000,914000+len(seeds)))):
        raise ValueError("registered primary seed matrix required")
    if stage in (*fixed, "confirmation"):
        cells = ["anchor_clean","outage_clean","severe_clean","severe_drift","severe_hotspot"]
        expected = {(s,c,spec["method"]) for s in seeds for c in cells for spec in
            (development_specs() if stage == "development" else method_specs(EstimatorConfig(),c,
                factorial=stage in ("validation","confirmation")))}
        if set(index) != expected: raise ValueError("incomplete or extra registered method/cell/world matrix")
    return rows, index


def bounded_selection(rows, index):
    seeds = sorted({r["seed"] for r in rows}); results = []
    cells = ["anchor_clean", "outage_clean", "severe_clean", "severe_drift", "severe_hotspot"]
    if seeds != list(range(911000,911008)) or any(r["stage"] != "development" for r in rows):
        raise ValueError("selection requires all eight registered development worlds")
    def metric(seed, cell, method, key):
        return index[seed, cell, method]["metrics"][key]
    for candidate in enumerate_candidates():
        name = candidate.candidate_id
        control = f"SQ_{candidate.objective}_reg{candidate.regularization_multiplier:g}"
        gates = []; ratios = {}; attenuation = {}; recall_loss = {}; invalid = []
        for cell in cells:
            ap = np.array([metric(s, cell, name, "rmse") for s in seeds])
            sq = np.array([metric(s, cell, control, "rmse") for s in seeds])
            ratios[cell] = float(ap.mean()/sq.mean()) if sq.mean() > 0 else None
            recalls = [(metric(s, cell, control, "event_recall"), metric(s, cell, name, "event_recall")) for s in seeds]
            recall_loss[cell] = None if any(a is None or b is None for a,b in recalls) else float(np.mean([a-b for a,b in recalls]))
            gates.append(recall_loss[cell] is not None and recall_loss[cell] <= .05)
            for seed in seeds:
                for method in (name, control):
                    row = index[seed, cell, method]
                    if row["metrics"]["solver_failure_rate"] != 0 or row["metrics"]["feasible_floor_violations"] != 0 or row["transport"]["token_violations"] != 0:
                        invalid.append([seed,cell,method])
            if cell.endswith("clean"):
                gates.append(ratios[cell] is not None and ratios[cell] <= 1.05)
            else:
                ga = np.mean([metric(s,cell,name,"rmse")-metric(s,"severe_clean",name,"rmse") for s in seeds])
                gs = np.mean([metric(s,cell,control,"rmse")-metric(s,"severe_clean",control,"rmse") for s in seeds])
                attenuation[cell] = float(1-ga/gs) if gs > 0 else None
                gates.append(attenuation[cell] is not None and attenuation[cell] >= .2)
        eligible = all(gates) and not invalid
        results.append({"candidate": name, "estimator": asdict(candidate), "eligible": eligible,
            "clean_and_stress_rmse_ratios": ratios, "growth_attenuation": attenuation,
            "event_recall_loss": recall_loss, "invalid_evaluations": invalid,
            "worst_normalized_loss": max(v for v in ratios.values() if v is not None),
            "elapsed_seconds": sum(index[s,c,name]["elapsed_seconds"] for s in seeds for c in cells)})
    eligible = sorted((r for r in results if r["eligible"]), key=lambda r:
        (r["worst_normalized_loss"], r["estimator"]["cap"], r["elapsed_seconds"]))
    return {"selection_passed": bool(eligible), "estimator": eligible[0]["estimator"] if eligible else None,
        "candidate": eligible[0]["candidate"] if eligible else None, "candidates": results,
        "decision": "proceed-to-independent-calibration-and-validation" if eligible else "no-eligible-candidate; confirmation-blocked; preserve-failures",
        "seeds": seeds, "extra_search_permitted": False}


def contrasts(index, seeds):
    def m(s,c,method,key="rmse"): return index[s,c,method]["metrics"][key]
    answer = {}
    for i,c in enumerate(("anchor_clean","outage_clean","severe_clean"),1):
        answer[f"H{i}"] = np.array([m(s,c,"AP")-1.05*m(s,c,"SQ") for s in seeds])
    for i,c in enumerate(("severe_drift","severe_hotspot"),4):
        answer[f"H{i}"] = np.array([(m(s,c,"AP")-m(s,"severe_clean","AP"))-.8*(m(s,c,"SQ")-m(s,"severe_clean","SQ")) for s in seeds])
    for h,key,ratio in (("H6","coverage_gap",.8),("H7","worst_group_rmse",.9)):
        answer[h] = np.array([m(s,"severe_clean","AP",key)-ratio*m(s,"severe_clean","F0R1B1",key) for s in seeds])
    return answer


def infer(values):
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all():
        return {"valid": False, "n": len(values), "p": None}
    mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd/np.sqrt(len(values))
    if se == 0:
        return {"valid": False, "n": len(values), "mean_contrast": mean, "sd": sd,
                "p": None, "reason": "zero paired variance; t inference undefined"}
    p = float(t.cdf(mean/se, len(values)-1))
    return {"valid": True, "n": len(values), "mean_contrast": mean, "sd": sd,
        "p": p, "descriptive_95_ci": [mean-t.ppf(.975,len(values)-1)*se,mean+t.ppf(.975,len(values)-1)*se]}


def inference(index, seeds):
    result = {h: infer(v) for h,v in contrasts(index,seeds).items()}
    order = sorted(result, key=lambda h: result[h]["p"] if result[h]["p"] is not None else 1.)
    previous = 0.
    for rank,h in enumerate(order):
        item = result[h]; adjusted = max(previous,min(1.,(7-rank)*(item["p"] if item["p"] is not None else 1.)))
        item["holm_p"] = adjusted; item["passed"] = item["valid"] and adjusted <= .05
        previous = adjusted
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args(); rows,index = read_complete(args.input)
    if args.select:
        result = bounded_selection(rows,index)
    else:
        seeds = sorted({r["seed"] for r in rows})
        result = {"stage": sorted({r["stage"] for r in rows}), "hypotheses": inference(index,seeds),
            "scope": "confirmation only if prospectively locked complete primary; otherwise validation descriptive"}
    result["source_outcomes_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(result.get("decision", "analysis written"))


if __name__ == "__main__": main()

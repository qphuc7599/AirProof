"""Build a quantitative obstruction certificate from exposed ceiling outcomes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def summary(values):
    x=np.asarray(values,float);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    radius=float(t.ppf(.975,len(x)-1)*se)
    return {"n":len(x),"mean":mean,"descriptive_95_ci":[mean-radius,mean+radius],
            "minimum":float(x.min()),"maximum":float(x.max())}
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def perfect_distinct_allocation(caps, total):
    """Exact minimum raw-count gap with ``total`` unit-cost services and group caps.

    This deliberately relaxes byte heterogeneity: every timely distinct identity costs
    one capacity unit.  Enumerating the attained minimum and maximum counts is exact
    for this relaxed integer problem and therefore gives allocation its best case.
    """
    caps=[int(value) for value in caps]
    if total<0 or total>sum(caps):raise ValueError("infeasible distinct-service total")
    groups=range(len(caps));best=None
    for low in range(min(caps)+1):
        for high in range(low,max(caps)+1):
            objective=0. if high==0 else 1-low/high
            if best is not None and objective>=best["gap"]-1e-15:continue
            for low_group in groups:
                if caps[low_group]<low:continue
                for high_group in groups:
                    if low_group==high_group:
                        if low!=high or caps[low_group]<high:continue
                        remaining=[g for g in groups if g!=low_group];remainder=total-high
                    else:
                        if caps[high_group]<high:continue
                        remaining=[g for g in groups if g not in (low_group,high_group)]
                        remainder=total-low-high
                    if len(remaining)*low<=remainder<=sum(min(caps[g],high) for g in remaining):
                        best={"gap":objective,"minimum":low,"maximum":high}
                        break
                if best is not None and best["gap"]==objective:break
    if best is None:raise AssertionError("finite allocation oracle found no feasible allocation")
    return best


def main():
    root=ROOT/"reports/v6/fairness_ceiling_diagnostic";path=root/"outcomes.json"
    data=json.loads(path.read_text());worlds=data["worlds"]
    if not data["complete"] or len(worlds)!=12:raise ValueError("complete 12-world exposed diagnostic required")
    alpha_grid=[0,4,8,16,32]
    alpha_results={}
    for alpha in alpha_grid:
        rows=[next(x for x in w["weighted"] if x["alpha"]==alpha) for w in worlds]
        alpha_results[str(alpha)]={"worst_group_rmse":summary([r["metrics"]["worst_group_rmse"] for r in rows]),
            "rmse":summary([r["metrics"]["rmse"] for r in rows]),
            "maximum_solver_failure_rate":max(r["solver_failure_rate"] for r in rows)}
    requirements=[]
    for w in worlds:
        population=list(w["population_counts"].values());timely=list(w["timely_distinct_counts"].values())
        selected_total=sum(w["selected_distinct_counts"].values())
        capacity_oracle=perfect_distinct_allocation(timely,selected_total)
        required_ratio=1-.8*w["utility_gap"]
        required_population_min=math.ceil(required_ratio*max(population))
        required_timely_min=math.ceil(required_ratio*max(timely))
        requirements.append({"seed":w["seed"],"utility_gap":w["utility_gap"],
            "required_min_to_meet_h6":required_population_min,
            "population_min":min(population),"additional_origin_pairs_required":max(0,required_population_min-min(population)),
            "timely_min":min(timely),"additional_timely_pairs_required":max(0,required_timely_min-min(timely)),
            "population_gap":w["population_gap"],"timely_opportunity_gap":w["timely_opportunity_gap"],
            "fixed_distinct_capacity":selected_total,
            "perfect_allocation_on_timely_opportunities":capacity_oracle,
            "h6_capacity_oracle_contrast":capacity_oracle["gap"]-.8*w["utility_gap"],
            "h6_population_contrast":w["h6_population_contrast"],"h6_timely_contrast":w["h6_timely_contrast"],
            "h7_target":w["h7_required_worst_group_rmse"],
            "h7_oracle":w["oracle_best"]["metrics"]["worst_group_rmse"],
            "h7_oracle_alpha":w["oracle_best"]["alpha"],
            "h7_oracle_contrast":w["oracle_best"]["metrics"]["worst_group_rmse"]-w["h7_required_worst_group_rmse"]})
    result={"role":"quantitative obstruction certificate from exposed validation; not independent evidence",
        "decision":"do_not_open_cross_layer_policy_development: both preregistered ceiling conditions fail",
        "H6":{
            "population_serve_all_gap":summary([w["population_gap"] for w in worlds]),
            "timely_serve_all_gap":summary([w["timely_opportunity_gap"] for w in worlds]),
            "utility_based_target_gap":summary([.8*w["utility_gap"] for w in worlds]),
            "population_ceiling_contrast":summary([w["h6_population_contrast"] for w in worlds]),
            "timely_ceiling_contrast":summary([w["h6_timely_contrast"] for w in worlds]),
            "perfect_allocation_fixed_capacity_gap":summary([r["perfect_allocation_on_timely_opportunities"]["gap"] for r in requirements]),
            "perfect_allocation_fixed_capacity_contrast":summary([r["h6_capacity_oracle_contrast"] for r in requirements]),
            "all_population_contrasts_positive":all(w["h6_population_contrast"]>0 for w in worlds),
            "all_timely_contrasts_positive":all(w["h6_timely_contrast"]>0 for w in worlds),
            "all_fixed_capacity_oracle_contrasts_positive":all(r["h6_capacity_oracle_contrast"]>0 for r in requirements),
            "mean_additional_origin_pairs_required":float(np.mean([r["additional_origin_pairs_required"] for r in requirements])),
            "mean_additional_timely_pairs_required":float(np.mean([r["additional_timely_pairs_required"] for r in requirements])),
            "capacity_oracle_scope":"Exact integer optimum for distributing the realized number of distinct services over timely group caps after relaxing every identity to one capacity unit. It favors allocation by ignoring heterogeneous bytes and causal scheduling.",
            "certificate_scope":"Under no deliberate withholding, even unlimited transport/allocation of every existing origin pair cannot meet the old count-gap target. This is a monotone-service obstruction, not a bound if high-service groups may be intentionally denied."},
        "H7":{"fixed_estimator_reweighting":alpha_results,
            "posthoc_oracle_worst_group_rmse":summary([w["oracle_best"]["metrics"]["worst_group_rmse"] for w in worlds]),
            "required_worst_group_rmse":summary([w["h7_required_worst_group_rmse"] for w in worlds]),
            "oracle_contrast":summary([r["h7_oracle_contrast"] for r in requirements]),
            "oracle_selected_alpha_zero_worlds":sum(r["h7_oracle_alpha"]==0 for r in requirements),
            "certificate_scope":"Finite registered debt-weight family on fixed selected records and fixed estimator. It rejects this fairness-only reweighting mechanism; it is not an impossibility theorem for arbitrary spatial acquisition or estimator changes."},
        "per_world_requirements":requirements,
        "required_architecture_changes":[
            "H6 without withholding requires origin/recruitment/mobility coverage that increases the smallest user-group population and timely-origin pool; transport cannot create missing identities.",
            "The old raw-count coverage gap is population-composition sensitive. A population-normalized endpoint would answer a different claim and cannot replace H6 without protocol revision.",
            "H7 requires spatial-information-aware acquisition/representative construction or a materially different calibrated estimator; cumulative group-count debt and scalar record reweighting do not supply the missing error reduction.",
            "Any later cross-layer policy must keep q=30 and bytes/copies/lag fixed, register on unused development seeds, and obtain fresh validation before confirmation."],
        "source_sha256":{"ceiling_outcomes":sha(path),
            "validation_outcomes":sha(ROOT/"reports/v6/fairness_validation_v1/outcomes.json"),
            "ceiling_registration":sha(ROOT/"configs/v6/fairness_ceiling_diagnostic.json"),
            "analysis_script":sha(Path(__file__))}}
    certificate=root/"certificate.json";write(certificate,result)
    manifest={"role":result["role"],"certificate_sha256":sha(certificate),**result["source_sha256"],
        "report_source_sha256":sha(root/"report-source.md"),
        "results_document_sha256":sha(ROOT/"docs/V6_FAIRNESS_FAILURE_CEILINGS.md"),
        "confirmation_seeds_opened":False,"new_policy_development_run":False}
    write(root/"analysis_manifest.json",manifest)
    print(json.dumps({"decision":result["decision"],"H6":result["H6"],"H7_oracle":{
        "achieved":result["H7"]["posthoc_oracle_worst_group_rmse"],"required":result["H7"]["required_worst_group_rmse"],
        "contrast":result["H7"]["oracle_contrast"],"alpha_zero_worlds":result["H7"]["oracle_selected_alpha_zero_worlds"]}},indent=2))


if __name__=="__main__":main()

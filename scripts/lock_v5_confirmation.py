"""Prospective sample-size/runtime gate. Cannot turn failed validation into primary."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t, nct
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airproof.config import load_config, config_hash
from airproof.v5_estimator import EstimatorConfig
from dataclasses import asdict
from analyze_v5_campaign import read_complete, contrasts
from run_v5_campaign import source_inventory, write_json


def required_worlds(paired, target=.9):
    powers = {}
    for n in range(30,101):
        powers = {}
        for name, values in paired.items():
            mu, sd = float(np.mean(values)), float(np.std(values,ddof=1))
            if mu >= 0 or sd <= 0 or not np.isfinite([mu,sd]).all():
                powers[name] = None
            else:
                powers[name] = float(nct.cdf(t.ppf(.05/7,n-1),n-1,mu*np.sqrt(n)/sd))
        if all(p is not None and p >= target for p in powers.values()): return n,powers
    return None,powers


def main():
    p = argparse.ArgumentParser(); p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4); args=p.parse_args()
    if not 1 <= args.workers <= 11: p.error("workers must be1..11")
    selected=json.loads(args.selection.read_text()); reasons=[]
    if not selected.get("selection_passed"): reasons.append("no eligible development candidate")
    rows,index=read_complete(args.validation/"outcomes.json")
    expected_estimator=asdict(EstimatorConfig(**selected["estimator"])) if selected.get("estimator") else None
    if any(r["estimator"] != expected_estimator for r in rows if r["method"] == "AP"):
        raise ValueError("validation estimator does not match selected estimator")
    validation_manifest=json.loads((args.validation/"manifest.json").read_text())
    if validation_manifest["selected"] != expected_estimator or any(r["source_hash"] != validation_manifest["source_hash"] for r in rows):
        raise ValueError("validation manifest/source binding mismatch")
    seeds=sorted({r["seed"] for r in rows})
    if seeds != list(range(913000,913012)) or any(r["stage"]!="validation" for r in rows):
        raise ValueError("complete12world validation stage required")
    paired=contrasts(index,seeds); n,powers=required_worlds(paired)
    if n is None: reasons.append("90 percent prospective power not attainable in30..100worlds with validation effects")
    if any(np.mean(v)>=0 for v in paired.values()): reasons.append("one or more unchanged H1-H7 targets fail on validation mean")
    for r in rows:
        if r["metrics"]["solver_failure_rate"] or r["metrics"]["feasible_floor_violations"] or r["transport"]["token_violations"]:
            reasons.append("numerical/resource correctness violation"); break
    for cell in ("anchor_clean","outage_clean","severe_clean","severe_drift","severe_hotspot"):
        pairs=[(index[s,cell,"SQ"]["metrics"]["event_recall"],index[s,cell,"AP"]["metrics"]["event_recall"]) for s in seeds]
        if any(a is None or b is None or not np.isfinite([a,b]).all() for a,b in pairs):
            reasons.append(f"invalid event recall support:{cell}")
        elif np.mean([a-b for a,b in pairs])>.05: reasons.append(f"event recall target failed:{cell}")
    runtime_records=[(f,json.loads(f.read_text())) for f in (args.validation/"worlds").glob("*.json")]
    times=[record["elapsed_seconds"] for _,record in runtime_records]
    if any(not r.get("runtime_from_complete_fresh_world",True) for _,r in runtime_records):
        reasons.append("partial-resume runtimes require independent full-world runtime preflight")
    if {int(f.stem) for f,_ in runtime_records}!=set(seeds) or {r["seed"] for _,r in runtime_records}!=set(seeds):
        raise ValueError("runtime seed identities do not match validation")
    if len(times)!=12 or any(not np.isfinite(v) or v<=0 for v in times):
        raise ValueError("all12 validated runtime records required")
    projected=None if n is None else max(times)*n/args.workers*1.5
    protocol=load_config("configs/v5/protocol.yaml")
    base=load_config(protocol["historical_configuration"])
    if validation_manifest["protocol_hash"]!=config_hash(protocol) or validation_manifest["base_hash"]!=config_hash(base):
        raise ValueError("protocol/base changed after validation")
    primary_end=datetime.fromisoformat(protocol["created_utc"].replace("Z","+00:00"))+timedelta(hours=25)
    now=datetime.now(timezone.utc)
    if projected is not None and projected>(primary_end-now).total_seconds(): reasons.append("projected primary exceeds hour25 boundary")
    inventory,source_hash=source_inventory()
    numerical={k:v for k,v in inventory.items() if k.startswith("airproof/")}
    validated_numerical={k:v for k,v in validation_manifest["source_files"].items() if k.startswith("airproof/")}
    if numerical!=validated_numerical: raise ValueError("numerical source changed after validation")
    result={**selected,"validation_means":{h:float(v.mean()) for h,v in paired.items()},
        "prospective_powers":powers,"required_worlds":n,"projected_seconds":projected,
        "lock_utc":now.isoformat(),"blocked_reasons":reasons,"confirmation_lock":None}
    if not reasons:
        result["confirmation_lock"]={"seeds":list(range(914000,914000+n)),"worlds":n,
            "source_hash":source_hash,"source_files":inventory,"workers":args.workers,
            "protocol_hash":config_hash(protocol),"base_hash":config_hash(base),
            "primary_end_utc":primary_end.isoformat(),"selection_frozen":True,"power_target":.9}
    write_json(args.output,result)
    print("BLOCKED: "+"; ".join(reasons) if reasons else f"LOCKED {n} worlds")


if __name__=="__main__":main()

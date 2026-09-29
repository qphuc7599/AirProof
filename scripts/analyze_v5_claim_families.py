"""Secondary registered families; paired-world contrasts and no favorable subsets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_v5_campaign import read_complete, infer

CELLS=("anchor_clean","outage_clean","severe_clean","severe_drift","severe_hotspot")


def holm(values):
    result={name:infer(value) for name,value in values.items()}
    order=sorted(result,key=lambda name: result[name]["p"] if result[name]["p"] is not None else 1.)
    previous=0.; family_valid=all(r["valid"] for r in result.values())
    for rank,name in enumerate(order):
        row=result[name]; adjusted=max(previous,min(1.,(len(result)-rank)*(row["p"] if row["p"] is not None else 1.)))
        row["holm_p"]=adjusted; row["passed"]=family_valid and adjusted<=.05
        previous=adjusted
    return {"valid":family_valid,"tests":result,"alpha":.05,"adjustment":"Holm within family"}


def analyze(campaign, release):
    rows,index=read_complete(campaign/"outcomes.json")
    seeds=sorted({r["seed"] for r in rows}); families={}
    def m(s,c,method,key): return index[s,c,method]["metrics"][key]
    families["citizen_value"]=holm({c:[m(s,c,"AP","rmse")-m(s,c,"PUBLIC","rmse") for s in seeds] for c in CELLS})
    wrapper={"clean:"+c:[m(s,c,"AP","rmse")-1.05*m(s,c,"SQ","rmse") for s in seeds] for c in CELLS[:3]}
    for c in CELLS:
        wrapper["event:"+c]=[np.nan if m(s,c,"SQ","event_recall") is None or m(s,c,"AP","event_recall") is None else
            m(s,c,"SQ","event_recall")-m(s,c,"AP","event_recall")-.05 for s in seeds]
    families["wrapper_and_events"]=holm(wrapper)
    release_rows=json.loads((release/"world_metrics.json").read_text())
    ri={(r["seed"],r["cell"],r["method"],r["clock"],r["support"]):r for r in release_rows}
    if len(ri)!=len(release_rows): raise ValueError("duplicate release world identity")
    values={}
    for c in CELLS:
        for clock in ("live","reconstructed"):
            for public in ("PUBLIC","PUBLIC_CALIBRATED"):
                values[f"{c}:{clock}:{public}"]=[
                    np.sqrt(ri[s,c,"CALIBRATED",clock,"all"]["mse"])-np.sqrt(ri[s,c,public,clock,"all"]["mse"])
                    for s in seeds]
    families["release_citizen_value"]=holm(values)
    return {"stage":rows[0]["stage"],"seeds":seeds,"families":families,
        "scope":"confirmatory only for prospectively locked complete primary; validation otherwise",
        "protocol_sha256":hashlib.sha256(Path("configs/v5/claim_families.yaml").read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--campaign",type=Path,required=True)
    p.add_argument("--release",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); result=analyze(args.campaign,args.release)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()

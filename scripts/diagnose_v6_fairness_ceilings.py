"""Exposed H6/H7 ceilings on the completed validation worlds."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airproof.config import config_hash, load_config
from airproof.simulator import generate_world
from airproof.v5_experiment import cell_configuration
from airproof.v6_fairness_validation import allocate_shared_arrivals, collector_view
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_numerical_experiment import evaluate, prepare_public
from airproof.v6_transport import simulate_transport


ROOT = Path(__file__).resolve().parents[1]
FILES = ("configs/v6/fairness_ceiling_diagnostic.json","configs/v6/fairness_validation.json",
    "configs/v6/protocol.yaml","airproof/v6_fairness.py","airproof/v6_fairness_validation.py",
    "airproof/v6_estimator.py","airproof/v6_numerical_experiment.py","airproof/v6_transport.py",
    "airproof/v6_mobility.py","airproof/simulator.py","scripts/diagnose_v6_fairness_ceilings.py")


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")


def gap(counts):
    values=list(counts.values()); maximum=max(values,default=0)
    return 0. if maximum==0 else 1-min(values)/maximum


def job(seed, output, source_hash):
    started=time.perf_counter();output=Path(output);target=output/"worlds"/str(seed);existing=target/"result.json"
    registration=json.loads((ROOT/"configs/v6/fairness_ceiling_diagnostic.json").read_text())
    prior=json.loads((ROOT/"reports/v6/fairness_validation_v1/outcomes.json").read_text())
    utility=next(r for r in prior["rows"] if r["seed"]==seed and r["allocator"]=="utility_only" and r["relay"] and r["robustification"])
    base=load_config(ROOT/"reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml")
    protocol=load_config(ROOT/"configs/v6/protocol.yaml");base["transport"]=protocol["transport"]|{"drain_epochs":24}
    cfg=cell_configuration(base,registration["cell"])
    identity=config_hash({"seed":seed,"registration":registration,"source":source_hash,"config":cfg})
    if existing.exists():
        old=json.loads(existing.read_text())
        if old["identity"]!=identity:raise ValueError("identity mismatch")
        return old
    try:
        world=generate_world(cfg,seed);trace,_=coupled_public_trace(cfg,seed,world.observations)
        transport=simulate_transport(trace,world.observations,cfg,registration["relay_policy"])
        allocation=allocate_shared_arrivals(world,transport.raw_arrivals,cfg,"v6_minimax")
        burn=int(cfg["world"]["burn_in_steps"]);lag=int(cfg["twin"]["fixed_lag"])
        populations={g:len({r.user_id for r in world.observations if r.group==g}) for g in range(cfg["world"]["groups"])}
        timely={g:len({r.user_id for r in world.observations if r.group==g and r.epoch>=burn
            and (a:=transport.raw_arrivals.get(r.nullifier)) is not None and r.epoch<=a<=r.epoch+lag})
            for g in populations}
        selected={g:len({r.user_id for r in allocation.selected if r.group==g and r.epoch>=burn}) for g in populations}
        rates={g:selected[g]/populations[g] if populations[g] else 0. for g in populations}
        positive=[v for v in rates.values() if v>0];minimum=min(positive,default=1.)
        prepared=prepare_public(world,cfg);weighted=[]
        for alpha in registration["record_reweighting"]["alpha_grid"]:
            factors={g:(minimum/rates[g])**alpha if rates[g]>0 else 0. for g in rates}
            records=[replace(r,quality=r.quality*factors[r.group]) for r in allocation.selected]
            view=collector_view(allocation);view.selected=records
            row,_=evaluate(world,cfg,prepared,view,f"debt_weight_alpha{alpha}",.1,robust=True)
            weighted.append({"alpha":alpha,"factors":factors,"metrics":row["metrics"],
                "solver_failure_rate":row["metrics"]["solver_failure_rate"]})
        best=min(weighted,key=lambda x:x["metrics"]["worst_group_rmse"])
        record={"status":"complete","identity":identity,"seed":seed,"source_hash":source_hash,
            "population_counts":populations,"timely_distinct_counts":timely,"selected_distinct_counts":selected,
            "population_gap":gap(populations),"timely_opportunity_gap":gap(timely),"selected_gap":gap(selected),
            "utility_gap":utility["metrics"]["coverage_gap"],"utility_worst_group_rmse":utility["metrics"]["worst_group_rmse"],
            "h6_population_contrast":gap(populations)-.8*utility["metrics"]["coverage_gap"],
            "h6_timely_contrast":gap(timely)-.8*utility["metrics"]["coverage_gap"],
            "h7_required_worst_group_rmse":.9*utility["metrics"]["worst_group_rmse"],
            "weighted":weighted,"oracle_best":best,"runtime_seconds":time.perf_counter()-started}
    except Exception:
        record={"status":"failed","identity":identity,"seed":seed,"source_hash":source_hash,
            "error":traceback.format_exc(),"runtime_seconds":time.perf_counter()-started}
    write(existing,record);return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="reports/v6/fairness_ceiling_diagnostic")
    parser.add_argument("--workers",type=int,default=4);args=parser.parse_args()
    registration=json.loads((ROOT/"configs/v6/fairness_ceiling_diagnostic.json").read_text())
    inventory={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES};source_hash=config_hash(inventory)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);manifest=output/"manifest.json"
    record={"created_utc":datetime.now(timezone.utc).isoformat(),"role":registration["role"],
        "registration":registration,"source_hash":source_hash,"source_files":inventory,
        "expected_worlds":len(registration["seeds"]),"expected_solves":len(registration["seeds"])*len(registration["record_reweighting"]["alpha_grid"])}
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old["source_hash"]!=source_hash:raise SystemExit("source changed; preserve existing directory")
    else:
        write(manifest,record)
        for name in FILES:
            dst=output/"source_snapshot"/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(job,seed,str(output),source_hash) for seed in registration["seeds"]]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            write(output/"progress.json",{"finished":len(results),"expected":len(futures),
                "failures":sum(r["status"]!="complete" for r in results)})
            print(json.dumps({"seed":result["seed"],"status":result["status"],"seconds":result["runtime_seconds"]}),flush=True)
    complete=len(results)==len(registration["seeds"]) and all(r["status"]=="complete" for r in results)
    write(output/"outcomes.json",{"complete":complete,"worlds":sorted(results,key=lambda x:x["seed"]),"source_hash":source_hash})
    if not complete:raise SystemExit(1)


if __name__=="__main__":main()

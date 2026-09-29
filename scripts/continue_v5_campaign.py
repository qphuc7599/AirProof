"""Fixed sequential stage controller. Scientific failures never open extra search."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--development",type=Path,required=True)
    p.add_argument("--output",type=Path,default=ROOT/"reports/v5/selected_campaign_v1")
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    if not 1<=args.workers<=11: p.error("workers must be1..11")
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    protocol=yaml.safe_load((ROOT/"configs/v5/protocol.yaml").read_text())
    deadline=datetime.fromisoformat(protocol["deadline_utc"].replace("Z","+00:00"))
    state={"created_utc":datetime.now(timezone.utc).isoformat(),"stages":{},"primary":"not-started"}
    def save():
        (out/"controller_status.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
    def run(name,arguments):
        if state["stages"].get(name)=="complete": return
        state["active"]=name; state["stages"][name]="running"; save()
        print(f"START {name}",flush=True)
        with (out/f"{name}.log").open("w",encoding="utf-8") as log:
            remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
            if remaining<=0: raise RuntimeError("29.5 hour execution boundary reached")
            result=subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
        state["stages"][name]="complete" if result.returncode==0 else "execution-failed"
        save(); print(f"END {name} code={result.returncode}",flush=True)
        if result.returncode: raise RuntimeError(f"{name} failed; see retained log; no seed substitution")
    # This controller owns future stages only; existing development has its own lock.
    try:
        state["active"]="waiting-complete-development"; save()
        while not (args.development/"outcomes.json").exists():
            if datetime.now(timezone.utc)>=deadline: raise RuntimeError("execution boundary reached waiting for development")
            time.sleep(5)
        selection=args.development/"selection.json"
        run("selection",["scripts/analyze_v5_campaign.py",args.development/"outcomes.json","--select","--output",selection])
        chosen=json.loads(selection.read_text())
        if not chosen["selection_passed"]:
            state.update(active=None,primary="blocked-no-eligible-candidate",scientific_goal="not-attained",
                dependent_evaluations="calibration/validation/EPA/stress/primary not launched without qualifying selected estimator")
            save(); return
        plain=out/"selected_estimator.json"; plain.write_text(json.dumps(chosen["estimator"],indent=2))
        common=["--workers",args.workers,"--selected-config",selection,"--save-predictions"]
        cal=out/"calibration"; val=out/"validation"
        run("calibration",["scripts/run_v5_campaign.py","--stage","calibration","--output",cal,*common])
        intervals=out/"uncertainty_calibration"; releases=out/"release_calibration"
        run("freeze_intervals",["scripts/analyze_v5_uncertainty.py","freeze","--campaign",cal,
            "--selected-manifest",selection,"--training-manifest",args.development/"manifest.json","--output-dir",intervals])
        run("freeze_release",["scripts/analyze_v5_release_twin.py","freeze-dense","--campaign",cal,"--output-dir",releases])
        run("validation",["scripts/run_v5_campaign.py","--stage","validation","--output",val,*common])
        run("validation_analysis",["scripts/analyze_v5_campaign.py",val/"outcomes.json","--output",out/"validation_analysis.json"])
        run("validation_intervals",["scripts/analyze_v5_uncertainty.py","evaluate","--campaign",val,
            "--frozen",intervals/"frozen_quantiles.json","--output-dir",out/"validation_intervals"])
        run("validation_release",["scripts/analyze_v5_release_twin.py","analyze","--campaign",val,
            "--calibration",releases/"frozen_calibration.npz","--calibration-manifest",releases/"manifest.json",
            "--output-dir",out/"validation_release"])
        run("validation_claim_families",["scripts/analyze_v5_claim_families.py","--campaign",val,
            "--release",out/"validation_release","--output",out/"validation_claim_families.json"])
        lock=out/"confirmation_lock.json"
        run("power_runtime_lock",["scripts/lock_v5_confirmation.py","--selection",selection,
            "--validation",val,"--workers",args.workers,"--output",lock])
        locked=json.loads(lock.read_text())
        if locked["confirmation_lock"]:
            primary=out/"confirmation"
            state["primary"]="prospectively-locked"; save()
            run("confirmation",["scripts/run_v5_campaign.py","--stage","confirmation","--output",primary,
                "--workers",args.workers,"--selected-config",lock,"--save-predictions","--seeds",
                *locked["confirmation_lock"]["seeds"]])
            run("confirmation_analysis",["scripts/analyze_v5_campaign.py",primary/"outcomes.json",
                "--output",out/"confirmation_analysis.json"])
            run("confirmation_intervals",["scripts/analyze_v5_uncertainty.py","evaluate","--campaign",primary,
                "--frozen",intervals/"frozen_quantiles.json","--output-dir",out/"confirmation_intervals"])
            run("confirmation_release",["scripts/analyze_v5_release_twin.py","analyze","--campaign",primary,
                "--calibration",releases/"frozen_calibration.npz","--calibration-manifest",releases/"manifest.json",
                "--output-dir",out/"confirmation_release"])
            run("confirmation_claim_families",["scripts/analyze_v5_claim_families.py","--campaign",primary,
                "--release",out/"confirmation_release","--output",out/"confirmation_claim_families.json"])
            state["primary"]="complete-no-retuning"
        else:
            state["primary"]="blocked-validation-or-power"; state["primary_blockers"]=locked["blocked_reasons"]
        # All remaining scored tasks use the same locked development choice.
        run("epa",["scripts/benchmark_v5_epa.py","--score","--selected-config",plain])
        run("stress",["scripts/benchmark_v5_stress.py","--score","--selected-config",plain,
            "--output",ROOT/"reports/v5/stress_registered_v1"])
        state["active"]=None; state["scientific_goal"]="requires-final-gate-review"; save()
    except Exception as error:
        state["error"]=repr(error); save(); raise


if __name__=="__main__":main()

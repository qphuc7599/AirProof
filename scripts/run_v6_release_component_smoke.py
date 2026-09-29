"""Deterministic B3 component smoke; not a scientific development experiment."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from airproof.continual_release import FixedReleasePlan
from airproof.v6_release_experiment import fit_release_controls, run_release_controls, score_controls


def fixture(phase):
    time=np.arange(112.)
    public=(10+.1*np.sin(time+phase))[:,None]
    truth=public+.2*np.cos(time[:,None]/4+phase)
    query=np.full_like(public,np.nan)
    slots=FixedReleasePlan(112,1,20,8/28,0,8).scheduled_epochs
    query[list(slots)]=truth[list(slots)]
    mask=np.isfinite(query); noisy=query.copy(); noisy[mask]+=.3
    return dict(public=public,actual_query=query,noisy_values=noisy,released_mask=mask),truth

if __name__ == '__main__':
    train,labels=fixture(0.)
    model=fit_release_controls(**train,calibration_truth=labels,calibration_id='engineering-fixture-A')
    test,truth=fixture(.7)
    result=run_release_controls(model,**test,evaluation_id='engineering-fixture-B')
    print(json.dumps({'scope':'deterministic engineering smoke; fixed offset is not a Laplace sample; no scientific gain or DP experiment',
                      'metadata':result['metadata'],'scores':score_controls(result,truth)},indent=2))

"""Historical matched availability-vs-protected-value release analysis."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.stats import t as student_t
from airproof.v6_release_value import release_age_features,ridge_correction

ROOT=Path('.');SOURCE=Path('reports/v5/selected_campaign_v1');OUT=Path('reports/v6/release_incremental_value')
CELLS=('anchor_clean','outage_clean','severe_clean','severe_drift','severe_hotspot')
CAL=range(912000,912008);EVAL=range(913000,913012)


def load(role,seed,cell):
    path=SOURCE/f'{role}/jobs/{seed}/{cell}/release.npz'
    with np.load(path) as z:return path,{k:z[k] for k in z.files}


def designs(data,noisy_center,query_center):
    scheduled=np.isfinite(data['residual_query']).all(1);mask=data['residual_mask'].astype(bool)
    actual=np.full_like(data['residual_query'],np.nan);actual[mask]=data['residual_query'][mask]
    available,noisy,_=release_age_features(data['residual'],mask,scheduled,centers=noisy_center)
    available2,oracle,_=release_age_features(actual,mask,scheduled,centers=query_center)
    assert np.array_equal(available,available2)
    return available,noisy,oracle,scheduled


def main():
    OUT.mkdir(exist_ok=False);inputs={};cal=[]
    for seed in CAL:
        for cell in CELLS:
            path,data=load('calibration',seed,cell);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();cal.append((seed,cell,data))
    noisy_center=np.array([np.mean(np.concatenate([d['residual'][d['residual_mask'].astype(bool)[:,g],g] for _,_,d in cal])) for g in range(4)])
    query_center=np.array([np.mean(np.concatenate([d['residual_query'][d['residual_mask'].astype(bool)[:,g],g] for _,_,d in cal])) for g in range(4)])
    # Strong matched public regression, fit once on calibration labels.
    public_x=np.concatenate([np.column_stack((np.ones(len(d['baseline'])),d['baseline'])) for _,_,d in cal])
    public_y=np.concatenate([d['truth'] for _,_,d in cal])
    public_coef=np.linalg.solve(public_x.T@public_x/len(public_x)+.1*np.eye(public_x.shape[1]),public_x.T@public_y/len(public_x))
    availability=[];noisy=[];oracle=[];target=[]
    for _,_,data in cal:
        a,n,q,_=designs(data,noisy_center,query_center);p0=np.maximum(np.column_stack((np.ones(len(data['baseline'])),data['baseline']))@public_coef,0)
        availability.append(a);noisy.append(n);oracle.append(q);target.append((data['truth']-p0).ravel())
    a=np.concatenate(availability);n=np.concatenate(noisy);q=np.concatenate(oracle);y=np.concatenate(target)
    numeric_scale=np.maximum(n.std(0),1e-6);oracle_scale=np.maximum(q.std(0),1e-6)
    schedule_coef=ridge_correction(y,a,ridge=.1)
    noisy_coef=ridge_correction(y,np.column_stack((a,n/numeric_scale)),ridge=.1)
    oracle_coef=ridge_correction(y,np.column_stack((a,q/oracle_scale)),ridge=.1)
    source_paths=[Path(__file__),Path('airproof/v6_release_value.py')]
    registration={'role':'historical reanalysis; calibration and evaluation caches already exposed; not confirmation',
      'pillar':'whole-history DP protected release contributes value beyond matched strong public and public availability',
      'hypothesis':'age-conditioned protected numeric value improves over same-schedule availability control at publication clock',
      'calibration_seeds':list(CAL),'evaluation_seeds':list(EVAL),'cells':CELLS,'ridge':.1,'age_bins_hours':[[0,6],[6,12],[12,24]],
      'controls':['strong public regression','public plus release availability/age','availability plus protected value','availability plus noiseless value oracle'],
      'schedule':'actual fixed28 query slots and actual suppression masks; acquisition+24 publication; burn48',
      'selection':'one registered mechanism; all cell outcomes retained; no parameter search',
      'privacy':'deployable value arm consumes only public baseline, schedule/mask and already protected noisy release; post-processing adds no epsilon',
      'claim_limits':'exposed historical worlds; oracle is nondeployable; linear family failure is not a no-information theorem',
      'input_hashes':inputs,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    (OUT/'registration.json').write_text(json.dumps(registration,indent=2))
    np.savez_compressed(OUT/'calibration.npz',public_coefficients=public_coef,noisy_centers=noisy_center,query_centers=query_center,
        numeric_scales=numeric_scale,oracle_scales=oracle_scale,schedule_coefficients=schedule_coef,
        protected_coefficients=noisy_coef,oracle_coefficients=oracle_coef)
    rows=[];started=time.perf_counter()
    for seed in EVAL:
        for cell in CELLS:
            path,data=load('validation',seed,cell);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            av,nv,qv,scheduled=designs(data,noisy_center,query_center)
            p0=np.maximum(np.column_stack((np.ones(len(data['baseline'])),data['baseline']))@public_coef,0);shape=p0.shape
            preds={'public':p0,'availability':np.maximum(p0+(av@schedule_coef).reshape(shape),0),
                   'protected':np.maximum(p0+(np.column_stack((av,nv/numeric_scale))@noisy_coef).reshape(shape),0),
                   'noiseless_oracle':np.maximum(p0+(np.column_stack((av,qv/oracle_scale))@oracle_coef).reshape(shape),0)}
            scores={name:float(np.sqrt(np.mean((pred[48:]-data['truth'][48:])**2))) for name,pred in preds.items()}
            rows.append({'seed':seed,'cell':cell,'scores':scores,'release_slots':int(scheduled.sum()),'released_values':int(data['residual_mask'].sum())})
            np.savez_compressed(OUT/f'{seed}_{cell}.npz',truth=data['truth'],**preds)
    summary={}
    for cell in CELLS:
        group=[r for r in rows if r['cell']==cell];means={arm:float(np.mean([r['scores'][arm] for r in group])) for arm in rows[0]['scores']}
        delta=np.array([r['scores']['protected']-r['scores']['availability'] for r in group]);se=delta.std(ddof=1)/np.sqrt(len(delta));half=float(student_t.ppf(.975,len(delta)-1)*se)
        summary[cell]={'mean_world_rmse':means,'protected_minus_availability':float(delta.mean()),
          'descriptive_95ci':[float(delta.mean()-half),float(delta.mean()+half)],'protected_better_worlds':int((delta<0).sum()),
          'noiseless_oracle_minus_availability':float(np.mean([r['scores']['noiseless_oracle']-r['scores']['availability'] for r in group])),'worlds':len(group)}
    registration['input_hashes']=inputs;registration['elapsed_seconds']=time.perf_counter()-started
    (OUT/'registration.json').write_text(json.dumps(registration,indent=2));(OUT/'world_metrics.json').write_text(json.dumps(rows,indent=2))
    result={'role':registration['role'],'summary':summary,'citizen_value_confirmed':False,'confirmation_authorized':False}
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

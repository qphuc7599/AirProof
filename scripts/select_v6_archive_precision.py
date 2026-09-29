"""Bounded exposed-validation check of prior/likelihood precision balance."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from dataclasses import asdict
import json,hashlib
import numpy as np
from airproof.predictor_wrapper import synthetic_citizen_replay
from airproof.v6_estimator import EstimatorConfig,estimate_public_field


def main():
    out=Path('reports/v6/archive_precision_validation');out.mkdir(exist_ok=False)
    source=Path('reports/v6/public_dynamics_development')
    selected=json.loads((source/'selection_lock.json').read_text())['selected']
    if selected is None:raise ValueError('public model not selected')
    prediction=source/(selected+'_validation.npz')
    truthfile=Path('reports/v6/backbone_validation_recovery/public_validation_predictions.npz')
    old=Path('reports/v4_validation/esntnn_purpleair_7800/heldout_predictions.npz')
    multipliers=[.01,.03,.1,1.]
    registration={'role':'separately registered exposed validation; no independent confirmation',
        'reason':'archive clean failure despite public-center improvement; measure prior/likelihood balance',
        'public_model':selected,'multipliers':multipliers,'base_lambdas':[.05,.05,.02],
        'fixed':{'time_step':4,'lag':1,'huber_delta':1.345,'cap':8,'scale':3,'drain':6},
        'seeds':[916000,916001],'kinds':['clean','adversarial_drift','negative_bias'],
        'targets':{'clean_ratio':1.05,'attack_attenuation':.2,'event_recall_loss':.05},
        'selection':'all paired SQ controls retained; require all gates then minimize clean RMSE; tie lower multiplier',
        'test_policy':'do not evaluate test in this runner; no margin/seed/cap changes',
        'event_threshold':'frozen public-model training prefix95th percentile',
        'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),prediction,truthfile,old,Path('airproof/v6_estimator.py'),Path('airproof/predictor_wrapper.py')]}}
    (out/'registration.json').write_text(json.dumps(registration,indent=2))
    with np.load(prediction) as z:public=z['prediction'].copy()
    with np.load(truthfile) as z:truth=z['truth'].copy()
    with np.load(old) as z:xy=z['coordinates_lat_lon'].copy()
    xy=(xy-xy.mean(0))*[111.2,111.2*np.cos(np.deg2rad(xy[:,0].mean()))]
    threshold=json.loads((source/'selection_lock.json').read_text())['event_threshold']
    events=truth>=threshold;extended=np.concatenate((public,np.repeat(public[-1:],6,axis=0)))
    rows=[]
    with (out/'results.jsonl').open('w') as stream:
        for seed in registration['seeds']:
            for kind in registration['kinds']:
                records=synthetic_citizen_replay(truth,seed=seed,kind=kind)
                for multiplier in multipliers:
                    for robust in (True,False):
                        cfg=EstimatorConfig(lambda_zero=.05*multiplier,lambda_temporal=.05*multiplier,
                            lambda_spatial=.02*multiplier,lag=1,time_step=4,loss='huber' if robust else 'quadratic',
                            output_cap=robust,numerical_refinements=2)
                        result=estimate_public_field(extended,xy,records,cfg,innovation_scales=3.)
                        live=result.live[:len(truth)];recon=result.reconstructed[:len(truth)]
                        row={'seed':seed,'kind':kind,'multiplier':multiplier,'robust':robust,
                            'rmse':float(np.sqrt(np.mean((recon-truth)**2))),
                            'event_recall':float(np.mean(live[events]>=threshold)) if events.any() else None,
                            'failure_rate':result.diagnostics['solver_failure_rate'],'config':asdict(cfg)}
                        rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({'seed':seed,'kind':kind,'rows':len(rows)}),flush=True)
    candidates=[]
    for multiplier in multipliers:
        group=[r for r in rows if r['multiplier']==multiplier]
        mean=lambda kind,robust,metric:float(np.mean([r[metric] for r in group if r['kind']==kind and r['robust']==robust]))
        ratio=mean('clean',True,'rmse')/mean('clean',False,'rmse')
        failed=[]
        if ratio>1.05:failed.append('clean')
        if any(r['failure_rate']>0 for r in group):failed.append('numerical')
        attacks={}
        for kind in registration['kinds'][1:]:
            ap=mean(kind,True,'rmse')-mean('clean',True,'rmse');sq=mean(kind,False,'rmse')-mean('clean',False,'rmse')
            attacks[kind]=1-ap/sq if sq>0 else None
            if sq<=0 or ap>.8*sq:failed.append(kind)
        for kind in registration['kinds']:
            if mean(kind,False,'event_recall')-mean(kind,True,'event_recall')>.05:failed.append(kind+'_event')
        candidates.append({'multiplier':multiplier,'clean_ratio':ratio,'clean_rmse':mean('clean',True,'rmse'),
            'attack_attenuation':attacks,'failures':failed})
    eligible=[c for c in candidates if not c['failures']]
    choice=min(eligible,key=lambda c:(c['clean_rmse'],c['multiplier'])) if eligible else None
    result={'rows':len(rows),'expected_rows':48,'candidates':candidates,'selected':choice,
        'event_support':int(events.sum()),'confirmation_authorized':False}
    (out/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

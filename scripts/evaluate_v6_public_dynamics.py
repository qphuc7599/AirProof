"""Four declared causal public models; archive remains exposed development."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,hashlib
import numpy as np
import rdata
from airproof.v6_public_dynamics import fit_public_increment
from airproof.v6_feasibility import cap_rmse_floor


def main():
    out=Path('reports/v6/public_dynamics_development');out.mkdir(exist_ok=False)
    datafile=Path('data/external/baselines/esntnn_author/Data/APFour.RData')
    old=Path('reports/v4_validation/esntnn_purpleair_7800/heldout_predictions.npz')
    cal=Path('reports/v6/backbone_validation_recovery/public_validation_predictions.npz')
    family=[{'id':f'increment_r{r}_w{w}','ridge':r,'event_weight':w} for r in (.1,1.) for w in (1.,2.)]
    paths=[Path(__file__),Path('airproof/v6_public_dynamics.py'),datafile,old,cal]
    registration={'role':'new registered model development on already exposed archive, not independent confirmation',
        'original_contribution':'robust event-preserving public-referenced twin',
        'reason':'matched bounded wrapper fails clean; affine center damages event recall',
        'fit':[0,638],'validation':[638,775],'exposed_test':[775,912],
        'family':family,'features':'last level, two increments, lag6 difference, four-neighbor contrast/increment, global median',
        'forecast_input':'public archive through t-1 only, matching frozen backbone information contract',
        'selection':'validation RMSE<=1.05 frozen backbone and recall>=backbone-.05; then lowest RMSE, tie by id',
        'event_threshold':'training prefix95th percentile; frozen before validation/test',
        'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'cap':8,'backbone_retrained':False,'confirmation_authorized':False}
    (out/'registration.json').write_text(json.dumps(registration,indent=2))
    data=np.asarray(rdata.read_rda(str(datafile))['newpoll'],float).T
    with np.load(old) as z:xy=z['coordinates_lat_lon'];test_public=z['ESN_TNN'].copy()
    dist=((xy[:,None,:]-xy[None,:,:])**2).sum(2);np.fill_diagonal(dist,np.inf)
    neighbors=np.argsort(dist,axis=1)[:,:4]
    threshold=float(np.quantile(data[:638],.95))
    with np.load(cal) as z:validation_public=z['prediction'].copy()
    def metrics(pred,truth):
        events=truth>=threshold
        return {'rmse':float(np.sqrt(np.mean((pred-truth)**2))),
                'recall':float(np.mean(pred[events]>=threshold)) if events.any() else None,
                'event_support':int(events.sum()),'cap8_floor':cap_rmse_floor(truth,pred,8)}
    baseline=metrics(validation_public,data[638:775]);models={};validation=[]
    for candidate in family:
        model=fit_public_increment(data[:638],neighbors,ridge=candidate['ridge'],event_weight=candidate['event_weight'])
        pred=np.array([model.predict_next(data[:t]) for t in range(638,775)])
        m=metrics(pred,data[638:775]);eligible=m['rmse']<=1.05*baseline['rmse'] and m['recall']>=baseline['recall']-.05
        validation.append({'id':candidate['id'],'metrics':m,'eligible':eligible});models[candidate['id']]=model
        np.savez_compressed(out/(candidate['id']+'_validation.npz'),prediction=pred,coefficients=model.coefficients,mean=model.mean,scale=model.scale,neighbors=neighbors)
    eligible=[v for v in validation if v['eligible']]
    selected=min(eligible,key=lambda v:(v['metrics']['rmse'],v['id']))['id'] if eligible else None
    lock={'selected':selected,'validation':validation,'baseline':baseline,'event_threshold':threshold,'test_metrics_read_for_selection':False}
    (out/'selection_lock.json').write_text(json.dumps(lock,indent=2))
    tests=[]
    for identifier,model in models.items():
        pred=np.array([model.predict_next(data[:t]) for t in range(775,912)])
        tests.append({'id':identifier,'metrics':metrics(pred,data[775:])})
        np.savez_compressed(out/(identifier+'_test.npz'),prediction=pred)
    result={'role':registration['role'],'selected':selected,'validation':validation,
            'test_all_candidates_descriptive':tests,'test_baseline':metrics(test_public,data[775:]),'confirmation_authorized':False}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

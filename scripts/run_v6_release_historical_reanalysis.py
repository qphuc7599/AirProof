"""Historical six-control reanalysis of exposed v5 caches; no simulation/search."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[name]='1'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from airproof.v6_release_experiment import (ReleaseControlsCalibration,from_release_evidence,
    run_release_controls,score_controls,_validate,_features)
from airproof.v6_release_twin import fit_sparse_release_calibration
from airproof.v5_release_twin import ReleaseCalibration

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/v5/selected_campaign_v1'
OUT=ROOT/'reports/v6/release_historical_reanalysis'
CELLS=('anchor_clean','outage_clean','severe_clean','severe_drift','severe_hotspot')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs={}; cal=[]
    for seed in range(912000,912008):
        for cell in CELLS:
            path=SOURCE/f'calibration/jobs/{seed}/{cell}/release.npz'
            inputs[str(path.relative_to(ROOT))]=sha(path)
            with np.load(path) as z: cal.append({k:z[k] for k in z.files})
    validated=[_validate(d['baseline'],d['residual_query'],d['residual'],d['residual_mask']) for d in cal]
    model=fit_sparse_release_calibration(np.stack([d['truth'] for d in cal]),
        np.stack([v[0] for v in validated]),np.stack([v[1] for v in validated]),np.stack([v[4] for v in validated]))
    x=np.concatenate([_features(v[0],v[2],v[3],model.mean) for v in validated])
    y=np.concatenate([d['truth'] for d in cal])
    coef=np.linalg.solve(x.T@x/len(x)+.1*np.eye(x.shape[1]),x.T@y/len(x))
    calibration=ReleaseControlsCalibration(model,coef,'historical-v5-calibration-912000-912007-all-five-cells',4)
    frozen=SOURCE/'release_calibration/frozen_calibration.npz'
    inputs[str(frozen.relative_to(ROOT))]=sha(frozen)
    with np.load(frozen) as z:
        old=ReleaseCalibration(z['mean'],z['transition'],z['process_covariance'],z['initial_covariance'],4)
    np.savez_compressed(OUT/'calibration.npz',mean=model.mean,transition=model.transition,
        initial_covariance=model.initial_covariance,process_covariance=model.process_covariance,oracle_coefficients=coef)
    manifest={'role':'historical reanalysis; all calibration/validation caches already exposed',
        'calibration_seeds':list(range(912000,912008)),'validation_seeds':list(range(913000,913012)),
        'cells':CELLS,'calibration_pooled_cells':True,'ridge':.1,'shrinkage':.1,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'airproof/v6_release_experiment.py',ROOT/'airproof/v6_release_twin.py']},
        'input_hashes':inputs,'confirmation':False,'noise_redrawn':False,'noise_scale':7.,'burn':48}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]; start=time.perf_counter()
    for seed in range(913000,913012):
        for cell in CELLS:
            path=SOURCE/f'validation/jobs/{seed}/{cell}/release.npz'
            inputs[str(path.relative_to(ROOT))]=sha(path)
            with np.load(path) as z: data={k:z[k] for k in z.files}
            result=run_release_controls(calibration,**from_release_evidence(data),evaluation_id=f'historical-{seed}-{cell}',v5_calibration=old)
            scores=score_controls(result,data['truth'])
            target=OUT/f'predictions/{seed}/{cell}.npz';target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,**result['predictions'],truth=data['truth'],public=data['baseline'],
                actual_query=data['residual_query'],noisy_query=data['residual'],release_mask=data['residual_mask'])
            rows.append({'seed':seed,'cell':cell,'scores':scores,'metadata':result['metadata']})
        print(f'completed {seed}; elapsed {time.perf_counter()-start:.1f}s',flush=True)
        (OUT/'world_metrics.json').write_text(json.dumps(rows,indent=2))
    summary={}
    for cell in CELLS:
        selected=[r for r in rows if r['cell']==cell]
        means={p:float(np.mean([r['scores'][p]['rmse'] for r in selected])) for p in ('P0','P1','P2','P3','P4','P5')}
        delta=np.array([r['scores']['P5']['rmse']-r['scores']['P0']['rmse'] for r in selected])
        se=delta.std(ddof=1)/np.sqrt(len(delta));half=float(student_t.ppf(.975,len(delta)-1)*se)
        summary[cell]={'mean_world_rmse':means,'P5_minus_P0_mean':float(delta.mean()),
            'P5_minus_P0_descriptive_95ci':[float(delta.mean()-half),float(delta.mean()+half)],
            'P5_better_worlds':int((delta<0).sum()),'worlds':len(delta),'citizen_gain_confirmed':False}
    manifest['elapsed_seconds']=time.perf_counter()-start
    manifest['output_hashes']={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'predictions').rglob('*.npz'))}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

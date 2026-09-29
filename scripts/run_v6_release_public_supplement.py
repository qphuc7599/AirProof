"""Post-exposure supplemental matched P3 public-only diagnostic, no tuning."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[name]='1'
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from airproof.v6_release_experiment import _features
BASE=ROOT/'reports/v6/release_historical_reanalysis'
SOURCE=ROOT/'reports/v5/selected_campaign_v1'
CELLS=('anchor_clean','outage_clean','severe_clean','severe_drift','severe_hotspot')

def main():
    out=BASE/'supplemental_public_only';out.mkdir(exist_ok=True)
    with np.load(BASE/'calibration.npz') as z: mean=z['mean']
    x=[];y=[]
    for seed in range(912000,912008):
        for cell in CELLS:
            with np.load(SOURCE/f'calibration/jobs/{seed}/{cell}/release.npz') as d:
                x.append(_features(d['baseline'],d['residual'],d['residual_mask'],mean)[:,:5]);y.append(d['truth'])
    x=np.concatenate(x);y=np.concatenate(y)
    coef=np.linalg.solve(x.T@x/len(x)+.1*np.eye(5),x.T@y/len(x))
    np.savez_compressed(out/'calibration.npz',coefficients=coef)
    rows=[]
    for seed in range(913000,913012):
        for cell in CELLS:
            with np.load(BASE/f'predictions/{seed}/{cell}.npz') as d:
                p=np.maximum(np.column_stack((np.ones(len(d['public'])),d['public']+mean[:4]))@coef,0)
                truth=d['truth'][48:]
                rmse=float(np.sqrt(np.mean((p[48:]-truth)**2)))
                p3=float(np.sqrt(np.mean((d['P3'][48:]-truth)**2)))
                rows.append({'seed':seed,'cell':cell,'public_only_rmse':rmse,'P3_rmse':p3,'P3_minus_public_only':p3-rmse})
                dest=out/f'predictions/{seed}';dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/f'{cell}.npz',public_only=p)
    summary={cell:{'mean_public_only_rmse':float(np.mean([r['public_only_rmse'] for r in rows if r['cell']==cell])),
        'mean_P3_minus_public_only':float(np.mean([r['P3_minus_public_only'] for r in rows if r['cell']==cell]))} for cell in CELLS}
    (out/'results.json').write_text(json.dumps({'role':'diagnostic added after six-control outcomes exposed; not confirmation',
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ridge':.1,
       'calibration':'same40 historical calibration caches; original P3 unchanged','summary':summary,'world_metrics':rows},indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

"""Post-exposure diagnostic public regression ablation; original P0-P5 preserved."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'reports/v5/selected_campaign_v1'
out=ROOT/'reports/v6/release_historical_reanalysis/supplemental_public_only'
out.mkdir(parents=True,exist_ok=True)
paths=sorted((source/'calibration/jobs').glob('*/*/release.npz'))
xs=[];ys=[];hashes={}
for path in paths:
    with np.load(path) as d:
        xs.append(np.column_stack((np.ones(len(d['baseline'])),d['baseline'])))
        ys.append(d['truth'])
    hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
x=np.concatenate(xs);y=np.concatenate(ys)
coef=np.linalg.solve(x.T@x/len(x)+.1*np.eye(x.shape[1]),x.T@y/len(x))
rows=[]
for path in sorted((source/'validation/jobs').glob('*/*/release.npz')):
    with np.load(path) as d:
        prediction=np.maximum(np.column_stack((np.ones(len(d['baseline'])),d['baseline']))@coef,0)
        rmse=float(np.sqrt(np.mean((prediction[48:]-d['truth'][48:])**2)))
    rows.append({'seed':int(path.parent.parent.name),'cell':path.parent.name,'public_regression_rmse':rmse})
    np.savez_compressed(out/(path.parent.parent.name+'_'+path.parent.name+'.npz'),prediction=prediction)
summary={cell:float(np.mean([r['public_regression_rmse'] for r in rows if r['cell']==cell])) for cell in sorted({r['cell'] for r in rows})}
(out/'result.json').write_text(json.dumps({'role':'supplemental diagnostic added after exposure to P3 outcomes; not confirmation',
 'ridge':.1,'fit':'calibration only; public features with intercept','calibration_hashes':hashes,'rows':rows,'mean_world_rmse':summary},indent=2))
print(json.dumps(summary))

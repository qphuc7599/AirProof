"""Registered one-candidate public-center diagnosis, not a confirmation result."""
from pathlib import Path
import json,hashlib
import numpy as np
from airproof.v6_feasibility import cap_rmse_floor
reg=Path('configs/v6/archive_center_diagnostic.json');protocol=json.loads(reg.read_text())
cal=Path('reports/v6/backbone_validation_recovery/public_validation_predictions.npz')
test=Path('reports/v4_validation/esntnn_purpleair_7800/heldout_predictions.npz')
with np.load(cal) as z:pred=z['prediction'];truth=z['truth']
coeff=[];means=pred.mean(0);scales=np.maximum(pred.std(0),1e-6)
for j in range(pred.shape[1]):
 x=np.column_stack((np.ones(len(pred)),(pred[:,j]-means[j])/scales[j]))
 prior=np.array([means[j],scales[j]])
 coeff.append(np.linalg.solve(x.T@x/len(x)+.1*np.eye(2),x.T@truth[:,j]/len(x)+.1*prior))
coeff=np.array(coeff)
with np.load(test) as z:
 original=np.maximum(z['ESN_TNN'],0);target=z['truth']
 corrected=np.maximum(coeff[:,0]+coeff[:,1]*(original-means)/scales,0)
result={'protocol':protocol,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (reg,cal,test,Path(__file__))},
 'original_public_rmse':float(np.sqrt(np.mean((original-target)**2))),
 'recalibrated_public_rmse':float(np.sqrt(np.mean((corrected-target)**2))),
 'original_cap_floor':cap_rmse_floor(target,original,8),'recalibrated_cap_floor':cap_rmse_floor(target,corrected,8),
 'historical_control_margin':1.05*2.062099,'v6_matched_control_evaluated':False,
 'claim':'a public-center feasibility diagnostic only; floor below margin would not establish achievable estimator accuracy'}
out=Path('reports/v6/archive_center_diagnostic');out.mkdir(exist_ok=False)
np.savez_compressed(out/'predictions.npz',original=original,recalibrated=corrected,truth=target,coefficients=coeff,means=means,scales=scales)
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','source_hashes')}))

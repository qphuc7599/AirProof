"""Historical cap feasibility without pretending archive targets calibrate public scales."""
from pathlib import Path
import json,hashlib
import numpy as np
from airproof.v6_feasibility import cap_rmse_floor
p=Path('reports/v4_validation/esntnn_purpleair_7800/heldout_predictions.npz')
with np.load(p) as d:
 truth=d['truth'];public=np.maximum(d['ESN_TNN'],0)
 result={'role':'historical archive diagnostic; not fresh confirmation','archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'keys':d.files,'cap':8,'cap_rmse_floor':cap_rmse_floor(truth,public,8),
 'public_rmse':float(np.sqrt(np.mean((truth-public)**2))),
 'same_information_v5_control_rmse_historical':2.062099,'allowed_margin':1.05,
 'public_loso_scale_available':False,
 'blocked_run_reason':'Archive contains target predictions, not independent public station training histories. Fitting innovation scale from these exposed targets would violate declared calibration protocol.'}
result['necessary_margin_floor_satisfied']=result['cap_rmse_floor']<=1.05*result['same_information_v5_control_rmse_historical']
out=Path('reports/v6/archive_cap_diagnostic');out.mkdir(parents=True,exist_ok=True)
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

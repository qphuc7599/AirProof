"""Recover only missing public validation predictions from frozen v4 weights."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,pickle,time
import numpy as np
import torch,rdata
from scripts.benchmark_esntnn import predict_tnn
from airproof.esntnn import make_transformer
ROOT=Path(__file__).resolve().parents[1];old=ROOT/'reports/v4_validation/esntnn_purpleair_7800'
out=ROOT/'reports/v6/backbone_validation_recovery';out.mkdir(parents=True,exist_ok=False)
m=json.loads((old/'manifest.json').read_text());selection=json.loads((old/'selection_locked.json').read_text())
data_path=ROOT/'data/external/baselines/esntnn_author/Data/APFour.RData'
for path,digest in [(old/'desn_model.pkl',selection['DESN_artifact_sha256']),(old/'tnn_state.pt',selection['TNN_artifact_sha256']),(data_path,m['data_sha256'])]:
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
(out/'registration.json').write_text(json.dumps({'role':'recovery of missing exposed public validation predictions, no retraining or independent confirmation',
 'original_contribution':'robust uncertainty-aware twin','weakness':'heldout cache lacks public residual calibration evidence',
 'smallest_new_compute':'frozen public inference only on prefix through774; no test targets read for prediction fitting',
 'split':m['split_half_open']['model_selection'],'weights_unchanged':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
start=time.perf_counter();torch.set_num_threads(1)
data=np.asarray(rdata.read_rda(str(data_path))['newpoll'],float).T
end=m['split_half_open']['model_selection'][1];prefix=data[:end].copy();del data
with (old/'desn_model.pkl').open('rb') as f:esn=pickle.load(f)
model=make_transformer();model.load_state_dict(torch.load(old/'tnn_state.pt',map_location='cpu',weights_only=True))
forecast=esn.predict_series(prefix);indices=np.arange(*m['split_half_open']['model_selection'])
tnn=predict_tnn(model,prefix,forecast,indices,history=m['TNN']['history'],mean=esn.mean,scale=esn.scale)
weight=selection['tnn_convex_weight'];prediction=(1-weight)*forecast[indices]+weight*tnn
np.savez_compressed(out/'public_validation_predictions.npz',prediction=prediction,truth=prefix[indices],indices=indices)
rmse=float(np.sqrt(np.mean((prediction-prefix[indices])**2)))
(out/'result.json').write_text(json.dumps({'validation_rmse':rmse,'historical_validation_rmse':selection['validation_rmse']['ESN_TNN'],
 'matches_historical':bool(np.isclose(rmse,selection['validation_rmse']['ESN_TNN'],rtol=1e-6)),
 'elapsed_seconds':time.perf_counter()-start,'scope':'public selection residuals; not LOSO, regulatory calibration, or independent uncertainty calibration'},indent=2))
print(rmse)

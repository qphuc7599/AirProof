"""One exposed-seed mechanistic decomposition; never a selection or test result."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import pickle
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from airproof.predictor_wrapper import synthetic_citizen_replay
from airproof.v6_estimator import EstimatorConfig,estimate_matched_controls
from scripts.select_v6_epa_2025_estimator import DATA,PARTITIONS,PROTOCOL,PUBLIC


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
 out=ROOT/'reports/v6/epa_2025_mechanistic_diagnostic';out.mkdir(exist_ok=False)
 protocol=json.loads(PROTOCOL.read_text());lock=json.loads((PUBLIC/'selection_lock.json').read_text())
 adapted=ROOT/'reports/v6/epa_2025_public_adaptation/selected_pretest_prediction.npz'
 cfg=EstimatorConfig(lambda_zero=.05,lambda_temporal=.05,lambda_spatial=.02,
  lag=6,time_step=1.,numerical_refinements=2,huber_delta=2.,clip_delta=4.)
 manifest={'role':'one exposed estimator-development seed; mechanistic decomposition only',
  'seed':917400,'kind':'clean','partition':PARTITIONS['development'],
  'config':asdict(cfg),'no_selection':True,'test_windows_read':False,
  'hashes':{str(p):sha(p) for p in (Path(__file__),adapted,PROTOCOL,DATA,ROOT/'airproof/v6_estimator.py')}}
 (out/'registration.json').write_text(json.dumps(manifest,indent=2))
 a,b=PARTITIONS['development'];warm=24;drain=24;start=a-warm
 frame=pd.read_parquet(DATA,filters=[('timestamp_utc','<',pd.Timestamp(protocol['start_utc'])+pd.Timedelta(hours=b))])
 table=frame.pivot(index='timestamp_utc',columns='station',values='value').sort_index()
 table=table.reindex(pd.date_range(table.index.min(),periods=b,freq='h',tz='UTC'))[protocol['stations']]
 truth=table.to_numpy()[start:b]
 with np.load(adapted) as z:public=z['prediction'][start:b]
 extended=np.concatenate((public,np.repeat(public[-1:],drain,axis=0)))
 channel=np.maximum(np.where(np.isfinite(truth),truth,public),0.)
 records=[x for x in synthetic_citizen_replay(channel,seed=917400,kind='clean') if np.isfinite(truth[x.epoch,x.cell])]
 archive=pickle.load(open(PUBLIC/'models.pkl','rb'));xy=archive['coordinates'];xy=(xy-xy.mean(0))*[111.2,111.2*np.cos(np.deg2rad(xy[:,0].mean()))]
 results=estimate_matched_controls(extended,xy,records,cfg,innovation_scales=3.)
 target=truth[warm:];mask=np.isfinite(target);rows=[]
 for name,result in results.items():
  estimate=result.reconstructed[:len(truth)][warm:]
  rows.append({'method':name,'rmse':float(np.sqrt(np.mean((estimate[mask]-target[mask])**2))),
   'failure_rate':result.diagnostics.get('solver_failure_rate',0),
   'maximum_correction':result.diagnostics.get('maximum_correction',0)})
 output={'role':manifest['role'],'rows':rows}
 (out/'result.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))

if __name__=='__main__':main()

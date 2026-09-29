"""Select rolling public ridge using development only; no estimator/test scoring."""
from pathlib import Path
import hashlib,json,pickle,sys,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from airproof.v6_feasibility import cap_rmse_floor
from airproof.v6_public_archive_model import causal_archive_design
from airproof.v6_public_rolling import rolling_stationwise_ridge
from scripts.select_v6_epa_2025_estimator import DATA,PARTITIONS,PROTOCOL,PUBLIC,public_prediction

OUT=ROOT/'reports/v6/epa_2025_public_rolling'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(pred,truth,threshold):
 m=np.isfinite(truth);e=m&(truth>=threshold)
 return {'rmse':float(np.sqrt(np.mean((pred[m]-truth[m])**2))),
  'event_recall':float(np.mean(pred[e]>=threshold)) if e.any() else None,
  'event_support':int(e.sum()),'cap8_floor':cap_rmse_floor(truth[m],pred[m],8.),
  'beyond_cap_fraction':float(np.mean(np.abs(truth[m]-pred[m])>8))}

def main():
 protocol=json.loads(PROTOCOL.read_text());lock=json.loads((PUBLIC/'selection_lock.json').read_text())
 OUT.mkdir(exist_ok=False);family=[{'id':f'rolling_ridge_h{h}','lookback':h} for h in (336,720,2160)]
 reg={'role':'rolling public-baseline selection on exposed development; later estimator-validation/tests unread',
  'family':family,'update_every_hours':24,'features':'same twelve locked causal archive features',
  'selection_partition':PARTITIONS['development'],'selection':'lowest observed-entry RMSE; exact tie by id',
  'hashes':{str(p):sha(p) for p in (Path(__file__),ROOT/'airproof/v6_public_rolling.py',
   ROOT/'airproof/v6_public_archive_model.py',PUBLIC/'models.pkl',PUBLIC/'selection_lock.json',PROTOCOL,DATA)},
  'estimator_validation_read':False,'test_windows_read':False,'confirmation_authorized':False}
 (OUT/'registration.json').write_text(json.dumps(reg,indent=2))
 end=PARTITIONS['validation'][1];cutoff=pd.Timestamp(protocol['start_utc'])+pd.Timedelta(hours=end)
 frame=pd.read_parquet(DATA,filters=[('timestamp_utc','<',cutoff)])
 table=frame.pivot(index='timestamp_utc',columns='station',values='value').sort_index()
 table=table.reindex(pd.date_range(table.index.min(),periods=end,freq='h',tz='UTC'))[protocol['stations']]
 values=table.to_numpy();archive=pickle.load(open(PUBLIC/'models.pkl','rb'));base=public_prediction(values,archive,lock['selected'])
 features=causal_archive_design(values,archive['medians'],archive['coordinates']);predictions={};fit_logs={};started=time.perf_counter()
 for candidate in family:
  predictions[candidate['id']],fit_logs[candidate['id']]=rolling_stationwise_ridge(features,values,base,
   start_epoch=7008,lookback=candidate['lookback'],update_every=24,penalty=1.,minimum_history=168)
 a,b=PARTITIONS['development'];rows=[{'id':c['id'],'metrics':metrics(predictions[c['id']][a:b],values[a:b],lock['event_threshold'])} for c in family]
 selected=min(rows,key=lambda x:(x['metrics']['rmse'],x['id']))['id']
 np.savez_compressed(OUT/'selected_pretest_prediction.npz',prediction=predictions[selected],selected=selected)
 result={'selected':selected,'development':rows,'fit_blocks':fit_logs[selected],
  'elapsed_seconds':time.perf_counter()-started,'estimator_validation_metrics_read_for_selection':False,
  'test_metrics_read_for_selection':False,'prediction_sha256':sha(OUT/'selected_pretest_prediction.npz'),
  'confirmation_authorized':False}
 (OUT/'selection_lock.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='fit_blocks'},indent=2))

if __name__=='__main__':main()

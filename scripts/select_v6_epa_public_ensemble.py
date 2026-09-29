"""Select a causal HGB/persistence public ensemble on exposed development only."""
from pathlib import Path
import hashlib,json,pickle,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from airproof.v6_feasibility import cap_rmse_floor
from scripts.select_v6_epa_2025_estimator import DATA,PARTITIONS,PROTOCOL,PUBLIC,public_prediction

OUT=ROOT/'reports/v6/epa_2025_public_ensemble_v2'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(pred,truth,threshold):
 m=np.isfinite(truth);e=m&(truth>=threshold)
 return {'rmse':float(np.sqrt(np.mean((pred[m]-truth[m])**2))),
  'event_recall':float(np.mean(pred[e]>=threshold)) if e.any() else None,
  'event_support':int(e.sum()),'cap8_floor':cap_rmse_floor(truth[m],pred[m],8.),
  'beyond_cap_fraction':float(np.mean(np.abs(truth[m]-pred[m])>8))}

def main():
 protocol=json.loads(PROTOCOL.read_text());lock=json.loads((PUBLIC/'selection_lock.json').read_text())
 OUT.mkdir(exist_ok=False)
 family=[{'id':'hgb','kind':'base'},{'id':'persistence','kind':'persistence'}]+[
  {'id':f'blend_p{w:g}','kind':'convex','persistence_weight':w} for w in (.25,.5,.75)]+[
  {'id':'high_hold_q95','kind':'high_state_hold','threshold_quantile':.95}]
 reg={'role':'post-shift public ensemble selection on exposed estimator-development; later validation/tests unread',
  'supersedes_invalid_artifact':'reports/v6/epa_2025_public_ensemble (negative persistence values violated public-field invariant before estimator evaluation)',
  'family':family,'selection_partition':PARTITIONS['development'],
  'selection':'lowest observed-entry RMSE, exact tie by id; all event/cap metrics retained',
  'information':'HGB and last observed regulatory values through t-1 only',
  'hashes':{str(p):sha(p) for p in (Path(__file__),PUBLIC/'models.pkl',PUBLIC/'selection_lock.json',PROTOCOL,DATA)},
  'estimator_validation_read':False,'test_windows_read':False,'confirmation_authorized':False}
 (OUT/'registration.json').write_text(json.dumps(reg,indent=2))
 end=PARTITIONS['validation'][1];cutoff=pd.Timestamp(protocol['start_utc'])+pd.Timedelta(hours=end)
 frame=pd.read_parquet(DATA,filters=[('timestamp_utc','<',cutoff)])
 table=frame.pivot(index='timestamp_utc',columns='station',values='value').sort_index()
 table=table.reindex(pd.date_range(table.index.min(),periods=end,freq='h',tz='UTC'))[protocol['stations']]
 values=table.to_numpy();archive=pickle.load(open(PUBLIC/'models.pkl','rb'));hgb=public_prediction(values,archive,lock['selected'])
 previous=archive['medians'].copy();persistence=np.empty_like(values)
 for epoch in range(len(values)):
  persistence[epoch]=previous;previous=np.where(np.isfinite(values[epoch]),values[epoch],previous)
 threshold=lock['event_threshold'];predictions={'hgb':hgb,'persistence':np.maximum(persistence,0.)}
 for weight in (.25,.5,.75):predictions[f'blend_p{weight:g}']=np.maximum((1-weight)*hgb+weight*persistence,0.)
 predictions['high_hold_q95']=np.maximum(np.where(persistence>=threshold,np.maximum(hgb,persistence),hgb),0.)
 a,b=PARTITIONS['development'];rows=[]
 for candidate in family:
  rows.append({'id':candidate['id'],'metrics':metrics(predictions[candidate['id']][a:b],values[a:b],threshold)})
 selected=min(rows,key=lambda r:(r['metrics']['rmse'],r['id']))['id']
 np.savez_compressed(OUT/'selected_pretest_prediction.npz',prediction=predictions[selected],selected=selected)
 selection={'selected':selected,'development':rows,'event_threshold':threshold,
  'estimator_validation_metrics_read_for_selection':False,'test_metrics_read_for_selection':False,
  'prediction_sha256':sha(OUT/'selected_pretest_prediction.npz'),'confirmation_authorized':False}
 (OUT/'selection_lock.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))

if __name__=='__main__':main()

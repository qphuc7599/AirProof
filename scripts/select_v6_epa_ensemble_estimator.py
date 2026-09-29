"""Repeat the fixed Huber-delta family because the public center changed."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,pickle,sys,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.select_v6_epa_2025_estimator as base_runner
from airproof.v6_estimator import EstimatorConfig

PROTOCOL=base_runner.PROTOCOL;DATA=base_runner.DATA;PARTITIONS=base_runner.PARTITIONS
PUBLIC=ROOT/'reports/v6/epa_2025_public_ensemble_v2';OUT=ROOT/'reports/v6/epa_2025_ensemble_estimator_v2'
base_runner.OUT=OUT

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 protocol=json.loads(PROTOCOL.read_text());center_lock=json.loads((PUBLIC/'selection_lock.json').read_text())
 OUT.mkdir(exist_ok=True)
 if any((OUT/name).exists() for name in ('development.jsonl','result.json','selection_lock.json')):
  raise FileExistsError('refusing to overwrite any evaluated ensemble-estimator artifact')
 family=[{'id':f'huber_d{d:g}','delta':d} for d in (1.345,2.,3.,4.)]
 cfg=EstimatorConfig(lambda_zero=.05,lambda_temporal=.05,lambda_spatial=.02,
  lag=6,time_step=1.,numerical_refinements=2)
 reg={'role':'changed-public-center repeat of fixed delta family; pre-test development/validation',
  'public_center':center_lock['selected'],'candidate_family':family,'base_config':asdict(cfg),
  'partitions':PARTITIONS,'development_seeds':list(range(917800,917808)),
  'validation_seeds':list(range(917900,917912)),
  'selection_gates':{'clean_ratio':1.05,'attack_attenuation':.2,
   'event_recall_loss_each_condition':.05,'solver_failure_rate':0.},
  'selection':'eligible only; lowest clean RMSE, then smaller delta',
  'hashes':{str(p):sha(p) for p in (Path(__file__),ROOT/'airproof/v6_estimator.py',
   ROOT/'scripts/select_v6_epa_2025_estimator.py',PUBLIC/'selection_lock.json',
   PUBLIC/'selected_pretest_prediction.npz',PROTOCOL,DATA)},
  'test_windows_read':False,'confirmation_authorized':False}
 (OUT/'registration.json').write_text(json.dumps(reg,indent=2))
 end=PARTITIONS['validation'][1];cutoff=pd.Timestamp(protocol['start_utc'])+pd.Timedelta(hours=end)
 frame=pd.read_parquet(DATA,filters=[('timestamp_utc','<',cutoff)])
 table=frame.pivot(index='timestamp_utc',columns='station',values='value').sort_index()
 table=table.reindex(pd.date_range(table.index.min(),periods=end,freq='h',tz='UTC'))[protocol['stations']]
 values=table.to_numpy()
 with np.load(PUBLIC/'selected_pretest_prediction.npz') as z:public=z['prediction'].copy()
 original_public=ROOT/'reports/v6/epa_2025_public_model/models.pkl'
 archive=pickle.load(open(original_public,'rb'));xy=archive['coordinates'];xy=(xy-xy.mean(0))*[
  111.2,111.2*np.cos(np.deg2rad(xy[:,0].mean()))]
 threshold=center_lock['event_threshold'];started=time.perf_counter()
 # Use the established runner with its explicit acquisition-only truth and drain contract.
 development_rows=base_runner.run_partition(
  'development',PARTITIONS['development'],reg['development_seeds'],family,values,public,xy,threshold,cfg)
 development=base_runner.summarize(development_rows,family);eligible=[x for x in development if x['eligible']]
 selected=min(eligible,key=lambda x:(x['clean_rmse'],float(x['id'].split('d')[1])))['id'] if eligible else None
 lock={'selected':selected,'development':development,'validation_metrics_read_for_selection':False,
  'test_metrics_read_for_selection':False,'confirmation_authorized':False}
 (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2))
 validation_rows=[];validation=[]
 if selected:
  chosen=[x for x in family if x['id']==selected]
  validation_rows=base_runner.run_partition('validation',PARTITIONS['validation'],reg['validation_seeds'],
   chosen,values,public,xy,threshold,cfg)
  validation=base_runner.summarize(validation_rows,chosen)
 result={'selected':selected,'development':development,'validation':validation,
  'development_rows':len(development_rows),'validation_rows':len(validation_rows),
  'validation_passes':bool(validation and validation[0]['eligible']),
  'elapsed_seconds':time.perf_counter()-started,
  'confirmation_authorized':bool(validation and validation[0]['eligible'])}
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()

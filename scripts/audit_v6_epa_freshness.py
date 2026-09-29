"""Chronology/support-only freshness audit; never reads forecast errors."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
p=Path('reports/v5/epa/protocol_manifest.json');m=json.loads(p.read_text())
archive=Path(m['archive']);frame=pd.read_parquet(archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['archive_sha256']
table=frame.pivot(index='timestamp_utc',columns='station',values='value').sort_index()
exposed=np.zeros(table.shape,bool)
intervals=[m['train_indices'],m['validation_indices'],*m['test_windows']]
for a,b in intervals:
 for station in m['stations']:exposed[a:b,table.columns.get_loc(station)]=True
candidate_windows=[];maximum_eligible=0
for end in range(len(table),335,-1):
 start=end-336
 eligible=[str(table.columns[j]) for j in range(table.shape[1]) if not exposed[start:end,j].any() and np.isfinite(table.iloc[start:end,j]).mean()>=.8]
 maximum_eligible=max(maximum_eligible,len(eligible))
 if len(eligible)>=4:candidate_windows.append({'indices':[start,end],'stations':eligible})
out={'role':'conservative freshness audit from known v5 exposure; older exposures could remove further support',
 'archive_sha256':m['archive_sha256'],'protocol_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'hours':len(table),'stations':len(table.columns),'known_exposed_stations':len(m['stations']),
 'known_exposed_intervals':intervals,'minimum_window_hours':336,'minimum_stations':4,'minimum_target_support':.8,
 'eligible_windows':candidate_windows,'maximum_unexposed_supported_stations':maximum_eligible,
 'fresh_confirmation_available':len(candidate_windows)>0,'errors_inspected':False}
target=Path('reports/v6/archive_usage');target.mkdir(exist_ok=True)
(target/'epa_freshness.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='eligible_windows'}))

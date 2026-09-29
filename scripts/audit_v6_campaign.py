"""Check complete exported arms and dependency identities without interpreting efficacy."""
from pathlib import Path
import json
import numpy as np

def audit_campaign(root):
 root=Path(root);manifest=json.loads((root/'manifest.json').read_text());problems=[];checked=0
 for slot in manifest['matrix']:
  folder=root/f"{slot['seed']}_{slot['cell']}";path=folder/'result.json'
  if not path.exists():continue
  job=json.loads(path.read_text())
  if job['status']!='complete':problems.append({'job':folder.name,'error':'execution failed'});continue
  if job.get('source_hash')!=manifest['source_hash']:problems.append({'job':folder.name,'error':'source mismatch'})
  expected={'PUBLIC','HUBER_reg0.1_cap8','HUBER_reg0.3_cap8','HUBER_reg1_cap8','SQ_reg0.1','SQ_reg0.3','SQ_reg1.0'}
  if {r['method'] for r in job['rows']}!=expected:problems.append({'job':folder.name,'error':'arm matrix mismatch'})
  for row in job['rows']:
   p=folder/(row['method']+'.npz')
   with np.load(p,allow_pickle=False) as z:
    if str(z['identity'])!=job['identity'] or str(z['source_hash'])!=manifest['source_hash']:problems.append({'job':folder.name,'error':'prediction identity mismatch'})
    for name in ('truth','live','reconstructed','public','scales'):
     if z[name].shape!=(624,1024) or not np.isfinite(z[name]).all():problems.append({'job':folder.name,'error':'shape or finite:'+name})
   checked+=1
 return {'role':'artifact integrity only; no scientific gate inference','checked_arms':checked,'problems':problems,'expected_arms':280,'complete':checked==280 and not problems}
if __name__=='__main__':
 root=Path('reports/v6/development_v1');result=audit_campaign(root);(root/'artifact_integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

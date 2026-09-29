"""All-cell development disposition; no optional stopping or target alteration."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('campaign');args=p.parse_args();root=Path(args.campaign)
 manifest=json.loads((root/'manifest.json').read_text());family=json.loads(Path('configs/v6/selection_family.json').read_text())
 expected={(r['seed'],r['cell']) for r in manifest['matrix']}
 jobs=[json.loads(f.read_text()) for f in root.glob('*_*/result.json')]
 found={(r['seed'],r['cell']) for r in jobs};complete=found==expected and all(r['status']=='complete' for r in jobs)
 result={'role':'development disposition, not confirmation','complete_matrix':complete,'expected_jobs':len(expected),'observed_jobs':len(found),'candidates':[],'selected':None}
 if not complete:
  result['reason']='missing or failed jobs; retain all partial outcomes'
 else:
  rows={(r['seed'],r['cell'],row['method']):row for r in jobs for row in r['rows']}
  seeds=sorted({s for s,c in expected})
  for candidate in family['candidates']:
   method=candidate['id'];sq='SQ_reg'+str(candidate['regularization_multiplier']);contrasts={};failures=[]
   for cell in family['cells']:
    ap=[rows[s,cell,method]['metrics'] for s in seeds];control=[rows[s,cell,sq]['metrics'] for s in seeds]
    contrasts[cell]={'rmse_ratio':float(np.mean([a['rmse'] for a in ap])/np.mean([c['rmse'] for c in control])),
      'event_recall_loss':None if any(a['event_recall'] is None or c['event_recall'] is None for a,c in zip(ap,control)) else float(np.mean([c['event_recall']-a['event_recall'] for a,c in zip(ap,control)]))}
    if any(a['solver_failure_rate']>0 or c['solver_failure_rate']>0 for a,c in zip(ap,control)):failures.append(cell+':numerical')
    if cell.endswith('clean') and contrasts[cell]['rmse_ratio']>1.05:failures.append(cell+':clean_margin')
    if contrasts[cell]['event_recall_loss'] is None or contrasts[cell]['event_recall_loss']>.05:failures.append(cell+':event')
   for attack in ('severe_drift','severe_hotspot'):
    a=np.mean([rows[s,attack,method]['metrics']['rmse']-rows[s,'severe_clean',method]['metrics']['rmse'] for s in seeds])
    c=np.mean([rows[s,attack,sq]['metrics']['rmse']-rows[s,'severe_clean',sq]['metrics']['rmse'] for s in seeds])
    contrasts[attack]['growth_contrast']=float(a-.8*c)
    if c<=0 or a-.8*c>0:failures.append(attack+':growth_target_or_denominator')
   result['candidates'].append({'id':method,'contrasts':contrasts,'failures':failures,'eligible':not failures,'worst_loss':max(x['rmse_ratio'] for x in contrasts.values())})
  eligible=[c for c in result['candidates'] if c['eligible']]
  if eligible:result['selected']=min(eligible,key=lambda c:(c['worst_loss'],c['id']))['id']
  result['confirmation_authorized']=False
  result['remaining_gates']=['independent calibration','validation H1-H7','archive/public-calibrated value','full audit and architecture comparators','prospective power and source lock']
 (root/'selection_disposition.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

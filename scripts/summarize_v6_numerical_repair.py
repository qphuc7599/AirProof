"""All-failure repair sensitivity, separately labeled from frozen selection."""
from pathlib import Path
import json
import numpy as np


def main():
    root = Path('reports/v6/development_v1')
    first = Path('reports/v6/targeted_solver_repair')
    rest = Path('reports/v6/remaining_solver_repair')
    # Require complete batch, not an opportunistic partial result.
    outcomes = json.loads((rest/'result.json').read_text())['outcomes']
    repaired = {(o['job'], r['method']): (r, rest/o['job']) for o in outcomes for r in o['rows']}
    for row in json.loads((first/'result.json').read_text())['rows']:
        repaired['6201001_severe_hotspot', row['method']] = row, first
    expected = {(r['job'], r['method']) for r in json.loads((root/'numerical_failure_audit.json').read_text())['failed_arms']}
    assert set(repaired) == expected
    metrics = {}
    for path in root.glob('*_*/result.json'):
        job = json.loads(path.read_text())
        for arm in job['rows']:
            metrics[job['seed'], job['cell'], arm['method']] = dict(arm['metrics'])
    changes = []
    for (job, method), (row, folder) in repaired.items():
        seed, cell = job.split('_', 1)
        metric = metrics[int(seed), cell, method]
        old_rmse = metric['rmse']
        with np.load(root/job/(method+'.npz')) as old, np.load(folder/(method+'.npz')) as new:
            truth = old['truth']
            live = new['live'][:len(truth)]
            reconstructed = new['reconstructed'][:len(truth)]
            metric['rmse'] = float(np.sqrt(np.mean((reconstructed-truth)**2)))
            events = truth >= metric['event_threshold']
            metric['event_recall'] = float(np.mean(live[events] >= metric['event_threshold'])) if events.any() else None
            delta = float(np.max(np.abs(reconstructed-old['reconstructed'])))
        metric['solver_failure_rate'] = row['solver_failure_rate']
        changes.append({'job': job, 'method': method, 'rmse_change': metric['rmse']-old_rmse,
                        'max_reconstructed_change': delta, 'remaining_failure_rate': row['solver_failure_rate']})
    contrasts = []
    for attack in ('severe_drift', 'severe_hotspot'):
        growth = {}
        for method in ('HUBER_reg0.1_cap8', 'SQ_reg0.1'):
            growth[method] = float(np.mean([metrics[s, attack, method]['rmse']-metrics[s, 'severe_clean', method]['rmse'] for s in range(6201000, 6201008)]))
        ap, sq = growth['HUBER_reg0.1_cap8'], growth['SQ_reg0.1']
        contrasts.append({'attack': attack, 'growth': growth, 'attenuation': 1-ap/sq if sq>0 else None,
                          'target': .2, 'passes': sq>0 and ap<=.8*sq})
    result = {'role': 'same-objective numerical-repair sensitivity; not independent confirmation',
        'frozen_development_unmodified': True, 'repaired_arms': len(changes),
        'all_repaired_arms_converged': all(r['remaining_failure_rate']==0 for r in changes),
        'changes': changes, 'attack_contrasts': contrasts,
        'confirmation_authorized': False}
    (rest/'sensitivity_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='changes'}, indent=2))


if __name__ == '__main__':
    main()

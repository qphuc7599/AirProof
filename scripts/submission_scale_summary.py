"""Require the complete planned scale matrix before exporting manuscript values."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/submission_checks_20260927'


def main():
    expected = {(n,g,s) for n,g in ((1000,32),(5000,32),(10000,32),(1000,64))
                for s in (927100,927101,927102)}
    results = {}
    for path in (OUT/'scalability').glob('n*_g*_s*.json'):
        row = json.loads(path.read_text())
        m = row['scale_measurement']
        key = (m['agents'],m['grid_side'],row['seed'])
        assert key not in results
        assert m['epochs'] == 672
        assert row['scenario']['attack_kind'] == 'clean'
        assert row['scenario']['fixed_lag'] == 6
        assert row['metrics']['constraint_violation_when_feasible_epochs'] == 0
        assert row['metrics']['privacy_budget_violation_count'] == 0
        assert row['metrics']['solver_failure_rate'] == 0
        assert row['metrics']['privacy_reserved_budget_bytes_per_epoch'] == 256*m['agents']
        assert m['runner_sha256'] == hashlib.sha256((ROOT/'scripts/submission_scalability.py').read_bytes()).hexdigest()
        results[key] = row
    assert set(results) == expected, f'Incomplete matrix: {sorted(expected-set(results))}'
    summaries = []
    for n,g in ((1000,32),(5000,32),(10000,32),(1000,64)):
        rows = [results[(n,g,s)] for s in (927100,927101,927102)]
        measurements = [x['scale_measurement'] for x in rows]
        summaries.append({'agents':n,'grid_side':g,'worlds':3,
            'epoch_update_p95_s_max':max(x['metrics']['epoch_update_p95_seconds'] for x in rows),
            'peak_rss_mib_max':max(x['end_to_end_peak_rss_mb'] for x in measurements),
            'cg_mean_mean':float(np.mean([x['cg_iterations_mean'] for x in measurements])),
            'cg_p95_max':max(x['cg_iterations_p95'] for x in measurements),
            'allocation_p95_ms_max':1000*max(x['allocation_seconds_p95'] for x in measurements),
            'allocation_total_s_mean':float(np.mean([x['allocation_seconds_total'] for x in measurements])),
            'wall_s_mean':float(np.mean([x['end_to_end_seconds'] for x in measurements])),
            'wall_s_range':[min(x['end_to_end_seconds'] for x in measurements),max(x['end_to_end_seconds'] for x in measurements)]})
    (OUT/'scalability_summary.json').write_text(json.dumps({'rows':summaries,'complete_jobs':12,
        'scope':'Descriptive retained-estimator scale extension; fixed processing budget; population-scaled DP reservation; sampled RSS including generation; shared host timings.'},indent=2))
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()

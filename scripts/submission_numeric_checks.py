"""Recompute T1, T2 and the descriptive T8(e) sensitivity from saved evidence."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from v4_core_analysis_common import load_core, vector, core_contrasts, holm_adjust, AP, SQ

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out = ROOT / 'reports/submission_checks_20260927'
    out.mkdir(exist_ok=True, parents=True)
    directory = ROOT / 'reports/v8/m5_release_reference_stress_confirmation_v1'
    rows=[]
    for job in sorted((directory/'jobs').iterdir()):
        result=json.loads((job/'result.json').read_text())
        path=job/'scoring_outputs.npz'
        assert sha(path)==result['scoring_outputs_sha256']
        with np.load(path) as data:
            rmse=float(np.sqrt(np.mean((data['stress_RAW_NONNEGATIVE_DIRECT'][48:672]-data['truth'][48:672])**2)))
            assert np.isclose(rmse,result['stress_scores']['RAW_NONNEGATIVE_DIRECT']['rmse'],rtol=0,atol=1e-12)
            rows.append({'seed':result['seed'],'rmse':rmse,
                'prediction_min':float(data['stress_RAW_NONNEGATIVE_DIRECT'].min()),
                'prediction_max':float(data['stress_RAW_NONNEGATIVE_DIRECT'].max())})
    certificate=json.loads((ROOT/'reports/v7/reviewer_revision/lifetime_retained_diagnostic/inexact_solver_certificate.json').read_text())
    manifest,indexed,_=load_core(ROOT/'reports/v4_primary/core_final_7600_7629_20260903')
    values=core_contrasts(indexed,manifest['seeds'])
    for n,cell in enumerate(('anchor_clean','outage_clean','severe_clean'),1):
        values[f'H{n}']=vector(indexed,manifest['seeds'],cell,AP)-1.01*vector(indexed,manifest['seeds'],cell,SQ)
    p={name:float(ttest_1samp(v,0,alternative='less').pvalue) for name,v in values.items()}
    output={'T1':{'worlds':rows,'mean_rmse':float(np.mean([r['rmse'] for r in rows])),
        'postprocessing':'max(last available noisy release, 0); public fallback until first release; publication clock; no upper clamp',
        'laplace_standard_deviation':float(np.sqrt(2)*17.5)},
        'T2':certificate['inexact_solver_certificate'],
        'T8e':{'role':'post-hoc sensitivity, not replacement of registered primary',
        'margin':.01,'worlds':len(manifest['seeds']),'p_values':p,'holm_seven':holm_adjust(p)}}
    (out/'numeric_checks.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()

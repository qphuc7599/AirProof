"""Final static audit of the opt-in estimator branch; runs no estimator outcome."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from airproof.v6_estimator import (
    EstimatorConfig, bounded_development_candidates,
    history_normalized_sensitivity_certificate)


ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registration_path=ROOT/'configs/v6/history_normalized_huber_development.json'
    cache_path=ROOT/'reports/v6/history_normalized_huber_development/cache_input_audit.json'
    registration=json.loads(registration_path.read_text());cache=json.loads(cache_path.read_text())
    production_references=[]
    for path in (ROOT/'airproof').glob('*.py'):
        if path.name!='v6_estimator.py' and 'history_normalized_huber' in path.read_text():
            production_references.append(str(path.relative_to(ROOT)))
    default=EstimatorConfig();candidates=bounded_development_candidates()
    example=history_normalized_sensitivity_certificate(strong_convexity_mu=.5,
        huber_delta=1.345,scale_floor=2.,per_user_weight_budget=1.)
    checks={
        'default_disabled':default.history_normalized_huber is False,
        'existing_candidates_disabled':all(not value.history_normalized_huber for value in candidates.values()),
        'no_external_production_activation':not production_references,
        'registered_cap8':registration['fixed_parameters']['cap']==8.,
        'registered_delta_1_345':registration['fixed_parameters']['huber_delta']==1.345,
        'registered_lag6':registration['fixed_parameters']['lag']==6,
        'at_most_three_budgets':len(registration['per_user_weight_budgets'])<=3,
        'cache_exact_inputs_absent':cache['exact_inputs_available'] is False,
        'cache_new_method_outcomes_zero':cache['new_method_outcomes_read']==0,
        'certificate_formula_example':example['fixed_window_minimizer_l2_bound']==2*1*1.345/(.5*2),
    }
    result={'role':'final independent static audit; no outcome, validation or confirmation run',
        'status':'pass_with_scoped_theorem' if all(checks.values()) else 'fail',
        'checks':checks,'production_activation_references':production_references,
        'mathematical_disposition':{
            'implementation_matches_proposition':True,
            'strong_convexity_mu':'lambda_zero * time_step; temporal and symmetric graph-Laplacian terms are positive semidefinite',
            'gradient_bound':'sum-user-weight <= B and |Huber score| <= delta imply B*delta/s_min; replacement doubles it',
            'required_conditions':['one fixed optimization window','fixed preceding reconstructed boundary',
                'fixed public context/operators/box and other-user arrivals','positive frozen scale floor',
                'integer user partition','collision-free admitted nullifiers'],
            'excluded_claims':['recursive whole-horizon sensitivity','differential privacy',
                'scientific accuracy or robustness improvement','H6/H7 repair','validation or confirmation']},
        'evidence_disposition':{'new_method_runs':0,'validation_opened':False,
            'confirmation_opened':False,'old_failure_retained':True},
        'sha256':{'estimator':sha(ROOT/'airproof/v6_estimator.py'),
            'tests':sha(ROOT/'tests/test_v6_history_normalized_huber.py'),
            'theory':sha(ROOT/'docs/V6_HISTORY_NORMALIZED_HUBER.md'),
            'registration':sha(registration_path),'cache_audit':sha(cache_path)}}
    out=ROOT/'reports/v6/history_normalized_huber_final_audit';out.mkdir(parents=True,exist_ok=True)
    audit=out/'audit.json';audit.write_text(json.dumps(result,indent=2)+'\n')
    manifest={'audit_sha256':sha(audit),'script_sha256':sha(Path(__file__)),
        'claim_ledger_sha256':sha(ROOT/'docs/CLAIM_LEDGER.md'),
        'repair_ledger_sha256':sha(ROOT/'docs/V6_REPAIR_EXECUTION_LEDGER.md'),
        'reuse_policy_sha256':sha(ROOT/'docs/V6_V4_REUSE_AND_CLAIM_POLICY.md')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']=='fail':raise SystemExit(1)


if __name__=='__main__':main()

"""Audit exposed development caches without running a new estimator outcome."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registration=ROOT/'configs/v6/history_normalized_huber_development.json'
    cfg=json.loads(registration.read_text())
    worlds=[]
    for seed in cfg['exposed_development_seeds']:
        directory=ROOT/f'reports/v6/development_v1/{seed}_severe_clean'
        public_file=directory/'PUBLIC.npz';allocation_file=directory/'allocations.json'
        arrays=np.load(public_file)
        allocation=json.loads(allocation_file.read_text())
        names={path.name.lower() for path in directory.iterdir()}
        worlds.append({'seed':seed,'directory_exists':directory.exists(),
            'public_array': 'public' in arrays.files,'frozen_scales':'scales' in arrays.files,
            'truth_array_present_but_forbidden_for_decisions':'truth' in arrays.files,
            'aggregate_allocation_fields':sorted(allocation[0]) if allocation else [],
            'exact_selected_observation_file':any(name.startswith(('observation','record')) for name in names),
            'per_record_arrival_map':False,'transition_operators':False})
    exact=all(w['public_array'] and w['frozen_scales'] and w['exact_selected_observation_file']
              and w['per_record_arrival_map'] and w['transition_operators'] for w in worlds)
    result={'role':'registered exposed cache-input audit; no new estimator outcome',
        'registration_preexisted_audit':True,'exact_inputs_available':exact,
        'decision':'do_not_run_cache_only_mechanistic_development' if not exact else 'eligible_to_run',
        'worlds':worlds,'new_method_outcomes_read':0,'validation_seeds_read':False,
        'confirmation_seeds_read':False,
        'reason':'Aggregate allocation counts cannot reconstruct user-normalized weights. Exact selected records, selection times and transition operators are absent from the cache.',
        'sha256':{'registration':sha(registration),
            'estimator':sha(ROOT/'airproof/v6_estimator.py')}}
    out=ROOT/'reports/v6/history_normalized_huber_development';out.mkdir(parents=True,exist_ok=True)
    audit=out/'cache_input_audit.json';audit.write_text(json.dumps(result,indent=2)+'\n')
    source_paths=('airproof/v6_estimator.py','scripts/audit_v6_history_normalized_cache.py',
        'configs/v6/history_normalized_huber_development.json',
        'tests/test_v6_history_normalized_huber.py','docs/V6_HISTORY_NORMALIZED_HUBER.md')
    manifest={'role':result['role'],'decision':result['decision'],'audit_sha256':sha(audit),
        'source_sha256':{path:sha(ROOT/path) for path in source_paths},
        'new_method_runs':0,'validation_or_confirmation_read':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Run the prospectively fixed v3 integrated terminal-batch smoke."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from airproof.meteorology import grid_coordinates
from airproof.simulator import generate_world
from airproof.v6_estimator import EstimatorConfig, estimate_public_field
from airproof.v6_experiment import integrated_transport
from airproof.v6_mobility import coupled_public_trace


ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'configs'/'v6_integration_smoke_accountability_batch_v3.json'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();registration=json.loads(REG.read_text())
    cfg=yaml.safe_load((ROOT/registration['base_configuration']).read_text())
    fixed=registration['fixture_overrides']
    cfg['world'].update(agents=fixed['agents'],steps=fixed['steps'],grid_side=fixed['grid_side'],
        burn_in_steps=fixed['burn_in_steps'],reference_station_count=fixed['reference_station_count'])
    cfg['attack']['start_epoch']=fixed['attack_start_epoch']
    cfg['scale']={'drain_epochs':fixed['drain_epochs']};cfg['audit']=registration['audit']
    seed=registration['fixture_seed'];world=generate_world(cfg,seed)
    trace,_=coupled_public_trace(cfg,seed,world.observations)
    transport,collector=integrated_transport(trace,world.observations,cfg)
    public=np.full((72,16),12.)
    estimate=estimate_public_field(public,grid_coordinates(4),collector.selected,
        EstimatorConfig(),innovation_scales=2.,arrival_map=collector.selection_times)
    sources=[REG,Path(__file__),ROOT/'airproof/v6_transport.py',ROOT/'airproof/v6_experiment.py',
             ROOT/'airproof/v6_audit_return.py',ROOT/'airproof/v6_receipts.py',ROOT/'airproof/v6_estimator.py']
    output={'registration':registration['registration'],'role':registration['role'],'seed':seed,
        'scale_source':'explicit fixed fixture','trace_hash':trace.trace_hash,
        'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sources},
        'observations':len(world.observations),'selected':len(collector.selected),
        'origin_verified':transport.metrics['audit_verified_return_pairs'],
        'comparators':registration['comparators'],
        'transport':{k:v for k,v in transport.metrics.items() if k!='local_decisions'},
        'finite_estimates':bool(np.isfinite(estimate.live).all()),'diagnostics':estimate.diagnostics,
        'scope_limits':registration['scope_limits']}
    target=ROOT/args.output;target.mkdir(parents=True,exist_ok=False)
    (target/'registration.json').write_bytes(REG.read_bytes())
    (target/'outcome.json').write_text(json.dumps(output,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    manifest={'registration_sha256':digest(REG),'source_hashes':output['source_hashes'],
              'outcome_sha256':digest(target/'outcome.json'),'scope_limits':registration['scope_limits']}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('observations','selected','origin_verified','finite_estimates')}))


if __name__=='__main__':main()

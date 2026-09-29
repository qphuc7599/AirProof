"""Engineering smoke only: no scientific model selection or confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from airproof.simulator import generate_world
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_experiment import integrated_transport
from airproof.v6_estimator import EstimatorConfig,estimate_public_field
from airproof.meteorology import grid_coordinates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,
        help='new artifact directory; existing paths are never overwritten')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path('reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml').read_text())
    cfg['world'].update(agents=24,steps=48,grid_side=4,burn_in_steps=8,reference_station_count=4)
    cfg['attack']['start_epoch']=24
    cfg['scale']={'drain_epochs':24}
    # Fixture identity, explicitly not a newly sealed scientific seed.
    world=generate_world(cfg,6006001)
    trace,paths=coupled_public_trace(cfg,6006001,world.observations)
    transport,collector=integrated_transport(trace,world.observations,cfg)
    # Constant public fixture avoids learning any innovation scale from test truth.
    public=np.full((72,16),12.)
    result=estimate_public_field(public,grid_coordinates(4),collector.selected,
        EstimatorConfig(),innovation_scales=2.,arrival_map=collector.selection_times)
    sources=[Path(__file__),Path('airproof/v6_transport.py'),Path('airproof/v6_experiment.py'),
             Path('airproof/v6_audit_return.py'),Path('airproof/v6_receipts.py'),
             Path('airproof/v6_estimator.py')]
    out={'role':'engineering_fixture_not_scientific_evidence','seed':6006001,
         'scale_source':'explicit fixed fixture','trace_hash':trace.trace_hash,
         'source_hashes':{str(path).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
         'observations':len(world.observations),'selected':len(collector.selected),
         'transport':{k:v for k,v in transport.metrics.items() if k!='local_decisions'},
         'finite_estimates':bool(np.isfinite(result.live).all()),'diagnostics':result.diagnostics}
    target=Path(args.output);target.mkdir(parents=True,exist_ok=False)
    (target/'outcome.json').write_text(json.dumps(out,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    print(json.dumps({k:out[k] for k in ('observations','selected','finite_estimates')}))

if __name__=='__main__':main()

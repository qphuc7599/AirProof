"""Paired transport engineering fixture; no scientific seed selection or primary."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time
import psutil
import yaml
from airproof.records import canonical_json
from airproof.simulator import generate_world
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_fairness import CumulativeServiceState
from airproof.v6_transport import simulate_transport
from airproof.v6_capacity_oracle import coupled_service_bound


class Collector:
    def __init__(self,config,fairness):
        self.config=config;self.fairness=fairness;self.backlog={};self.state=CumulativeServiceState()
        self.selected=[];self.violations=[];self.max_processing=0
    def __call__(self,arrivals,epoch):
        self.backlog.update((r.nullifier,r) for r in arrivals)
        self.backlog={k:r for k,r in self.backlog.items() if epoch<=r.epoch+6}
        result=self.state.allocate(self.backlog.values(),
            budget_bytes=self.config['scheduler']['budget_bytes_per_epoch'],
            targets={g:self.config['scheduler']['target_contributors'] for g in range(self.config['world']['groups'])},
            allocation_epoch=epoch,fairness=self.fairness)
        self.violations.extend(result.violations)
        self.max_processing=max(self.max_processing,result.spent_bytes)
        for r in result.selected:
            self.backlog.pop(r.nullifier)
        self.selected.extend(result.selected)
        return result.selected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-scale',action='store_true')
    parser.add_argument('--output',default='reports/v6/transport_component_fixture')
    parser.add_argument('--policies',nargs='+',default=['airproof_deadline','deadline6','combined','binary_spray_wait','epidemic_cap'])
    args=parser.parse_args()
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    source=Path('reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml')
    config=yaml.safe_load(source.read_text())
    if not args.full_scale:
        config['world'].update(agents=24,steps=48,grid_side=4,reference_station_count=4)
    config['transport']={'drain_epochs':24,'useful_lag_epochs':6,'ttl_epochs':24,
        'capacity_bytes_per_direction':1024,'raw_packet_bytes':512,'release_packet_bytes':256,
        'raw_buffer_bytes':18432,'release_buffer_bytes':6144,'copy_tokens':4,
        'release_gateway_reservation_bytes':256,'control_budget_bytes':128}
    source_files=['airproof/v6_transport.py','airproof/v6_mobility.py','airproof/v6_fairness.py',
                  'airproof/v6_capacity_oracle.py','airproof/v5_transport.py','airproof/simulator.py',__file__]
    role='exposed_fixture_runtime_only_not_validation'
    manifest={'role':role,'seed':6006001,'full_scale':args.full_scale,'config':config,
        'policies':args.policies,'allocation_fairness':True,
        'receipt_scope':'disabled component experiment; see integrated controller for receipt costs',
        'source_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
        'python':platform.python_version(),'started_unix':time.time(),
        'source_configuration_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    # Write exact design before generating outcomes.
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2))
    start=time.perf_counter();world=generate_world(config,6006001)
    trace,paths=coupled_public_trace(config,6006001,world.observations)
    generation_seconds=time.perf_counter()-start
    population={g:len({r.user_id for r in world.observations if r.group==g}) for g in range(config['world']['groups'])}
    common={'role':role,'seed':6006001,'trace_hash':trace.trace_hash,'records':len(world.observations),
        'generation_seconds':generation_seconds,'trajectory_sha256':hashlib.sha256(paths.tobytes()).hexdigest(),
        'records_sha256':hashlib.sha256(canonical_json([r.public_dict() for r in world.observations])).hexdigest()}
    (target/'common.json').write_text(json.dumps(common,indent=2))
    with gzip.open(target/'shared_trace.json.gz','wt',encoding='utf-8') as handle:
        json.dump(asdict(trace),handle)
    outcomes=[]
    for policy in args.policies:
        collector=Collector(config,True)
        start=time.perf_counter()
        result=simulate_transport(trace,world.observations,config,policy,collector_selector=collector)
        seconds=time.perf_counter()-start
        metrics=dict(result.metrics);decisions=metrics.pop('local_decisions',[])
        served={g:len({r.user_id for r in collector.selected if r.group==g}) for g in population}
        rates=[served[g]/population[g] for g in population if population[g]]
        checks=dict(buffer=metrics['max_raw_buffer_bytes']<=18432,
            release_buffer=metrics['max_release_buffer_bytes']<=6144,
            contact=metrics['max_contact_direction_bytes']<=1024,
            control=metrics['max_control_direction_bytes']<=128,
            copies=metrics['max_live_copies']<=4,tokens=metrics['token_violations']==0,
            processing=collector.max_processing<=83200,floor=not collector.violations,
            denominator=metrics['raw_generated']==metrics['raw_delivered']+metrics['raw_undelivered'])
        row={'policy':policy,'role':role,'runtime_seconds':seconds,
             'process_rss_bytes':psutil.Process().memory_info().rss,
             'process_peak_wset_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),
             'metrics':metrics,'selected_records':len(collector.selected),
             'selected_unique_by_group':served,'participating_unique_by_group':population,
             'participating_service_gap_diagnostic':max(rates)-min(rates) if rates else None,
             'checks':checks,'all_checks_pass':all(checks.values())}
        with gzip.open(target/(policy+'_raw.json.gz'),'wt',encoding='utf-8') as handle:
            json.dump({'arrivals':dict(result.raw_arrivals),'decisions':decisions,
                       'selected_nullifiers':[r.nullifier for r in collector.selected]},handle)
        (target/(policy+'.json')).write_text(json.dumps(row,indent=2))
        outcomes.append(row)
        print(json.dumps({'policy':policy,'runtime_seconds':seconds,'all_checks_pass':row['all_checks_pass']}),flush=True)
    # Full-scale is intentionally not silently downsampled for an LP diagnostic.
    if args.full_scale:
        bound={'status':'not_run_full_scale_resource_guard','upper_bound':None}
    else:
        bound=coupled_service_bound(trace,world.observations,config,max_variables=100000,max_constraints=100000)
    (target/'capacity_bound.json').write_text(json.dumps(bound,indent=2))
    (target/'summary.json').write_text(json.dumps({'common':common,'outcomes':outcomes,'capacity_bound':bound},indent=2))


if __name__=='__main__':main()

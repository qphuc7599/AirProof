"""Run the registered durable-inclusion async-return development smoke."""
import argparse,hashlib,json
from pathlib import Path

import numpy as np
import yaml

from airproof.meteorology import grid_coordinates
from airproof.simulator import generate_world
from airproof.v6_estimator import EstimatorConfig,estimate_public_field
from airproof.v6_experiment import integrated_transport
from airproof.v6_mobility import coupled_public_trace

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'configs'/'v6_durable_inclusion_async_return_v1.json'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    reg=json.loads(REG.read_text());f=reg['immutable_fixture']
    cfg=yaml.safe_load((ROOT/reg['base_configuration']).read_text())
    cfg['world'].update(agents=f['agents'],steps=f['steps'],grid_side=f['grid_side'],
        burn_in_steps=f['burn_in_steps'],reference_station_count=f['reference_station_count'])
    cfg['attack']['start_epoch']=f['attack_start_epoch'];cfg['scale']={'drain_epochs':f['drain_epochs']}
    cfg['audit']=reg['audit'];world=generate_world(cfg,reg['fixture_seed'])
    trace,_=coupled_public_trace(cfg,reg['fixture_seed'],world.observations)
    transport,collector=integrated_transport(trace,world.observations,cfg)
    public=np.full((72,16),12.)
    estimate=estimate_public_field(public,grid_coordinates(4),collector.selected,EstimatorConfig(),
        innovation_scales=2.,arrival_map=collector.selection_times)
    comparator=json.loads((ROOT/reg['paired_comparator']).read_text())
    if comparator['trace_hash']!=trace.trace_hash:raise AssertionError('paired workload trace mismatch')
    m=transport.metrics
    result={'registration':reg['registration'],'role':reg['role'],'trace_hash':trace.trace_hash,
        'observations':len(world.observations),'physical_raw_delivered':m['raw_delivered'],
        'timely_raw_delivered':m['raw_timely_delivered'],'stale_rejected':m['audit_stale_rejected'],
        'storage_rejected':m['audit_capacity_rejected'],'selected':len(collector.selected),
        'collector_included':m['audit_collector_log']['resolved_on_time'],
        'externally_checkpointed':m['audit_externally_checkpointed'],
        'publicly_checkpointed':m['audit_publicly_checkpointed'],
        'origin_verified':m['audit_verified_return_pairs'],
        'durable_storage_bytes':m['audit_durable_storage_bytes'],
        'durable_storage_capacity_bytes':m['audit_durable_storage_capacity_bytes'],
        'durable_storage_peak_bytes':m['audit_durable_storage_peak_bytes'],
        'inclusion_proof_bytes':m['audit_inclusion_proof_bytes'],
        'return_raw_bytes':m['receipt_payload_bytes'],'return_control_bytes':m['audit_return_control_bytes'],
        'expired_return_objects':m['receipt_expired'],'pending_return_objects':m['receipt_pending'],
        'return_backlog_objects':sum(len(x) for x in collector.audit.return_backlog.values()),
        'finite_estimates':bool(np.isfinite(estimate.live).all()),
        'paired_batch_v3':{'selected':comparator['selected'],'origin_verified':comparator['origin_verified'],
            'return_raw_bytes':comparator['transport']['receipt_payload_bytes'],
            'return_control_bytes':comparator['transport']['audit_return_control_bytes']},
        'detection_scope':{'omission_after_received_acceptance':'visible as missing terminal state',
            'mutation':'receipt content hash and Merkle proof verification',
            'rollback':'retained newer signed tree size detects older-head replay',
            'split_view':'conflicting signed roots detectable only when heads are compared',
            'public_availability':'not modeled'},
        'confirmation_eligible':False,'confirmation_blocker':reg['decision_rule'],
        'scope_limits':reg['scope_limits']}
    target=ROOT/args.output;target.mkdir(parents=True,exist_ok=False)
    (target/'registration.json').write_bytes(REG.read_bytes());(target/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[REG,Path(__file__),ROOT/'airproof/v6_experiment.py',ROOT/'airproof/v6_audit_return.py',
             ROOT/'airproof/v6_receipts.py',ROOT/'airproof/audit.py']
    manifest={'registration_sha256':sha(REG),'results_sha256':sha(target/'results.json'),
        'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},
        'confirmation_eligible':False,'scope_limits':reg['scope_limits']}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('selected','collector_included','externally_checkpointed','origin_verified','durable_storage_bytes','return_raw_bytes','return_control_bytes')},indent=2))

if __name__=='__main__':main()

"""Run registered durable inclusion plus finite public-checkpoint uplink."""
import argparse,hashlib,json
from dataclasses import replace
from pathlib import Path

import numpy as np,yaml

from airproof.audit import TransparencyLog,verify_tree_head
from airproof.meteorology import grid_coordinates
from airproof.simulator import generate_world
from airproof.v6_estimator import EstimatorConfig,estimate_public_field
from airproof.v6_experiment import integrated_transport
from airproof.v6_mobility import coupled_public_trace

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'configs'/'v6_durable_public_checkpoint_v2.json'
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
    transport,collector=integrated_transport(trace,world.observations,cfg);m=transport.metrics
    estimate=estimate_public_field(np.full((72,16),12.),grid_coordinates(4),collector.selected,
        EstimatorConfig(),innovation_scales=2.,arrival_map=collector.selection_times)
    batch=json.loads((ROOT/reg['comparators']['batch_v3']).read_text())
    durable=json.loads((ROOT/reg['comparators']['durable_v1']).read_text())
    if batch['trace_hash']!=trace.trace_hash or durable['trace_hash']!=trace.trace_hash:
        raise AssertionError('paired workload mismatch')
    heads=collector.publication.accepted_heads;latest=heads[-1]
    tampered=replace(latest,root_hex='0'*64)
    alternate=TransparencyLog(collector.key)
    for i in range(latest.tree_size):alternate.append(f'alternate-{i}'.encode())
    alternate_head=alternate.checkpoint(timestamp_ms=latest.timestamp_ms)
    checks={'normal_heads_signed':all(verify_tree_head(h,collector.key.public_key()) for h in heads),
        'tampered_head_rejected':not verify_tree_head(tampered,collector.key.public_key()),
        'rollback_detected_when_compared':len(heads)>1 and heads[0].tree_size<latest.tree_size,
        'split_view_detected_when_compared':verify_tree_head(alternate_head,collector.key.public_key())
            and alternate_head.tree_size==latest.tree_size and alternate_head.root_hex!=latest.root_hex,
        'missing_publication_not_counted':m['audit_publicly_checkpointed']<m['audit_collector_log']['resolved_on_time'],
        'byte_control_exact':m['audit_public_checkpoint_control_bytes']==
            m['audit_public_checkpoint_raw_bytes']//512*16}
    result={'registration':reg['registration'],'role':reg['role'],'trace_hash':trace.trace_hash,
        'observations':len(world.observations),'selected':len(collector.selected),
        'collector_included':m['audit_collector_log']['resolved_on_time'],
        'origin_verified':m['audit_verified_return_pairs'],'externally_checkpointed':m['audit_externally_checkpointed'],
        'publicly_checkpointed':m['audit_publicly_checkpointed'],
        'public_heads_verified':m['audit_public_checkpoint_heads_verified'],
        'public_uplink_contacts':m['audit_public_checkpoint_contacts'],
        'public_raw_bytes':m['audit_public_checkpoint_raw_bytes'],
        'public_control_bytes':m['audit_public_checkpoint_control_bytes'],
        'public_expired':m['audit_public_checkpoint_expired'],
        'public_pending':m['audit_public_checkpoint_pending'],
        'public_peak_queue_bytes':m['audit_public_checkpoint_peak_buffer_bytes'],
        'public_peak_reassembly_bytes':m['audit_public_checkpoint_reassembly_peak_bytes'],
        'origin_return_raw_bytes':m['receipt_payload_bytes'],'origin_return_control_bytes':m['audit_return_control_bytes'],
        'durable_storage_bytes':m['audit_durable_storage_bytes'],
        'durable_storage_capacity_bytes':m['audit_durable_storage_capacity_bytes'],
        'inclusion_proof_bytes':m['audit_inclusion_proof_bytes'],
        'fault_checks':checks,'all_fault_checks_passed':all(checks.values()),
        'comparators':{'batch_v3':{'selected':batch['selected'],'origin_verified':batch['origin_verified']},
            'durable_v1':{'selected':durable['selected'],'origin_verified':durable['origin_verified'],
                          'publicly_checkpointed':durable['publicly_checkpointed']}},
        'schedule_source':reg['audit']['public_checkpoint']['schedule_source'],
        'finite_estimates':bool(np.isfinite(estimate.live).all()),'confirmation_eligible':False,
        'confirmation_blocker':reg['decision_rule'],'scope_limits':reg['scope_limits']}
    target=ROOT/args.output;target.mkdir(parents=True,exist_ok=False)
    (target/'registration.json').write_bytes(REG.read_bytes());(target/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[REG,Path(__file__),ROOT/'airproof/v6_public_checkpoint.py',ROOT/'airproof/v6_experiment.py',
             ROOT/'airproof/v6_audit_return.py',ROOT/'airproof/v6_receipts.py',ROOT/'airproof/audit.py']
    manifest={'registration_sha256':sha(REG),'results_sha256':sha(target/'results.json'),
        'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},
        'all_fault_checks_passed':result['all_fault_checks_passed'],'confirmation_eligible':False,
        'scope_limits':reg['scope_limits']}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('selected','collector_included','origin_verified','publicly_checkpointed','public_heads_verified','public_raw_bytes','public_control_bytes')},indent=2))

if __name__=='__main__':main()

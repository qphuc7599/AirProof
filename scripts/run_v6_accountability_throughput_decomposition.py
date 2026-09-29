"""Run registered accountability bottleneck oracles on the exact smoke trace."""
import argparse,collections,hashlib,json,math
from pathlib import Path

import yaml

from airproof.records import canonical_json
from airproof.simulator import generate_world
from airproof.v6_experiment import integrated_transport
from airproof.v6_mobility import coupled_public_trace
from airproof.v6_receipts import CHUNK,FRAME_SIZE,HEADER,ReceiptReturnQueue

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'configs'/'v6_accountability_throughput_decomposition_v1.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def config(reg,buffer_bytes):
    cfg=yaml.safe_load((ROOT/reg['base_configuration']).read_text());f=reg['immutable_fixture']
    cfg['world'].update(agents=f['agents'],steps=f['steps'],grid_side=f['grid_side'],
        burn_in_steps=f['burn_in_steps'],reference_station_count=f['reference_station_count'])
    cfg['attack']['start_epoch']=f['attack_start_epoch'];cfg['scale']={'drain_epochs':f['drain_epochs']}
    cfg['audit']=dict(reg['frozen_audit']);cfg['audit']['collector_return_buffer_bytes']=buffer_bytes
    return cfg

def summarize(reg,buffer_bytes):
    seed=reg['fixture_seed'];cfg=config(reg,buffer_bytes);world=generate_world(cfg,seed)
    trace,_=coupled_public_trace(cfg,seed,world.observations)
    original=ReceiptReturnQueue.issue
    def measured(self,payload,origin,epoch,deadline,**kwargs):
        result=original(self,payload,origin,epoch,deadline,**kwargs)
        if result:
            frames=math.ceil(len(payload)/CHUNK)
            self.__dict__.setdefault('_decomposition_objects',[]).append({
                'origin':origin,'epoch':epoch,'deadline':deadline,'payload_bytes':len(payload),
                'frames':frames,'kind':json.loads(payload)['kind']})
        return result
    ReceiptReturnQueue.issue=measured
    try:transport,collector=integrated_transport(trace,world.observations,cfg)
    finally:ReceiptReturnQueue.issue=original
    m=transport.metrics;objects=getattr(collector.returns,'_decomposition_objects',[])
    fixed=sum(o['frames']*FRAME_SIZE for o in objects)
    exact=sum(o['payload_bytes']+o['frames']*(HEADER.size+64) for o in objects)
    lag=int(cfg.get('transport',{}).get('useful_lag_epochs',6))
    timely=[r for r in world.observations if r.nullifier in transport.raw_arrivals and
            transport.raw_arrivals[r.nullifier]<=r.epoch+lag]
    multiplicity=collections.Counter(r.user_id for r in timely)
    contacts=[]
    for receipt in collector.log._receipts.values():
        origin=collector.receipt_records[receipt.receipt_id][0]
        available=sum(origin in trace.gateway_contacts[e] for e in range(receipt.accepted_at+1,
            min(receipt.inclusion_deadline+1,len(trace.gateway_contacts))))
        contacts.append(available)
    verified=collections.Counter(collector.receipt_records[rid][0] for rid in collector.returned_pairs())
    return {'trace_hash':trace.trace_hash,'observations':len(world.observations),
        'selected':len(collector.selected),'origin_verified':m['audit_verified_return_pairs'],
        'admitted':m['receipt_issued'],'stale_rejected':m['audit_stale_rejected'],
        'capacity_rejected':m['audit_capacity_rejected'],'return_raw_bytes':m['receipt_payload_bytes'],
        'return_control_bytes':m['audit_return_control_bytes'],'expired_objects':m['receipt_expired'],
        'pending_objects':m['receipt_pending'],'terminal_objects':m['audit_terminal_objects_issued'],
        'issued_objects':len(objects),'serialized_payload_bytes':sum(o['payload_bytes'] for o in objects),
        'fixed_frame_wire_bytes_issued':fixed,'byte_exact_last_fragment_wire_bytes':exact,
        'zero_padding_bytes':fixed-exact,'zero_padding_fraction':(fixed-exact)/fixed,
        'timely_per_origin':dict(sorted(multiplicity.items())),'origins_with_timely':len(multiplicity),
        'timely_per_origin_summary':{'min':min(multiplicity.values()),'median':sorted(multiplicity.values())[len(multiplicity)//2],
                                     'max':max(multiplicity.values())},
        'future_contacts_per_admitted_receipt':{'min':min(contacts,default=0),
            'median':sorted(contacts)[len(contacts)//2] if contacts else 0,'max':max(contacts,default=0),
            'zero':sum(x==0 for x in contacts),'total':sum(contacts)},
        'verified_per_origin':dict(sorted(verified.items()))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    reg=json.loads(REG.read_text());actual=summarize(reg,reg['frozen_audit']['collector_return_buffer_bytes'])
    sufficient=summarize(reg,2097152)
    assert actual['trace_hash']==sufficient['trace_hash']
    padding_gate=actual['zero_padding_fraction']>=reg['codec_prototype_gate']['minimum_padding_fraction_of_fixed_wire']
    contact_limited=sufficient['origin_verified']<sufficient['selected']
    result={'registration':reg['registration'],'role':reg['role'],'trace_hash':actual['trace_hash'],
      'comparators':reg['comparators'],'oracles':{
        'actual_v3':actual,
        'infinite_return_frames_same_contacts_buffer':dict(actual,explanation='identical: v3 has no max_return_frames cap'),
        'sufficient_buffer_same_contacts':sufficient,
        'zero_return_cost':{'selected_ceiling':sufficient['selected'],'origin_verified_ceiling':sufficient['selected'],
            'definition':'instantaneous zero-byte verification of every audit-admitted timely record'},
        'byte_exact_serialization_bound':{'selected_ceiling':181,'origin_verified_ceiling':181,
            'issued_fixed_wire_bytes':actual['fixed_frame_wire_bytes_issued'],
            'issued_byte_exact_wire_bytes':actual['byte_exact_last_fragment_wire_bytes'],
            'padding_bytes':actual['zero_padding_bytes'],'padding_fraction':actual['zero_padding_fraction'],
            'note':'loose ceiling only; it removes final-fragment padding but does not invent reverse contacts'}},
      'decomposition':{'finite_buffer_selected_loss_vs_zero_cost':sufficient['selected']-actual['selected'],
        'same_contacts_verified_loss_with_sufficient_buffer':sufficient['selected']-sufficient['origin_verified'],
        'padding_gate_passed':padding_gate,'sufficient_buffer_still_contact_limited':contact_limited,
        'codec_prototype_authorized_by_registered_gate':padding_gate and not contact_limited},
      'scope_limits':reg['scope_limits']}
    target=ROOT/args.output;target.mkdir(parents=True,exist_ok=False)
    (target/'registration.json').write_bytes(REG.read_bytes())
    (target/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[REG,Path(__file__),ROOT/'airproof/v6_experiment.py',ROOT/'airproof/v6_audit_return.py',ROOT/'airproof/v6_receipts.py']
    manifest={'registration_sha256':sha(REG),'results_sha256':sha(target/'results.json'),
        'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},
        'scope_limits':reg['scope_limits']}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result['decomposition'],indent=2));print(json.dumps({k:(v.get('selected'),v.get('origin_verified')) for k,v in result['oracles'].items() if isinstance(v,dict)},indent=2))

if __name__=='__main__':main()

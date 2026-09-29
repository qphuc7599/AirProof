"""Export only checkpoints actually returned on the shared finite reverse path.

These feed a local-EVM causal replay, not a public-network latency experiment.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import json, sys, time, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_v7_shared_resource_confirmation import _config
from airproof.simulator import generate_world
from airproof.v6_mobility import coupled_public_trace
from airproof.v7_shared_execution import integrated_transport_reviewer
from airproof.audit import IngestionReceipt, ReceiptStatusWitness, SignedTreeHead
from airproof.v6_audit_return import verify_returned_bundle

def main():
    output=ROOT/'reports/submission_checks_20260927/anchoring'
    output.mkdir(parents=True,exist_ok=True)
    registration=json.loads((ROOT/'configs/v7/reviewer_shared_resource_core_confirmation_v3.json').read_text())
    cfg=_config(registration,'severe_clean')
    seed=927200
    started=time.perf_counter()
    world=generate_world(cfg,seed)
    trace,_=coupled_public_trace(cfg,seed,world.observations)
    transport,collector=integrated_transport_reviewer(trace,world.observations,cfg,
        policy=registration['execution']['relay_policy'],fairness=True,
        allocation_policy=registration['execution']['allocation_policy'])
    receipts,terminals={},{}
    for origin,receiver in collector.receivers.items():
        for identity,payload in receiver.completed.items():
            item=json.loads(payload)
            arrived=collector.returns.delivered[identity.hex()]
            if item['kind']=='acceptance':
                receipt=IngestionReceipt(**item['receipt'])
                receipts[receipt.receipt_id]=(receipt,arrived)
            elif item['kind']=='terminal':
                witness=ReceiptStatusWitness(**item['witness'])
                terminals[witness.receipt_id]=(witness,SignedTreeHead(**item['checkpoint']),arrived)
    verified=[]
    for rid,(receipt,receipt_epoch) in receipts.items():
        if rid not in terminals: continue
        witness,head,terminal_epoch=terminals[rid]
        if not verify_returned_bundle(receipt,witness,head,collector.contents[rid],collector.key.public_key()):
            raise AssertionError('Invalid returned proof')
        verified.append({'receipt_id':rid,'root':head.root_hex,'count':head.tree_size,
            'checkpoint_epoch':head.timestamp_ms//3600000,
            'available_epoch':max(receipt_epoch,terminal_epoch),
            'receipt_deadline_epoch':receipt.inclusion_deadline,
            'receipt_accepted_epoch':receipt.accepted_at,
            'proof_verified':True})
    verified.sort(key=lambda x:(x['available_epoch'],x['receipt_id']))
    timely=sum(row['available_epoch']<=row['receipt_deadline_epoch'] for row in verified)
    assert timely==transport.metrics['audit_verified_return_pairs']
    result={'role':'descriptive shared-resource checkpoint-to-local-EVM replay',
        'seed':seed,'agents':1000,'grid_side':32,'acquisition_epochs':672,'drain_epochs':24,
        'trace_hash':trace.trace_hash,'verified_receipts':verified,
        'transport_metrics':{k:v for k,v in transport.metrics.items() if k!='local_decisions'},
        'runtime_seconds':time.perf_counter()-started,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Anchor one returned signed checkpoint per unique root (batch size equals its tree size). Only existing receipt/terminal wire traffic is used. Local EVM begins after full receipt/proof arrival; public blockchain network is not simulated.'}
    (output/'returned_checkpoints.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'returned_pairs':len(verified),'issued':transport.metrics['receipt_issued'],
        'runtime_seconds':result['runtime_seconds']}),flush=True)
if __name__=='__main__':main()

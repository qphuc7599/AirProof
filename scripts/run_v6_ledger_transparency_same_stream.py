"""Run registered v6 same-stream ledger/transparency benchmark locally."""
import argparse,hashlib,json,time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airproof.v6_audit_return import DurableInclusionAsyncReturn
from airproof.v6_ledger_transparency import run_matrix,summarize
from airproof.v6_public_checkpoint import PublicCheckpointPublisher

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'configs'/'v6_ledger_transparency_same_stream_v1.json'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def interface_regression(reg):
    key=Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b'v6-interface-regression').digest())
    durable=DurableInclusionAsyncReturn(key,max_records=8,max_content_bytes=2048,
        durable_log_capacity_bytes=65536)
    receipts=[durable.accept(f'interface-{i}'.encode(),0,i) for i in range(4)]
    cfg=dict(reg['public_uplink']);cfg['uplink_epochs']=[4,6,8,10]
    publisher=PublicCheckpointPublisher(key,durable.tree,cfg)
    for epoch in range(11):publisher.tick(epoch)
    return {'receipts':len([x for x in receipts if x is not None]),'included':durable.tree.size,
        'publicly_checkpointed':publisher.publicly_checkpointed,
        'public_raw_bytes':publisher.queue.transmitted_bytes,
        'public_control_bytes':publisher.queue.control_bytes,
        'pass':len(receipts)==durable.tree.size==publisher.publicly_checkpointed==4
            and publisher.queue.control_bytes==publisher.queue.transmitted_bytes//512*16}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--registration',default=str(REG));args=parser.parse_args()
    registration_path=Path(args.registration).resolve();reg=json.loads(registration_path.read_text())
    target=ROOT/args.output;target.mkdir(parents=True,exist_ok=False)
    (target/'registration.json').write_bytes(registration_path.read_bytes());started=time.perf_counter()
    rows=run_matrix(reg)
    with (target/'results.jsonl').open('w') as handle:
        for row in rows:handle.write(json.dumps(row)+'\n')
    groups=summarize(rows);regression=interface_regression(reg)
    result={'registration':reg['registration'],'role':reg['role'],'rows':len(rows),
        'expected_rows':reg['expected_rows'],'complete':len(rows)==reg['expected_rows'],
        'all_clean_controls_valid':all(x['clean_control_valid'] for x in rows),
        'paired_stream_invariant':all(len({x['stream_sha256'] for x in rows
            if x['batch_size']==size and x['trial']==trial})==1
            for size in reg['batch_sizes'] for trial in range(reg['trials'])),
        'groups':groups,'interface_regression':regression,'v5_reuse_disposition':reg['v5_reuse_disposition'],
        'elapsed_seconds':time.perf_counter()-started,'results_sha256':sha(target/'results.jsonl'),
        'scope_limits':reg['scope_limits']}
    (target/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[registration_path,Path(__file__),ROOT/'airproof/v6_ledger_transparency.py',
        ROOT/'airproof/v6_public_checkpoint.py',ROOT/'airproof/v6_audit_return.py',ROOT/'airproof/audit.py']
    manifest={'registration_sha256':sha(registration_path),'results_sha256':result['results_sha256'],
        'analysis_sha256':sha(target/'analysis.json'),'source_hashes':{
            str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},
        'complete':result['complete'],'scope_limits':reg['scope_limits']}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('rows','complete','all_clean_controls_valid','paired_stream_invariant','elapsed_seconds')},indent=2))

if __name__=='__main__':main()

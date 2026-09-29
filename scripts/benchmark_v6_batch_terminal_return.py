"""Registered batch-receipt comparison on the same finite return workload."""
from pathlib import Path
import hashlib,json,time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from airproof.v6_audit_return import BatchTerminalAuditReturn,verify_returned_batch
from airproof.v6_resources import ContactBudget
from airproof.audit import IngestionReceipt,ReceiptStatusWitness,SignedTreeHead

out=Path('reports/v6/batch_terminal_return');out.mkdir(exist_ok=False)
sources=[Path(__file__),Path('airproof/v6_audit_return.py'),Path('airproof/v6_receipts.py'),Path('airproof/audit.py')]
manifest={'role':'registered engineering component comparison; exposed deterministic workload; not confirmation',
 'pillar':'accountability for accepted evidence under finite shared return resources',
 'batch_sizes':[32,128,512,2048],'batch_contract':'one ordered atomic batch, one origin, origin retains submitted contents',
 'forward_path':'raw contents excluded here and charged by integrated transport; no claim of free upload',
 'return_path':'signed batch-root acceptance plus signed outer-log terminal inclusion/checkpoint',
 'resources':{'collector_buffer_bytes':18432,'future_contacts':24,'capacity':1024,'raw_receipt_lane':640,
              'control_reserved':128,'release_reserved':256,'frame_bytes':512},
 'admission':'reserve acceptance+terminal frames before signed receipt exists','deadline':24,'max_batches':1,
 'comparators':'prior per-record unreserved failure and per-record admission-reserved result retained separately',
 'claim_limit':'same-origin atomic batch; origin must retain order; no cross-origin aggregation, availability, BFT or censorship claim',
 'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(out/'registration.json').write_text(json.dumps(manifest,indent=2))
rows=[]
for size in manifest['batch_sizes']:
 started=time.perf_counter();key=Ed25519PrivateKey.generate();origin=0
 contents=tuple(json.dumps({'id':i,'batch':size},sort_keys=True).encode() for i in range(size))
 engine=BatchTerminalAuditReturn(key,max_batches=1,max_return_frames=24)
 receipt=engine.accept_batch(contents,origin,0);engine.finalize(0)
 for epoch in range(1,25):engine.contact(origin,epoch,ContactBudget())
 objects=[json.loads(p) for p in engine.engine.receivers[origin].completed.values()]
 returned=IngestionReceipt(**next(x['receipt'] for x in objects if x['kind']=='acceptance'))
 terminal=next(x for x in objects if x['kind']=='terminal')
 witness=ReceiptStatusWitness(**terminal['witness']);head=SignedTreeHead(**terminal['checkpoint'])
 verified=verify_returned_batch(returned,witness,head,contents,origin,key.public_key())
 queue=engine.engine.queue;audit=engine.engine.log.audit(now=25)
 rows.append({'batch_size':size,'records_attempted':size,'records_admitted':engine.accepted_records,
  'records_rejected_before_receipt':engine.rejected_records,'returned_objects':len(objects),
  'verified_batches':int(verified),'verified_records':size if verified else 0,
  'unresolved_promises':len(audit['unresolved']),'late_promises':len(audit['late']),
  'queue_drops_after_admission':len(queue.dropped),'expired_objects':len(queue.expired),'pending_objects':len(queue.pending),
  'wire_bytes':queue.transmitted_bytes,'control_bytes':queue.control_bytes,'issued_frames':engine.engine.issued_frames,
  'peak_physical_queue_bytes':queue.peak_buffer,'peak_committed_queue_bytes':queue.peak_committed_buffer,
  'return_bytes_per_record':queue.transmitted_bytes/size,'elapsed_seconds':time.perf_counter()-started})
(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

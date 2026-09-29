"""Predeclared admission-reserved audit return under the old finite workload."""
from pathlib import Path
import hashlib,json,time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from airproof.v6_audit_return import TerminalAuditReturn,verify_returned_bundle
from airproof.v6_resources import ContactBudget
from airproof.audit import IngestionReceipt,ReceiptStatusWitness,SignedTreeHead

out=Path('reports/v6/terminal_return_reserved');out.mkdir(exist_ok=False)
manifest={'role':'registered engineering component comparison on prior adverse workload; not confirmation',
 'original_pillar':'blockchain/accountability with independently verifiable receipt and terminal status',
 'existing_failure':'terminal_return_probe issued promises before return-buffer/service reservation',
 'attempted_batches':[32,128,512,2048],'collector_buffer_bytes':18432,
 'service_budget':'24 future direct contacts, one 512-byte frame/contact in shared 640-byte raw/receipt lane',
 'admission':'reserve conservative frames for signed acceptance plus terminal proof before issuing receipt',
 'deadline':24,'max_return_frames':24,'contacts':'origin0 epochs1..24; attempts epoch0',
 'metrics':'attempted/admitted/rejected, issued/delivered promises, verified bundle pairs, bytes, buffers, expiry',
 'unchanged':'512-byte authenticated framing,128-byte control reserve,256-byte release reserve,24h deadline, cryptographic log',
 'claim_limit':'bounded component accountability; no consensus, BFT, censorship-resistance or public-network latency claim',
 'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
  [Path(__file__),Path('airproof/v6_audit_return.py'),Path('airproof/v6_receipts.py'),Path('airproof/audit.py'),Path('airproof/v6_resources.py')]}}
(out/'registration.json').write_text(json.dumps(manifest,indent=2))
rows=[]
for attempted in manifest['attempted_batches']:
 started=time.perf_counter();key=Ed25519PrivateKey.generate()
 engine=TerminalAuditReturn(key,batch_size=attempted,max_records=attempted,max_return_frames=24)
 contents={}
 for i in range(attempted):
  content=json.dumps({'id':i,'batch':attempted},sort_keys=True).encode()
  receipt=engine.accept(content,0,0)
  if receipt is not None:contents[receipt.receipt_id]=content
 engine.finalize(0)
 for epoch in range(1,25):engine.contact(0,epoch,ContactBudget())
 objects=[] if 0 not in engine.receivers else [json.loads(p) for p in engine.receivers[0].completed.values()]
 receipts={};terminals={}
 for item in objects:
  if item['kind']=='acceptance':
   receipt=IngestionReceipt(**item['receipt']);receipts[receipt.receipt_id]=receipt
  else:
   witness=ReceiptStatusWitness(**item['witness']);terminals[witness.receipt_id]=(witness,SignedTreeHead(**item['checkpoint']))
 verified=sum(verify_returned_bundle(receipts[rid],*terminals[rid],contents[rid],key.public_key())
              for rid in receipts.keys()&terminals.keys())
 audit=engine.log.audit(now=25)
 rows.append({'attempted':attempted,'admitted':len(engine.acceptance_ids),'rejected_before_receipt':len(engine.rejected_admission),
  'acceptance_generated':len(engine.acceptance_ids),'terminal_generated':len(engine.terminal_ids),
  'returned_objects':len(objects),'verified_terminal_pairs':verified,
  'unresolved_promises':len(audit['unresolved']),'late_promises':len(audit['late']),
  'queue_drops_after_admission':len(engine.queue.dropped),'expired_objects':len(engine.queue.expired),
  'pending_objects':len(engine.queue.pending),'issued_frames':engine.issued_frames,
  'wire_bytes':engine.queue.transmitted_bytes,'control_bytes':engine.queue.control_bytes,
  'peak_physical_queue_bytes':engine.queue.peak_buffer,'peak_committed_queue_bytes':engine.queue.peak_committed_buffer,
  'reservation_frames_per_record':engine.reservation_frames,'elapsed_seconds':time.perf_counter()-started})
(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

"""Terminal return workload probe, four batch sizes, finite buffers/contact schedule."""
import json,time,hashlib
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from airproof.v6_audit_return import TerminalAuditReturn
from airproof.v6_resources import ContactBudget
from airproof.audit import IngestionReceipt,ReceiptStatusWitness,verify_receipt_inclusion
out=Path('reports/v6/terminal_return_probe');out.mkdir(parents=True,exist_ok=True)
manifest={'role':'deterministic component workload probe, not world confirmation',
 'batches':[32,128,512,2048],'collector_buffer_bytes':18432,'contact_capacity':1024,
 'contacts':'one origin0 direct contact per epoch1..24; all accepted epoch0',
 'scope':'acceptance+terminal proof return; no new chain transaction or same-workload consensus claim'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
rows=[]
for size in manifest['batches']:
 start=time.perf_counter();key=Ed25519PrivateKey.generate();engine=TerminalAuditReturn(key,batch_size=size)
 contents={};receipts={}
 for i in range(size):
  content=json.dumps({'id':i,'batch':size},sort_keys=True).encode()
  receipt=engine.accept(content,0,0);contents[receipt.receipt_id]=content
 for epoch in range(1,25):engine.contact(0,epoch,ContactBudget())
 returned=[json.loads(p) for p in engine.receivers[0].completed.values()]
 for item in returned:
  if item['kind']=='acceptance':
   r=IngestionReceipt(**item['receipt']);receipts[r.receipt_id]=r
 verified=0
 for item in returned:
  if item['kind']=='terminal':
   w=ReceiptStatusWitness(**item['witness'])
   if w.receipt_id in receipts:verified+=verify_receipt_inclusion(receipts[w.receipt_id],w,contents[w.receipt_id],key.public_key())
 rows.append({'batch':size,'issued_objects':2*size,'returned_objects':len(returned),
  'verified_terminal_pairs':verified,'queue_drops':len(engine.queue.dropped),'pending':len(engine.queue.pending),
  'wire_bytes':engine.queue.transmitted_bytes,'control_bytes':engine.queue.control_bytes,
  'peak_queue_bytes':engine.queue.peak_buffer,'elapsed_seconds':time.perf_counter()-start})
(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))

"""Same-stream local ledger/transparency benchmark for the v6 evidence objects."""
from __future__ import annotations

import base64,hashlib,json,math,time
from dataclasses import asdict,replace

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .audit import (ReceiptAccountabilityLog,ReceiptStatusWitness,TransparencyLog,verify_consistency,
    verify_receipt,verify_receipt_inclusion,verify_status_witness,verify_tree_head)
from .records import canonical_json
from .v6_audit_return import verify_returned_bundle
from .v6_public_checkpoint import PublicCheckpointPublisher
from .v6_receipts import FRAME_SIZE,fragment


def _flat_head(payloads,key,log_id):
    body={'log_id':log_id,'tree_size':len(payloads),
          'root_hex':hashlib.sha256(b''.join(payloads)).hexdigest(),'timestamp_ms':10*3600000}
    return {**body,'signature_b64':base64.b64encode(key.sign(canonical_json(body))).decode()}


def _verify_flat(head,payloads,key):
    try:
        body={k:v for k,v in head.items() if k!='signature_b64'}
        key.verify(base64.b64decode(head['signature_b64']),canonical_json(body))
        return head['tree_size']==len(payloads) and head['root_hex']==hashlib.sha256(b''.join(payloads)).hexdigest()
    except (InvalidSignature,ValueError,TypeError,KeyError):return False


def _flat_witness(receipt,key,flat,index):
    body={'receipt_id':receipt.receipt_id,'status':'INCLUDED','witnessed_at':10,
          'batch_id':flat['log_id'],'leaf_index':index,'tree_size':flat['tree_size'],
          'root_hex':flat['root_hex'],'inclusion_proof_b64':(),'reason_code':None}
    return ReceiptStatusWitness(**body,signature_b64=base64.b64encode(key.sign(canonical_json(body))).decode())


def _publisher(key,tree,uplinks,public_cfg):
    cfg=dict(public_cfg);cfg['uplink_epochs']=uplinks
    publisher=PublicCheckpointPublisher(key,tree,cfg);detected=None
    for epoch in range(61):
        publisher.tick(epoch)
        if detected is None and publisher.publicly_checkpointed==tree.size:detected=epoch
    return publisher,detected


def run_matrix(registration):
    modes=registration['modes'];fault_epoch=registration['schedule']['fault_epoch']
    horizon=registration['schedule']['horizon_epoch'];deadline=registration['schedule']['receipt_deadline_epoch']
    rows=[]
    for size in registration['batch_sizes']:
      for trial in range(registration['trials']):
        key=Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f'v6-ledger:{size}:{trial}'.encode()).digest())
        public=key.public_key();payloads=tuple(hashlib.sha256(f'v6-evidence:{size}:{trial}:{i}'.encode()).digest() for i in range(size))
        stream_hash=hashlib.sha256(b''.join(payloads)).hexdigest();log_id=f'v6-{size}-{trial}'
        receipt_log=ReceiptAccountabilityLog(key)
        receipts=tuple(receipt_log.issue(x,policy_id='v6-terminal-component',accepted_at=0,
            inclusion_deadline=deadline) for x in payloads)
        tree=TransparencyLog(key,log_id=log_id)
        for payload in payloads:tree.append(payload)
        head=tree.checkpoint(timestamp_ms=fault_epoch*3600000);index=size//2;prefix=size//2
        proof=tree.inclusion_proof(index);consistency=tree.consistency_proof(prefix)
        witness=receipt_log.resolve(receipts[index].receipt_id,status='INCLUDED',witnessed_at=fault_epoch,
            batch_id=str(size),leaf_index=index,tree_size=size,root_hex=head.root_hex,inclusion_proof=proof)
        flat=_flat_head(payloads,key,log_id)
        signed_witness=_flat_witness(receipts[index],key,flat,index)
        fork_payloads=list(payloads);fork_payloads[index]=hashlib.sha256(payloads[index]+b'fork').digest()
        fork_tree=TransparencyLog(key,log_id=log_id)
        for payload in fork_payloads:fork_tree.append(payload)
        fork_head=fork_tree.checkpoint(timestamp_ms=fault_epoch*3600000)
        normal_anchor,anchor_epoch=_publisher(key,tree,registration['schedule']['normal_anchor_uplinks'],registration['public_uplink'])
        delayed_anchor,delayed_epoch=_publisher(key,tree,registration['schedule']['delayed_anchor_uplinks'],registration['public_uplink'])
        acceptance_bytes=sum(len(canonical_json({'kind':'acceptance','origin':0,'receipt':asdict(r)})) for r in receipts)
        sample=asdict(witness);signed_sample=asdict(signed_witness);terminal_bytes=0;signed_terminal_bytes=0
        for i,r in enumerate(receipts):
            item=dict(sample);item['receipt_id']=r.receipt_id;item['leaf_index']=i
            terminal_bytes+=len(canonical_json({'kind':'terminal','origin':0,'witness':item,'checkpoint':asdict(head)}))
            flat_item=dict(signed_sample);flat_item['receipt_id']=r.receipt_id;flat_item['leaf_index']=i
            signed_terminal_bytes+=len(canonical_json({'kind':'signed_terminal','origin':0,'witness':flat_item,'head':flat}))
        transparency_storage=sum(map(len,payloads))+acceptance_bytes+terminal_bytes
        signed_storage=sum(map(len,payloads))+acceptance_bytes+signed_terminal_bytes
        full_proof_bytes=size*len(proof)*32
        terminal_payload=canonical_json({'kind':'terminal','origin':0,'witness':asdict(witness),'checkpoint':asdict(head)})
        acceptance_payload=canonical_json({'kind':'acceptance','origin':0,'receipt':asdict(receipts[index])})
        transparency_frames=len(fragment(acceptance_payload,key))+len(fragment(terminal_payload,key))
        signed_response=canonical_json({'kind':'signed_log_response','head':flat,
            'payloads_b64':[base64.b64encode(x).decode() for x in payloads]})
        signed_frames=len(fragment(acceptance_payload,key))+len(fragment(signed_response,key))
        for mode in modes:
          started=time.perf_counter_ns()
          clean=(_verify_flat(flat,payloads,public) and verify_receipt(receipts[index],public)
                 and verify_status_witness(signed_witness,public)) if mode=='signed_log' else (
                 verify_returned_bundle(receipts[index],witness,head,payloads[index],public)
                 and verify_consistency(old_size=prefix,new_size=size,old_root=tree.root(prefix),
                    new_root=bytes.fromhex(head.root_hex),proof=consistency))
          if mode=='transparency_finite_public_anchor':clean=clean and normal_anchor.publicly_checkpointed==size
          verify_ns=time.perf_counter_ns()-started
          if not clean:raise AssertionError('clean control failed')
          base_frames=signed_frames if mode=='signed_log' else transparency_frames
          for fault in registration['applicability'][mode]:
            detected=None;reason='unobserved-by-horizon';publication=normal_anchor
            check=time.perf_counter_ns()
            if fault=='mutation':
                bad=payloads[index]+b'x'
                rejected=(not _verify_flat(flat,payloads[:index]+(bad,)+payloads[index+1:],public)
                    if mode=='signed_log' else not verify_receipt_inclusion(receipts[index],witness,bad,public))
                if rejected:detected,reason=11,'content-binding-rejection'
            elif fault=='omission_after_receipt':
                if verify_receipt(receipts[index],public):detected,reason=deadline+1,'signed-acceptance-terminal-timeout'
            elif fault=='rollback':
                if head.tree_size>prefix:detected,reason=11,'older-size-against-retained-head'
            elif fault=='split_view':
                conflict=verify_tree_head(fork_head,public) and fork_head.root_hex!=head.root_hex
                if conflict:
                    detected=(anchor_epoch if mode=='transparency_finite_public_anchor' else registration['schedule']['comparison_epoch'])
                    reason=('fork-disagrees-with-finite-public-anchor' if mode=='transparency_finite_public_anchor'
                            else 'conflicting-signed-heads-compared')
            elif fault=='checkpoint_loss':
                detected=(anchor_epoch if mode=='transparency_finite_public_anchor' else deadline+1)
                reason=('external-anchor-copy-reassembled' if mode=='transparency_finite_public_anchor'
                        else 'checkpoint-service-timeout')
            elif fault=='auditor_partition':
                publication=None # no auditor observation: deliberately censored
            elif fault=='delayed_anchor':
                publication=delayed_anchor
                if delayed_epoch is not None:detected,reason=delayed_epoch,'delayed-anchor-eventually-reassembled'
            elif fault=='forged_terminal_witness':
                forged=replace(signed_witness if mode=='signed_log' else witness,signature_b64='A'*86+'==')
                if not verify_status_witness(forged,public):detected,reason=11,'forged-terminal-signature-rejected'
            anchor_raw=0 if publication is None or mode!='transparency_finite_public_anchor' else publication.queue.transmitted_bytes
            anchor_control=0 if publication is None or mode!='transparency_finite_public_anchor' else publication.queue.control_bytes
            rows.append({'batch_size':size,'trial':trial,'mode':mode,'fault':fault,'stream_sha256':stream_hash,
                'detected':detected is not None,'detection_epoch':detected,
                'time_from_fault':None if detected is None else detected-fault_epoch,
                'right_censored':detected is None,'restricted_detection_time':horizon-fault_epoch if detected is None else detected-fault_epoch,
                'horizon_epoch':horizon,'reason':reason,'clean_control_valid':clean,
                'payload_bytes':sum(map(len,payloads)),'durable_storage_bytes':signed_storage if mode=='signed_log' else transparency_storage,
                'proof_bytes':0 if mode=='signed_log' else full_proof_bytes,
                'return_raw_bytes':base_frames*FRAME_SIZE,'return_control_bytes':base_frames*16,
                'public_anchor_raw_bytes':anchor_raw,'public_anchor_control_bytes':anchor_control,
                'clean_verify_ns':verify_ns,'fault_check_ns':time.perf_counter_ns()-check,
                'clock_scope':'registered logical fault/uplink clock; CPU verify cost measured separately'})
    return rows


def summarize(rows):
    groups={}
    for row in rows:groups.setdefault((row['batch_size'],row['mode'],row['fault']),[]).append(row)
    return [{'batch_size':k[0],'mode':k[1],'fault':k[2],'trials':len(items),
        'detected':sum(x['detected'] for x in items),'right_censored':sum(x['right_censored'] for x in items),
        'restricted_mean_detection_time':sum(x['restricted_detection_time'] for x in items)/len(items),
        'mean_storage_bytes':sum(x['durable_storage_bytes'] for x in items)/len(items),
        'mean_proof_bytes':sum(x['proof_bytes'] for x in items)/len(items),
        'mean_return_raw_bytes':sum(x['return_raw_bytes'] for x in items)/len(items),
        'mean_return_control_bytes':sum(x['return_control_bytes'] for x in items)/len(items),
        'mean_public_raw_bytes':sum(x['public_anchor_raw_bytes'] for x in items)/len(items),
        'mean_public_control_bytes':sum(x['public_anchor_control_bytes'] for x in items)/len(items),
        'mean_clean_verify_ns':sum(x['clean_verify_ns'] for x in items)/len(items)}
        for k,items in sorted(groups.items())]

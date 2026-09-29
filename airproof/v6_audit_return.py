"""Admission-reserved return of acceptance and terminal inclusion evidence."""
from dataclasses import asdict
import base64,hashlib,math
from .audit import (ReceiptAccountabilityLog,TransparencyLog,verify_receipt_inclusion,
                    verify_tree_head,_tree_hash)
from .records import canonical_json
from .v6_receipts import ReceiptReturnQueue,Receiver,CHUNK,FRAME_SIZE,fragment


def _frames(length):return (length+CHUNK-1)//CHUNK


def return_reservation_frames(*,max_records=2048,max_origin=2**31-1,max_epoch=10**9):
    """Conservative serialized bound for one receipt plus its terminal witness."""
    if any(type(x) is not int or x<1 for x in (max_records,max_origin,max_epoch)):
        raise ValueError('positive finite workload bounds required')
    signature=base64.b64encode(bytes(64)).decode();proof=base64.b64encode(bytes(32)).decode()
    receipt={'content_hash':'f'*64,'policy_id':'v6-terminal-component','receive_sequence':max_records,
             'accepted_at':max_epoch,'inclusion_deadline':max_epoch,'signature_b64':signature}
    acceptance=canonical_json({'kind':'acceptance','origin':max_origin,'receipt':receipt})
    witness={'receipt_id':'f'*64,'status':'INCLUDED','witnessed_at':max_epoch,
             'batch_id':str(max_records),'leaf_index':max_records-1,'tree_size':max_records,
             'root_hex':'f'*64,'inclusion_proof_b64':[proof]*math.ceil(math.log2(max_records)),
             'reason_code':None,'signature_b64':signature}
    checkpoint={'log_id':'airproof-log-v1','tree_size':max_records,'root_hex':'f'*64,
                'timestamp_ms':max_epoch*3600000,'signature_b64':signature}
    terminal=canonical_json({'kind':'terminal','origin':max_origin,'witness':witness,'checkpoint':checkpoint})
    return _frames(len(acceptance))+_frames(len(terminal))


def acceptance_reservation_frames(*,max_records=2048,max_origin=2**31-1,max_epoch=10**9):
    signature=base64.b64encode(bytes(64)).decode()
    receipt={'content_hash':'f'*64,'policy_id':'v6-terminal-component','receive_sequence':max_records,
             'accepted_at':max_epoch,'inclusion_deadline':max_epoch,'signature_b64':signature}
    return _frames(len(canonical_json({'kind':'acceptance','origin':max_origin,'receipt':receipt})))


def terminal_batch_reservation_frames(count,*,max_records=2048,max_origin=2**31-1,max_epoch=10**9):
    """Worst-case shared-checkpoint envelope for a bounded witness count."""
    if any(type(x) is not int or x<1 for x in (count,max_records,max_origin,max_epoch)) or count>max_records:
        raise ValueError('positive bounded batch/workload required')
    signature=base64.b64encode(bytes(64)).decode();proof=base64.b64encode(bytes(32)).decode()
    witness={'receipt_id':'f'*64,'status':'INCLUDED','witnessed_at':max_epoch,
             'batch_id':str(max_records),'leaf_index':max_records-1,'tree_size':max_records,
             'root_hex':'f'*64,'inclusion_proof_b64':[proof]*math.ceil(math.log2(max_records)),
             'reason_code':None,'signature_b64':signature}
    checkpoint={'log_id':'airproof-log-v1','tree_size':max_records,'root_hex':'f'*64,
                'timestamp_ms':max_epoch*3600000,'signature_b64':signature}
    payload=canonical_json({'kind':'terminal_batch','origin':max_origin,
                            'witnesses':[witness]*count,'checkpoint':checkpoint})
    return _frames(len(payload))


def verify_returned_bundle(receipt,witness,checkpoint,content,public_key):
    """Bind both signed promises to the same Merkle tree and deadline."""
    return (verify_receipt_inclusion(receipt,witness,content,public_key)
            and verify_tree_head(checkpoint,public_key)
            and witness.tree_size==checkpoint.tree_size
            and witness.root_hex==checkpoint.root_hex
            and witness.witnessed_at<=receipt.inclusion_deadline
            and checkpoint.timestamp_ms==witness.witnessed_at*3600000)


def batch_manifest(contents,origin):
    """Commit an ordered, atomic, same-origin evidence batch without echoing it."""
    items=tuple(bytes(item) for item in contents)
    if not items or any(not item for item in items) or type(origin) is not int or origin<0:
        raise ValueError('nonempty same-origin batch required')
    root=_tree_hash(items).hex()
    return canonical_json({'kind':'evidence_batch','origin':origin,'count':len(items),'root_hex':root})


def verify_returned_batch(receipt,witness,checkpoint,contents,origin,public_key):
    try:manifest=batch_manifest(contents,origin)
    except (TypeError,ValueError):return False
    return verify_returned_bundle(receipt,witness,checkpoint,manifest,public_key)

class TerminalAuditReturn:
    def __init__(self,key,*,buffer_bytes=18432,batch_size=32,max_records=2048,
                 max_return_frames=None):
        if batch_size<1 or max_records<1 or batch_size>max_records:raise ValueError('invalid bounded batch/workload')
        if max_return_frames is not None and (type(max_return_frames) is not int or max_return_frames<0):
            raise ValueError('invalid return-frame service budget')
        self.key=key;self.log=ReceiptAccountabilityLog(key);self.tree=TransparencyLog(key)
        self.queue=ReceiptReturnQueue(key,buffer_bytes);self.batch_size=batch_size
        self.pending=[];self.acceptance_ids=[];self.terminal_ids=[];self.receivers={}
        self.max_records=max_records;self.max_return_frames=max_return_frames
        self.reservation_frames=return_reservation_frames(max_records=max_records)
        self.frame_reservations={};self.issued_frames=0;self.rejected_admission=[]

    @property
    def committed_frames(self):return self.issued_frames+sum(self.frame_reservations.values())

    def _reserve(self,token):
        frames=self.reservation_frames
        if self.max_return_frames is not None and self.committed_frames+frames>self.max_return_frames:return False
        if not self.queue.reserve(token,frames*FRAME_SIZE):return False
        self.frame_reservations[token]=frames;return True

    def _issue(self,payload,origin,epoch,deadline,token):
        before=self.queue.reservations[token]
        if not self.queue.issue(payload,origin,epoch,deadline,reservation=token):
            raise AssertionError('unique reserved object rejected')
        used=(before-self.queue.reservations[token])//FRAME_SIZE
        self.frame_reservations[token]-=used;self.issued_frames+=used

    def accept(self,content,origin,epoch):
        token=hashlib.sha256(canonical_json({'content':hashlib.sha256(content).hexdigest(),
                 'origin':origin,'epoch':epoch,'sequence':len(self.acceptance_ids)+1})).hexdigest()
        if self.tree.size>=self.max_records or not self._reserve(token):
            self.rejected_admission.append({'content_hash':hashlib.sha256(content).hexdigest(),
                'origin':origin,'epoch':epoch,'reason':'reserved_return_capacity'})
            return None
        receipt=self.log.issue(content,policy_id='v6-terminal-component',accepted_at=epoch,inclusion_deadline=epoch+24)
        self.receivers.setdefault(origin,Receiver(self.key.public_key()))
        self._issue(canonical_json({'kind':'acceptance','origin':origin,'receipt':asdict(receipt)}),origin,epoch,epoch+24,token)
        index=self.tree.append(content);self.pending.append((receipt,origin,index))
        self.pending[-1]=(*self.pending[-1],token)
        self.acceptance_ids.append(receipt.receipt_id)
        if len(self.pending)>=self.batch_size:self.flush(epoch)
        return receipt

    def flush(self,epoch):
        if not self.pending:return
        checkpoint=self.tree.checkpoint(timestamp_ms=epoch*3600000)
        for receipt,origin,index,token in self.pending:
            witness=self.log.resolve(receipt.receipt_id,status='INCLUDED',witnessed_at=epoch,
                batch_id=str(self.tree.size),leaf_index=index,tree_size=self.tree.size,
                root_hex=self.tree.root().hex(),inclusion_proof=self.tree.inclusion_proof(index))
            payload=canonical_json({'kind':'terminal','origin':origin,'witness':asdict(witness),'checkpoint':asdict(checkpoint)})
            self._issue(payload,origin,epoch,epoch+24,token);self.terminal_ids.append(receipt.receipt_id)
            self.queue.release_reservation(token)
            self.frame_reservations.pop(token)
        self.pending.clear()

    def finalize(self,epoch):self.flush(epoch)

    def contact(self,origin,epoch,budget):
        return self.queue.contact(origin,epoch,budget,self.receivers[origin])


class BatchTerminalAuditReturn:
    """One verifiable receipt per ordered same-origin batch.

    Raw evidence still consumes the forward path. The return path carries only a
    signed Merkle commitment and its outer-log terminal witness. The origin must
    retain its ordered submitted batch to verify individual membership.
    """
    def __init__(self,key,*,buffer_bytes=18432,max_batches=2048,max_return_frames=None):
        self.engine=TerminalAuditReturn(key,buffer_bytes=buffer_bytes,batch_size=1,
            max_records=max_batches,max_return_frames=max_return_frames)
        self.accepted_records=0;self.rejected_records=0;self.batch_sizes={}

    def accept_batch(self,contents,origin,epoch):
        items=tuple(bytes(item) for item in contents)
        manifest=batch_manifest(items,origin)
        receipt=self.engine.accept(manifest,origin,epoch)
        if receipt is None:
            self.rejected_records+=len(items);return None
        self.accepted_records+=len(items);self.batch_sizes[receipt.receipt_id]=len(items)
        return receipt

    def finalize(self,epoch):self.engine.finalize(epoch)
    def contact(self,origin,epoch,budget):return self.engine.contact(origin,epoch,budget)


class SameOriginTerminalBatchAuditReturn:
    """Prompt per-record acceptances with reservation-safe terminal batching."""
    def __init__(self,key,*,buffer_bytes=18432,batch_size=8,batch_timeout_epochs=1,
                 max_records=2048,max_return_frames=None,max_origin=2**31-1,max_epoch=10**9,
                 inclusion_deadline_epochs=24):
        if any(type(x) is not int for x in (batch_size,batch_timeout_epochs,max_records,
                                             max_origin,max_epoch,inclusion_deadline_epochs)):
            raise ValueError('integral batch bounds required')
        if not 1<=batch_size<=max_records or not 0<=batch_timeout_epochs<=inclusion_deadline_epochs:
            raise ValueError('invalid prospective batch size/timeout')
        if min(max_origin,max_epoch,inclusion_deadline_epochs)<1:raise ValueError('invalid workload bounds')
        if max_return_frames is not None and (type(max_return_frames) is not int or max_return_frames<0):
            raise ValueError('invalid return-frame service budget')
        self.key=key;self.log=ReceiptAccountabilityLog(key);self.tree=TransparencyLog(key)
        self.queue=ReceiptReturnQueue(key,buffer_bytes);self.batch_size=batch_size
        self.batch_timeout_epochs=batch_timeout_epochs;self.max_records=max_records
        self.max_return_frames=max_return_frames;self.max_origin=max_origin;self.max_epoch=max_epoch
        self.inclusion_deadline_epochs=inclusion_deadline_epochs
        self.acceptance_frame_bound=acceptance_reservation_frames(max_records=max_records,
            max_origin=max_origin,max_epoch=max_epoch)
        self.batches={};self.batch_counter=0;self.acceptance_ids=[];self.terminal_ids=[]
        self.frame_reservations={};self.issued_frames=0;self.rejected_admission=[]
        self.terminal_object_count=0;self.flushed_batch_sizes=[]

    @property
    def committed_frames(self):return self.issued_frames+sum(self.frame_reservations.values())

    def _extend(self,token,frames):
        if self.max_return_frames is not None and self.committed_frames+frames>self.max_return_frames:return False
        ok=(self.queue.extend_reservation(token,frames*FRAME_SIZE) if token in self.queue.reservations
            else self.queue.reserve(token,frames*FRAME_SIZE))
        if ok:self.frame_reservations[token]=self.frame_reservations.get(token,0)+frames
        return ok

    def _issue(self,payload,origin,epoch,deadline,token):
        before=self.queue.reservations[token]
        if not self.queue.issue(payload,origin,epoch,deadline,reservation=token):
            raise AssertionError('unique reserved object rejected')
        used=(before-self.queue.reservations[token])//FRAME_SIZE
        self.frame_reservations[token]-=used;self.issued_frames+=used

    def advance(self,epoch):
        for origin,batch in list(self.batches.items()):
            if epoch>=batch['opened']+self.batch_timeout_epochs:self.flush_origin(origin,epoch)

    def accept(self,content,origin,epoch):
        if origin>self.max_origin or epoch>self.max_epoch or self.tree.size>=self.max_records:
            self.rejected_admission.append({'content_hash':hashlib.sha256(content).hexdigest(),
                'origin':origin,'epoch':epoch,'reason':'declared_workload_bound'})
            return None
        self.advance(epoch)
        batch=self.batches.get(origin)
        old_count=0 if batch is None else len(batch['entries'])
        new_count=old_count+1
        old_envelope=0 if old_count==0 else terminal_batch_reservation_frames(old_count,
            max_records=self.max_records,max_origin=self.max_origin,max_epoch=self.max_epoch)
        new_envelope=terminal_batch_reservation_frames(new_count,max_records=self.max_records,
            max_origin=self.max_origin,max_epoch=self.max_epoch)
        if batch is None:
            token=f'terminal-batch:{origin}:{self.batch_counter}';self.batch_counter+=1
        else:token=batch['token']
        needed=self.acceptance_frame_bound+new_envelope-old_envelope
        if not self._extend(token,needed):
            self.rejected_admission.append({'content_hash':hashlib.sha256(content).hexdigest(),
                'origin':origin,'epoch':epoch,'reason':'reserved_return_capacity'})
            return None
        receipt=self.log.issue(content,policy_id='v6-terminal-component',accepted_at=epoch,
            inclusion_deadline=epoch+self.inclusion_deadline_epochs)
        acceptance=canonical_json({'kind':'acceptance','origin':origin,'receipt':asdict(receipt)})
        self._issue(acceptance,origin,epoch,receipt.inclusion_deadline,token)
        index=self.tree.append(content)
        if batch is None:
            batch={'token':token,'opened':epoch,'entries':[]};self.batches[origin]=batch
        batch['entries'].append((receipt,index))
        self.acceptance_ids.append(receipt.receipt_id)
        if len(batch['entries'])>=self.batch_size or self.batch_timeout_epochs==0:
            self.flush_origin(origin,epoch)
        return receipt

    def flush_origin(self,origin,epoch):
        batch=self.batches.get(origin)
        if not batch:return
        checkpoint=self.tree.checkpoint(timestamp_ms=epoch*3600000);witnesses=[]
        for receipt,index in batch['entries']:
            witness=self.log.resolve(receipt.receipt_id,status='INCLUDED',witnessed_at=epoch,
                batch_id=str(self.tree.size),leaf_index=index,tree_size=self.tree.size,
                root_hex=self.tree.root().hex(),inclusion_proof=self.tree.inclusion_proof(index))
            witnesses.append(witness);self.terminal_ids.append(receipt.receipt_id)
        payload=canonical_json({'kind':'terminal_batch','origin':origin,
            'witnesses':[asdict(w) for w in witnesses],'checkpoint':asdict(checkpoint)})
        deadline=min(r.inclusion_deadline for r,_ in batch['entries'])
        self._issue(payload,origin,epoch,deadline,batch['token'])
        self.queue.release_reservation(batch['token']);self.frame_reservations.pop(batch['token'])
        self.flushed_batch_sizes.append(len(witnesses));self.terminal_object_count+=1
        del self.batches[origin]

    def finalize(self,epoch):
        for origin in list(self.batches):self.flush_origin(origin,epoch)


def durable_record_storage_bound(content_bytes,*,max_records=2048,max_origin=2**31-1,
                                 max_epoch=10**9,inclusion_deadline_epochs=24):
    """Bound leaf plus signed acceptance and terminal state before signing."""
    if any(type(x) is not int or x<1 for x in (content_bytes,max_records,max_origin,max_epoch,
                                                inclusion_deadline_epochs)):
        raise ValueError('positive durable workload bounds required')
    clock=max_epoch+inclusion_deadline_epochs
    signature=base64.b64encode(bytes(64)).decode();proof=base64.b64encode(bytes(32)).decode()
    receipt={'content_hash':'f'*64,'policy_id':'v6-terminal-component','receive_sequence':max_records,
             'accepted_at':max_epoch,'inclusion_deadline':clock,'signature_b64':signature}
    acceptance=canonical_json({'kind':'acceptance','origin':max_origin,'receipt':receipt})
    witness={'receipt_id':'f'*64,'status':'INCLUDED','witnessed_at':max_epoch,
             'batch_id':str(max_records),'leaf_index':max_records-1,'tree_size':max_records,
             'root_hex':'f'*64,'inclusion_proof_b64':[proof]*math.ceil(math.log2(max_records)),
             'reason_code':None,'signature_b64':signature}
    checkpoint={'log_id':'airproof-log-v1','tree_size':max_records,'root_hex':'f'*64,
                'timestamp_ms':max_epoch*3600000,'signature_b64':signature}
    terminal=canonical_json({'kind':'terminal','origin':max_origin,'witness':witness,
                             'checkpoint':checkpoint})
    return content_bytes+len(acceptance)+len(terminal)


class DurableInclusionAsyncReturn:
    """Bounded durable inclusion whose later return never gates allocation."""
    def __init__(self,key,*,buffer_bytes=18432,max_records=2048,max_content_bytes=2048,
                 durable_log_capacity_bytes=1048576,max_origin=2**31-1,max_epoch=10**9,
                 inclusion_deadline_epochs=24):
        values=(buffer_bytes,max_records,max_content_bytes,durable_log_capacity_bytes,
                max_origin,max_epoch,inclusion_deadline_epochs)
        if any(type(x) is not int or x<1 for x in values):raise ValueError('positive fixed durable bounds required')
        self.key=key;self.log=ReceiptAccountabilityLog(key);self.tree=TransparencyLog(key)
        self.queue=ReceiptReturnQueue(key,buffer_bytes);self.max_records=max_records
        self.max_content_bytes=max_content_bytes;self.durable_log_capacity_bytes=durable_log_capacity_bytes
        self.max_origin=max_origin;self.max_epoch=max_epoch;self.inclusion_deadline_epochs=inclusion_deadline_epochs
        self.acceptance_ids=[];self.terminal_ids=[];self.rejected_admission=[]
        self.durable_records={};self.return_backlog={};self.storage_bytes=0;self.storage_reserved_bytes=0
        self.peak_storage_bytes=0;self.proof_bytes=0;self.terminal_object_count=0
        self.flushed_batch_sizes=[]

    def accept(self,content,origin,epoch):
        content=bytes(content)
        if (len(content)>self.max_content_bytes or origin>self.max_origin or epoch>self.max_epoch
                or self.tree.size>=self.max_records):
            self.rejected_admission.append({'content_hash':hashlib.sha256(content).hexdigest(),
                'origin':origin,'epoch':epoch,'reason':'declared_durable_bound'})
            return None
        bound=durable_record_storage_bound(len(content),max_records=self.max_records,
            max_origin=self.max_origin,max_epoch=self.max_epoch,
            inclusion_deadline_epochs=self.inclusion_deadline_epochs)
        if self.storage_bytes+bound>self.durable_log_capacity_bytes:
            self.rejected_admission.append({'content_hash':hashlib.sha256(content).hexdigest(),
                'origin':origin,'epoch':epoch,'reason':'durable_log_capacity'})
            return None
        self.storage_reserved_bytes+=bound
        receipt=self.log.issue(content,policy_id='v6-terminal-component',accepted_at=epoch,
            inclusion_deadline=epoch+self.inclusion_deadline_epochs)
        index=self.tree.append(content);checkpoint=self.tree.checkpoint(timestamp_ms=epoch*3600000)
        proof=self.tree.inclusion_proof(index)
        witness=self.log.resolve(receipt.receipt_id,status='INCLUDED',witnessed_at=epoch,
            batch_id=str(self.tree.size),leaf_index=index,tree_size=self.tree.size,
            root_hex=self.tree.root().hex(),inclusion_proof=proof)
        acceptance=canonical_json({'kind':'acceptance','origin':origin,'receipt':asdict(receipt)})
        terminal=canonical_json({'kind':'terminal','origin':origin,'witness':asdict(witness),
                                 'checkpoint':asdict(checkpoint)})
        actual=len(content)+len(acceptance)+len(terminal)
        if actual>bound:raise AssertionError('durable object exceeds prospective storage bound')
        self.storage_reserved_bytes-=bound;self.storage_bytes+=actual
        self.peak_storage_bytes=max(self.peak_storage_bytes,self.storage_bytes)
        self.proof_bytes+=sum(len(x) for x in proof)
        self.durable_records[receipt.receipt_id]={'content':content,'receipt':receipt,
            'witness':witness,'checkpoint':checkpoint,'storage_bytes':actual}
        self.return_backlog.setdefault(origin,[]).extend(((receipt.receipt_id,'acceptance',acceptance,epoch),
                                                          (receipt.receipt_id,'terminal',terminal,epoch)))
        self.acceptance_ids.append(receipt.receipt_id);self.terminal_ids.append(receipt.receipt_id)
        self.terminal_object_count+=1;self.flushed_batch_sizes.append(1)
        return receipt

    def advance(self,epoch):pass
    def finalize(self,epoch):pass

    def contact(self,origin,epoch,budget,receiver):
        backlog=self.return_backlog.get(origin,[])
        if backlog:
            rid,kind,payload,issued=backlog[0];needed=len(fragment(payload,self.key))*FRAME_SIZE
            if self.queue.committed_occupancy+needed<=self.queue.buffer_bytes:
                if not self.queue.issue(payload,origin,issued,self.max_epoch):
                    raise AssertionError('durable return object unexpectedly rejected')
                backlog.pop(0)
        return self.queue.contact(origin,epoch,budget,receiver)

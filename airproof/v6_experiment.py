"""V6 causal collector integration, with bounded receipt return on real contacts."""
import json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .audit import IngestionReceipt, ReceiptStatusWitness, SignedTreeHead, verify_tree_head
from .v6_audit_return import (TerminalAuditReturn, SameOriginTerminalBatchAuditReturn,
                              DurableInclusionAsyncReturn, verify_returned_bundle)
from .records import canonical_json
from .v6_fairness import CumulativeServiceState
from .v6_receipts import Receiver
from .v6_transport import simulate_transport
from .v6_public_checkpoint import PublicCheckpointPublisher


class IntegratedCollector:
    def __init__(self, config, nodes, *, fairness=True):
        self.config=config; self.fairness=fairness
        self.backlog={}; self.selected=[]; self.selection_times={}; self.allocations=[]
        self.service=CumulativeServiceState()
        self.key=Ed25519PrivateKey.generate()
        audit=config.get('audit',{})
        max_return_frames=audit.get('max_return_frames')
        return_mode=audit.get('return_mode','per_record_v1')
        common=dict(max_records=int(audit.get('max_records',1000000)),
            buffer_bytes=int(audit.get('collector_return_buffer_bytes',18432)),
            max_return_frames=None if max_return_frames is None else int(max_return_frames))
        if return_mode=='per_record_v1':
            self.audit=TerminalAuditReturn(self.key,batch_size=1,**common)
        elif return_mode=='same_origin_terminal_batch_v1':
            self.audit=SameOriginTerminalBatchAuditReturn(self.key,
                batch_size=int(audit['terminal_batch_size']),
                batch_timeout_epochs=int(audit['terminal_batch_timeout_epochs']),
                inclusion_deadline_epochs=int(audit.get('inclusion_deadline_epochs',24)),**common)
        elif return_mode=='durable_inclusion_async_return_v1':
            if max_return_frames is not None:
                raise ValueError('durable async mode does not make admission depend on return-frame capacity')
            self.audit=DurableInclusionAsyncReturn(self.key,
                buffer_bytes=common['buffer_bytes'],max_records=common['max_records'],
                max_content_bytes=int(audit['max_content_bytes']),
                durable_log_capacity_bytes=int(audit['durable_log_capacity_bytes']),
                max_epoch=int(audit.get('max_epoch',10**9)),
                inclusion_deadline_epochs=int(audit.get('inclusion_deadline_epochs',24)))
        else:raise ValueError('unknown audit return mode')
        self.audit_return_mode=return_mode
        public_cfg=audit.get('public_checkpoint',{})
        self.publication=(PublicCheckpointPublisher(self.key,self.audit.tree,public_cfg)
                          if return_mode=='durable_inclusion_async_return_v1'
                          and public_cfg.get('enabled',False) else None)
        self.log=self.audit.log; self.returns=self.audit.queue
        self.contents={}; self.receipt_records={}; self.attempted=set()
        self.stale_rejected=[]
        self.receivers={i:Receiver(self.key.public_key()) for i in range(nodes)}
        self.accepted=set()

    def accept(self, record, epoch):
        if record.nullifier in self.attempted: raise ValueError('duplicate collector acceptance')
        self.attempted.add(record.nullifier)
        lag=int(self.config.get('transport',{}).get('useful_lag_epochs',6))
        if epoch>record.epoch+lag:
            self.stale_rejected.append({'nullifier':record.nullifier,'origin':record.user_id,
                'acquisition_epoch':record.epoch,'arrival_epoch':epoch,
                'reason':'past_useful_allocation_deadline'})
            return False
        content=canonical_json(record.public_dict())
        receipt=self.audit.accept(content,record.user_id,epoch)
        if receipt is None:return False
        self.accepted.add(record.nullifier)
        self.contents[receipt.receipt_id]=content
        self.receipt_records[receipt.receipt_id]=(record.user_id,receipt.inclusion_deadline)
        return True

    def returned_pairs(self):
        """Verify evidence actually reassembled at its origin, not collector state."""
        receipts={}; terminals={}
        for origin,receiver in self.receivers.items():
            for identity,payload in receiver.completed.items():
                obj=json.loads(payload)
                if obj['origin']!=origin:raise AssertionError('wrong origin return')
                if obj['kind']=='acceptance':
                    receipt=IngestionReceipt(**obj['receipt'])
                    receipts[receipt.receipt_id]=(receipt,self.returns.delivered[identity.hex()])
                elif obj['kind']=='terminal':
                    witness=ReceiptStatusWitness(**obj['witness'])
                    terminals[witness.receipt_id]=(witness,SignedTreeHead(**obj['checkpoint']),
                        self.returns.delivered[identity.hex()])
                elif obj['kind']=='terminal_batch':
                    head=SignedTreeHead(**obj['checkpoint'])
                    for raw_witness in obj['witnesses']:
                        witness=ReceiptStatusWitness(**raw_witness)
                        if witness.receipt_id in terminals:raise AssertionError('duplicate terminal witness')
                        terminals[witness.receipt_id]=(witness,head,self.returns.delivered[identity.hex()])
                else:raise AssertionError('unknown audit return object')
        verified=[]
        for rid,(receipt,received) in receipts.items():
            if rid not in terminals:continue
            witness,head,returned=terminals[rid]
            if (self.audit_return_mode!='durable_inclusion_async_return_v1'
                    and max(received,returned)>receipt.inclusion_deadline):continue
            if verify_returned_bundle(receipt,witness,head,self.contents[rid],self.key.public_key()):
                verified.append(rid)
        return verified

    def externally_checkpointed(self):
        receipt_ids=set()
        for origin,receiver in self.receivers.items():
            for payload in receiver.completed.values():
                obj=json.loads(payload)
                if obj.get('origin')!=origin:continue
                if obj.get('kind')=='terminal':
                    head=SignedTreeHead(**obj['checkpoint'])
                    if verify_tree_head(head,self.key.public_key()):
                        receipt_ids.add(obj['witness']['receipt_id'])
                elif obj.get('kind')=='terminal_batch':
                    head=SignedTreeHead(**obj['checkpoint'])
                    if verify_tree_head(head,self.key.public_key()):
                        receipt_ids.update(w['receipt_id'] for w in obj['witnesses'])
        return receipt_ids

    def allocate(self, arrivals, epoch):
        if hasattr(self.audit,'advance'):self.audit.advance(epoch)
        if self.publication is not None:self.publication.tick(epoch)
        lag=int(self.config.get('transport',{}).get('useful_lag_epochs',6))
        self.backlog.update((r.nullifier,r) for r in arrivals if r.nullifier in self.accepted)
        self.backlog={k:r for k,r in self.backlog.items() if epoch<=r.epoch+lag}
        scheduler=self.config.get('scheduler',{})
        groups=int(self.config.get('world',{}).get('groups',4))
        result=self.service.allocate(self.backlog.values(),
            budget_bytes=int(scheduler.get('budget_bytes_per_epoch',83200)),
            targets={g:int(scheduler.get('target_contributors',30)) for g in range(groups)},
            allocation_epoch=epoch,fairness=self.fairness)
        if result.violations: raise AssertionError(result.violations)
        for r in result.selected:
            self.backlog.pop(r.nullifier)
            self.selection_times[r.nullifier]=epoch
        self.selected.extend(result.selected)
        self.allocations.append({'epoch':epoch,'spent':result.spent_bytes,
            'counts':result.counts,'avoidable':result.avoidable_deficits,
            'unavoidable':result.unavoidable_deficits,'feasible':result.globally_feasible})
        return result.selected

    def return_receipt(self,node,epoch,budget):
        if self.audit_return_mode=='durable_inclusion_async_return_v1':
            return self.audit.contact(node,epoch,budget,self.receivers[node])
        return self.returns.contact(node,epoch,budget,self.receivers[node])


def integrated_transport(trace,observations,config,*,policy='combined',fairness=True):
    collector=IntegratedCollector(config,trace.nodes,fairness=fairness)
    result=simulate_transport(trace,observations,config,policy,
        collector_selector=collector.allocate,raw_acceptor=collector.accept,
        receipt_return=collector.return_receipt)
    collector.audit.finalize(trace.acquisition_epochs+trace.drain_epochs-1)
    # Expire even when no later gateway contact occurs.
    collector.returns.expire(trace.acquisition_epochs+trace.drain_epochs-1)
    pairs=collector.returned_pairs()
    external=collector.externally_checkpointed()
    publication=collector.publication
    capacity_rejected=len(collector.audit.rejected_admission)
    stale_rejected=len(collector.stale_rejected)
    result.metrics.update(receipt_issued=len(collector.accepted),
        audit_attempted=len(collector.attempted),
        audit_admission_rejected=capacity_rejected+stale_rejected,
        audit_capacity_rejected=capacity_rejected,
        audit_stale_rejected=stale_rejected,
        audit_terminal_issued=len(collector.audit.terminal_ids),
        audit_terminal_objects_issued=getattr(collector.audit,'terminal_object_count',len(collector.audit.terminal_ids)),
        audit_return_mode=collector.audit_return_mode,
        audit_flushed_batch_sizes=getattr(collector.audit,'flushed_batch_sizes',[1]*len(collector.audit.terminal_ids)),
        audit_verified_return_pairs=len(pairs),
        audit_externally_checkpointed=len(external),
        audit_publicly_checkpointed=0 if publication is None else publication.publicly_checkpointed,
        audit_public_checkpoint_raw_bytes=0 if publication is None else publication.queue.transmitted_bytes,
        audit_public_checkpoint_control_bytes=0 if publication is None else publication.queue.control_bytes,
        audit_public_checkpoint_expired=0 if publication is None else len(publication.queue.expired),
        audit_public_checkpoint_pending=0 if publication is None else len(publication.queue.pending),
        audit_public_checkpoint_peak_buffer_bytes=0 if publication is None else publication.queue.peak_buffer,
        audit_public_checkpoint_reassembly_peak_bytes=0 if publication is None else publication.auditor.peak_buffer,
        audit_public_checkpoint_contacts=0 if publication is None else publication.uplink_contacts,
        audit_public_checkpoint_heads_verified=0 if publication is None else len(publication.accepted_heads),
        audit_public_checkpoint_signature_failures=0 if publication is None else publication.signature_failures,
        audit_public_checkpoint_rollback_detections=0 if publication is None else publication.rollback_detections,
        audit_public_checkpoint_split_view_detections=0 if publication is None else publication.split_view_detections,
        audit_public_checkpoint_consistency_failures=0 if publication is None else publication.consistency_failures,
        audit_unreturned_pairs=len(collector.accepted)-len(pairs),
        audit_return_control_bytes=collector.returns.control_bytes,
        audit_peak_committed_buffer_bytes=collector.returns.peak_committed_buffer,
        audit_durable_storage_bytes=getattr(collector.audit,'storage_bytes',0),
        audit_durable_storage_capacity_bytes=getattr(collector.audit,'durable_log_capacity_bytes',0),
        audit_durable_storage_peak_bytes=getattr(collector.audit,'peak_storage_bytes',0),
        audit_inclusion_proof_bytes=getattr(collector.audit,'proof_bytes',0),
        audit_collector_log=collector.log.audit(now=trace.acquisition_epochs+trace.drain_epochs-1),
        receipt_returned=len(collector.returns.delivered),
        receipt_buffer_dropped=len(collector.returns.dropped),
        receipt_expired=len(collector.returns.expired),
        receipt_pending=len(collector.returns.pending),
        receipt_peak_buffer_bytes=collector.returns.peak_buffer,
        receipt_reassembly_peak_bytes=max((r.peak_buffer for r in collector.receivers.values()),default=0),
        receipt_durable_storage_bytes=sum(len(p) for r in collector.receivers.values() for p in r.completed.values()),
        receipt_scope='useful-deadline and capacity admission before prompt per-record signed acceptance; terminal return mode is configured and physical arrival remains charged and ACKed')
    return result,collector

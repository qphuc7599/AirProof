"""Finite opt-in publication of signed append-only checkpoints."""
from __future__ import annotations

import base64,json
from dataclasses import asdict

from .audit import SignedTreeHead,verify_consistency,verify_tree_head
from .records import canonical_json
from .v6_receipts import ReceiptReturnQueue,Receiver
from .v6_resources import ContactBudget


class PublicCheckpointPublisher:
    def __init__(self,key,tree,config):
        epochs=tuple(int(x) for x in config['uplink_epochs'])
        if tuple(sorted(set(epochs)))!=epochs or any(x<0 for x in epochs):
            raise ValueError('public uplink epochs must be unique causal order')
        self.key=key;self.tree=tree;self.uplink_epochs=set(epochs);self.schedule=epochs
        self.capacity=int(config['capacity_bytes_per_direction'])
        self.release_reserved=int(config['release_reserved_bytes'])
        self.control_budget=int(config['control_budget_bytes']);self.expiry=int(config['expiry_epochs'])
        if min(self.capacity,self.control_budget,self.expiry)<1 or self.release_reserved<0:
            raise ValueError('invalid finite public uplink resources')
        self.queue=ReceiptReturnQueue(key,int(config['queue_buffer_bytes']))
        self.auditor=Receiver(key.public_key(),int(config['auditor_reassembly_buffer_bytes']))
        self.last_head=None;self.processed=set();self.accepted_heads=[]
        self.signature_failures=0;self.rollback_detections=0;self.split_view_detections=0
        self.consistency_failures=0;self.uplink_contacts=0;self.enqueued=0

    @property
    def publicly_checkpointed(self):return 0 if self.last_head is None else self.last_head.tree_size

    def _payload(self,epoch):
        head=self.tree.checkpoint(timestamp_ms=epoch*3600000)
        if self.last_head is None:
            previous_size=0;previous_root='';proof=()
        else:
            previous_size=self.last_head.tree_size;previous_root=self.last_head.root_hex
            proof=self.tree.consistency_proof(previous_size,head.tree_size)
        return canonical_json({'kind':'public_checkpoint','previous_size':previous_size,
            'previous_root_hex':previous_root,'consistency_proof_b64':[
                base64.b64encode(x).decode() for x in proof], 'checkpoint':asdict(head)})

    def process_payload(self,payload):
        try:
            obj=json.loads(payload);head=SignedTreeHead(**obj['checkpoint'])
            proof=tuple(base64.b64decode(x) for x in obj['consistency_proof_b64'])
        except (KeyError,TypeError,ValueError,json.JSONDecodeError):
            self.signature_failures+=1;return False
        if obj.get('kind')!='public_checkpoint' or not verify_tree_head(head,self.key.public_key()):
            self.signature_failures+=1;return False
        if self.last_head is None:
            if obj['previous_size']!=0 or obj['previous_root_hex']!='' or proof:
                self.consistency_failures+=1;return False
        else:
            if head.tree_size<self.last_head.tree_size:
                self.rollback_detections+=1;return False
            if head.tree_size==self.last_head.tree_size and head.root_hex!=self.last_head.root_hex:
                self.split_view_detections+=1;return False
            if (obj['previous_size']!=self.last_head.tree_size
                    or obj['previous_root_hex']!=self.last_head.root_hex):
                self.consistency_failures+=1;return False
            if not verify_consistency(old_size=self.last_head.tree_size,new_size=head.tree_size,
                    old_root=bytes.fromhex(self.last_head.root_hex),new_root=bytes.fromhex(head.root_hex),proof=proof):
                self.consistency_failures+=1;return False
        self.last_head=head;self.accepted_heads.append(head);return True

    def tick(self,epoch):
        self.queue.expire(epoch)
        if epoch not in self.uplink_epochs:return
        self.uplink_contacts+=1
        budget=ContactBudget(self.capacity,self.release_reserved,self.control_budget)
        self.queue.contact(0,epoch,budget,self.auditor)
        for identity,payload in self.auditor.completed.items():
            if identity in self.processed:continue
            self.processed.add(identity);self.process_payload(payload)
        if not self.queue.pending and self.tree.size>self.publicly_checkpointed:
            payload=self._payload(epoch)
            self.queue.issue(payload,0,epoch,epoch+self.expiry)
            self.enqueued+=1

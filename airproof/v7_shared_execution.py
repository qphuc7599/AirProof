"""Shared-resource execution used by the reviewer revision.

The frozen v6 modules remain unchanged. This additive adapter lets one common
transport/audit execution use either the retained v4 allocator or the v6 exact
representative minimax allocator under an explicit versioned name.
"""
from __future__ import annotations

import base64
import hashlib
from collections import Counter
from dataclasses import asdict

from .audit import (
    ReceiptAccountabilityLog,
    SignedTreeHead,
    _leaf_hash,
    _node_hash,
    _subproof,
    _tree_hash,
)
from .records import canonical_json
from .scheduler import select_evidence
from .v6_audit_return import durable_record_storage_bound
from .v6_experiment import IntegratedCollector
from .v6_receipts import ReceiptReturnQueue, fragment
from .v6_public_checkpoint import PublicCheckpointPublisher
from .v6_transport import simulate_transport


class IncrementalAppendLog:
    """RFC-9162 tree frontier supporting the newest-leaf proof in O(log n)."""

    def __init__(
        self,
        key,
        *,
        log_id="airproof-v7-selected-log-v1",
        retain_leaves=False,
    ):
        self.key = key
        self.log_id = log_id
        self.frontier = {}
        self._leaves = [] if retain_leaves else None
        self._size = 0
        self._root = hashlib.sha256(b"").digest()

    @property
    def size(self):
        return self._size

    def append_with_proof(self, payload):
        payload = bytes(payload)
        proof = tuple(self.frontier[level] for level in sorted(self.frontier))
        carry = _leaf_hash(payload)
        level = 0
        while level in self.frontier:
            carry = _node_hash(self.frontier.pop(level), carry)
            level += 1
        self.frontier[level] = carry
        root = _leaf_hash(bytes(payload))
        for sibling in proof:
            root = _node_hash(sibling, root)
        index = self._size
        self._size += 1
        self._root = root
        if self._leaves is not None:
            self._leaves.append(payload)
        return index, proof

    def root(self):
        return self._root

    def consistency_proof(self, old_size, new_size=None):
        """Return an RFC-9162 consistency proof when publication is enabled.

        The normal selected-evidence path keeps only the logarithmic frontier.
        A common execution that requests public checkpoints explicitly retains
        the selected payload leaves so its externally delivered heads can be
        related by a standard consistency proof.  This opt-in storage is
        charged separately by the caller and does not affect numerical input.
        """
        if self._leaves is None:
            raise RuntimeError("consistency proofs require retained leaves")
        size = self.size if new_size is None else int(new_size)
        if not 0 <= int(old_size) <= size <= self.size:
            raise ValueError("invalid consistency range")
        if int(old_size) in (0, size):
            return ()
        leaves = tuple(self._leaves[:size])
        if size == self.size and _tree_hash(leaves) != self.root():
            raise AssertionError("incremental and retained-leaf roots differ")
        return tuple(_subproof(int(old_size), leaves, True))

    def checkpoint(self, *, timestamp_ms):
        payload = {
            "log_id": self.log_id,
            "tree_size": self.size,
            "root_hex": self.root().hex(),
            "timestamp_ms": int(timestamp_ms),
        }
        signature = self.key.sign(canonical_json(payload))
        return SignedTreeHead(
            **payload, signature_b64=base64.b64encode(signature).decode()
        )


class IncrementalSelectedAuditReturn:
    """Durable selected-evidence receipts with logarithmic append proofs."""

    def __init__(
        self,
        key,
        *,
        buffer_bytes,
        max_records,
        max_content_bytes,
        durable_log_capacity_bytes,
        max_origin,
        max_epoch,
        inclusion_deadline_epochs,
        retain_leaves=False,
    ):
        values = (
            buffer_bytes,
            max_records,
            max_content_bytes,
            durable_log_capacity_bytes,
            max_origin,
            max_epoch,
            inclusion_deadline_epochs,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("positive selected-audit bounds required")
        self.key = key
        self.log = ReceiptAccountabilityLog(key)
        self.tree = IncrementalAppendLog(key, retain_leaves=retain_leaves)
        self.queue = ReceiptReturnQueue(key, buffer_bytes)
        self.max_records = max_records
        self.max_content_bytes = max_content_bytes
        self.durable_log_capacity_bytes = durable_log_capacity_bytes
        self.max_origin = max_origin
        self.max_epoch = max_epoch
        self.inclusion_deadline_epochs = inclusion_deadline_epochs
        self.acceptance_ids = []
        self.terminal_ids = []
        self.rejected_admission = []
        self.durable_records = {}
        self.return_backlog = {}
        self.storage_bytes = 0
        self.peak_storage_bytes = 0
        self.proof_bytes = 0
        self.terminal_object_count = 0
        self.flushed_batch_sizes = []

    def accept(self, content, origin, epoch):
        content = bytes(content)
        if (
            len(content) > self.max_content_bytes
            or origin > self.max_origin
            or epoch > self.max_epoch
            or self.tree.size >= self.max_records
        ):
            self.rejected_admission.append(
                {
                    "content_hash": hashlib.sha256(content).hexdigest(),
                    "origin": origin,
                    "epoch": epoch,
                    "reason": "declared_selected_audit_bound",
                }
            )
            return None
        bound = durable_record_storage_bound(
            len(content),
            max_records=self.max_records,
            max_origin=self.max_origin,
            max_epoch=self.max_epoch,
            inclusion_deadline_epochs=self.inclusion_deadline_epochs,
        )
        if self.storage_bytes + bound > self.durable_log_capacity_bytes:
            self.rejected_admission.append(
                {
                    "content_hash": hashlib.sha256(content).hexdigest(),
                    "origin": origin,
                    "epoch": epoch,
                    "reason": "durable_log_capacity",
                }
            )
            return None
        receipt = self.log.issue(
            content,
            policy_id="v7-selected-evidence",
            accepted_at=epoch,
            inclusion_deadline=epoch + self.inclusion_deadline_epochs,
        )
        index, proof = self.tree.append_with_proof(content)
        checkpoint = self.tree.checkpoint(timestamp_ms=epoch * 3_600_000)
        witness = self.log.resolve(
            receipt.receipt_id,
            status="INCLUDED",
            witnessed_at=epoch,
            batch_id=str(self.tree.size),
            leaf_index=index,
            tree_size=self.tree.size,
            root_hex=self.tree.root().hex(),
            inclusion_proof=proof,
        )
        acceptance = canonical_json(
            {"kind": "acceptance", "origin": origin, "receipt": asdict(receipt)}
        )
        terminal = canonical_json(
            {
                "kind": "terminal",
                "origin": origin,
                "witness": asdict(witness),
                "checkpoint": asdict(checkpoint),
            }
        )
        actual = len(content) + len(acceptance) + len(terminal)
        if actual > bound:
            raise AssertionError("selected audit object exceeds storage bound")
        self.storage_bytes += actual
        self.peak_storage_bytes = max(self.peak_storage_bytes, self.storage_bytes)
        self.proof_bytes += sum(len(item) for item in proof)
        self.durable_records[receipt.receipt_id] = {
            "content": content,
            "receipt": receipt,
            "witness": witness,
            "checkpoint": checkpoint,
            "storage_bytes": actual,
        }
        self.return_backlog.setdefault(origin, []).extend(
            (
                (receipt.receipt_id, "acceptance", acceptance, epoch),
                (receipt.receipt_id, "terminal", terminal, epoch),
            )
        )
        self.acceptance_ids.append(receipt.receipt_id)
        self.terminal_ids.append(receipt.receipt_id)
        self.terminal_object_count += 1
        self.flushed_batch_sizes.append(1)
        return receipt

    def advance(self, epoch):
        return None

    def finalize(self, epoch):
        return None

    def contact(self, origin, epoch, budget, receiver):
        backlog = self.return_backlog.get(origin, [])
        if backlog:
            _, _, payload, issued = backlog[0]
            needed = len(fragment(payload, self.key)) * 512
            if self.queue.committed_occupancy + needed <= self.queue.buffer_bytes:
                if not self.queue.issue(payload, origin, issued, self.max_epoch):
                    raise AssertionError("selected audit return unexpectedly rejected")
                backlog.pop(0)
        return self.queue.contact(origin, epoch, budget, receiver)


class ReviewerIntegratedCollector(IntegratedCollector):
    """Integrated collector with a registered allocation policy."""

    def __init__(self, config, nodes, *, fairness=True, allocation_policy=None):
        audit_mode = config.get("audit", {}).get("return_mode")
        if audit_mode == "incremental_selected_receipts_v1":
            bootstrap_config = dict(config)
            bootstrap_audit = dict(config.get("audit", {}))
            bootstrap_audit["return_mode"] = "durable_inclusion_async_return_v1"
            bootstrap_audit["public_checkpoint"] = {"enabled": False}
            bootstrap_config["audit"] = bootstrap_audit
            super().__init__(bootstrap_config, nodes, fairness=fairness)
            self.config = config
            audit = config["audit"]
            self.audit = IncrementalSelectedAuditReturn(
                self.key,
                buffer_bytes=int(audit["collector_return_buffer_bytes"]),
                max_records=int(audit["max_records"]),
                max_content_bytes=int(audit["max_content_bytes"]),
                durable_log_capacity_bytes=int(audit["durable_log_capacity_bytes"]),
                max_origin=int(audit.get("max_origin", 2**31 - 1)),
                max_epoch=int(audit.get("max_epoch", 10**9)),
                inclusion_deadline_epochs=int(audit["inclusion_deadline_epochs"]),
                retain_leaves=bool(audit.get("public_checkpoint", {}).get("enabled", False)),
            )
            self.audit_return_mode = audit_mode
            public_cfg = audit.get("public_checkpoint", {})
            self.publication = (
                PublicCheckpointPublisher(self.key, self.audit.tree, public_cfg)
                if public_cfg.get("enabled", False)
                else None
            )
            self.log = self.audit.log
            self.returns = self.audit.queue
        else:
            super().__init__(config, nodes, fairness=fairness)
        self.audit_selected_attempted = set()
        scheduler = config.get("scheduler", {})
        self.allocation_policy = allocation_policy or scheduler.get(
            "algorithm", "v6_exact_minimax_representatives"
        )
        if self.allocation_policy not in {
            "v4_outcome_effective_hard_if_feasible",
            "v6_exact_minimax_representatives",
        }:
            raise ValueError("unknown reviewer allocation policy")

    def accept(self, record, epoch):
        """Admit a timely raw candidate; durable audit starts only if selected.

        This prevents unallocated network arrivals from creating quadratic audit
        work or consuming the accountability contract. The shared execution
        reports raw candidate admission and selected-evidence receipts separately.
        """
        if record.nullifier in self.attempted:
            raise ValueError("duplicate collector acceptance")
        self.attempted.add(record.nullifier)
        lag = int(self.config.get("transport", {}).get("useful_lag_epochs", 6))
        if epoch > record.epoch + lag:
            self.stale_rejected.append(
                {
                    "nullifier": record.nullifier,
                    "origin": record.user_id,
                    "acquisition_epoch": record.epoch,
                    "arrival_epoch": epoch,
                    "reason": "past_useful_allocation_deadline",
                }
            )
            return False
        self.accepted.add(record.nullifier)
        return True

    def _commit_selected_audit(self, records, epoch):
        for record in records:
            if record.nullifier in self.audit_selected_attempted:
                raise ValueError("selected evidence received duplicate audit influence")
            self.audit_selected_attempted.add(record.nullifier)
            content = canonical_json(record.public_dict())
            receipt = self.audit.accept(content, record.user_id, epoch)
            if receipt is None:
                raise RuntimeError("registered audit storage rejected selected evidence")
            self.contents[receipt.receipt_id] = content
            self.receipt_records[receipt.receipt_id] = (
                record.user_id,
                receipt.inclusion_deadline,
            )

    def return_receipt(self, node, epoch, budget):
        if self.audit_return_mode == "incremental_selected_receipts_v1":
            return self.audit.contact(node, epoch, budget, self.receivers[node])
        return super().return_receipt(node, epoch, budget)

    def allocate(self, arrivals, epoch):
        if self.allocation_policy == "v6_exact_minimax_representatives":
            selected = super().allocate(arrivals, epoch)
            self._commit_selected_audit(selected, epoch)
            self.allocations[-1]["allocator"] = self.allocation_policy
            return selected

        if hasattr(self.audit, "advance"):
            self.audit.advance(epoch)
        if self.publication is not None:
            self.publication.tick(epoch)
        lag = int(self.config.get("transport", {}).get("useful_lag_epochs", 6))
        self.backlog.update(
            (record.nullifier, record)
            for record in arrivals
            if record.nullifier in self.accepted
        )
        self.backlog = {
            key: record
            for key, record in self.backlog.items()
            if epoch <= record.epoch + lag
        }
        scheduler = self.config.get("scheduler", {})
        groups = int(self.config.get("world", {}).get("groups", 4))
        targets = {
            group: int(scheduler.get("target_contributors", 30))
            for group in range(groups)
        }
        result = select_evidence(
            self.backlog.values(),
            budget_bytes=int(scheduler.get("budget_bytes_per_epoch", 83_200)),
            reserve_fraction=float(scheduler.get("reserve_fraction", 0.25)),
            targets=targets,
            fairness=self.fairness,
            fairness_strength=float(scheduler.get("fairness_strength", 1.0)),
            constraint_mode=str(scheduler.get("constraint_mode", "hard_if_feasible")),
            allocation_epoch=epoch,
        )
        available = Counter(
            group for _, group in {
                (int(record.user_id), int(record.group))
                for record in self.backlog.values()
            }
        )
        unavoidable = {
            group: max(0.0, (target - min(target, available[group])) / target)
            for group, target in targets.items()
        }
        avoidable = {
            group: max(
                0.0,
                (min(target, available[group]) - result.counts.get(group, 0)) / target,
            )
            for group, target in targets.items()
        }
        if result.globally_feasible and self.fairness and not result.constraint_satisfied:
            raise AssertionError("feasible v4 floor was not satisfied")
        self._commit_selected_audit(result.selected, epoch)
        for record in result.selected:
            self.backlog.pop(record.nullifier)
            self.selection_times[record.nullifier] = epoch
        self.selected.extend(result.selected)
        self.allocations.append(
            {
                "epoch": epoch,
                "spent": result.spent_bytes,
                "counts": result.counts,
                "avoidable": avoidable,
                "unavoidable": unavoidable,
                "feasible": result.globally_feasible,
                "constraint_satisfied": result.constraint_satisfied,
                "opportunity_shortfall_groups": result.opportunity_shortfall_groups,
                "budget_infeasible_groups": result.budget_infeasible_groups,
                "allocator": self.allocation_policy,
            }
        )
        return result.selected


def _finalize_metrics(result, collector, trace):
    final_epoch = trace.acquisition_epochs + trace.drain_epochs - 1
    collector.audit.finalize(final_epoch)
    collector.returns.expire(final_epoch)
    pairs = collector.returned_pairs()
    external = collector.externally_checkpointed()
    publication = collector.publication
    capacity_rejected = len(collector.audit.rejected_admission)
    stale_rejected = len(collector.stale_rejected)
    result.metrics.update(
        allocation_policy=collector.allocation_policy,
        raw_candidate_attempted=len(collector.attempted),
        raw_candidates_accepted=len(collector.accepted),
        receipt_issued=len(collector.contents),
        audit_attempted=len(collector.audit_selected_attempted),
        audit_capacity_rejected=capacity_rejected,
        audit_stale_rejected=stale_rejected,
        audit_admission_rejected=capacity_rejected + stale_rejected,
        audit_terminal_issued=len(collector.audit.terminal_ids),
        audit_verified_return_pairs=len(pairs),
        audit_externally_checkpointed=len(external),
        audit_publicly_checkpointed=(
            0 if publication is None else publication.publicly_checkpointed
        ),
        audit_public_checkpoint_raw_bytes=(
            0 if publication is None else publication.queue.transmitted_bytes
        ),
        audit_public_checkpoint_control_bytes=(
            0 if publication is None else publication.queue.control_bytes
        ),
        audit_public_checkpoint_expired=(
            0 if publication is None else len(publication.queue.expired)
        ),
        audit_public_checkpoint_pending=(
            0 if publication is None else len(publication.queue.pending)
        ),
        audit_public_checkpoint_heads_verified=(
            0 if publication is None else len(publication.accepted_heads)
        ),
        audit_public_checkpoint_peak_buffer_bytes=(
            0 if publication is None else publication.queue.peak_buffer
        ),
        audit_public_checkpoint_reassembly_peak_bytes=(
            0 if publication is None else publication.auditor.peak_buffer
        ),
        audit_public_checkpoint_contacts=(
            0 if publication is None else publication.uplink_contacts
        ),
        audit_public_checkpoint_signature_failures=(
            0 if publication is None else publication.signature_failures
        ),
        audit_public_checkpoint_rollback_detections=(
            0 if publication is None else publication.rollback_detections
        ),
        audit_public_checkpoint_split_view_detections=(
            0 if publication is None else publication.split_view_detections
        ),
        audit_public_checkpoint_consistency_failures=(
            0 if publication is None else publication.consistency_failures
        ),
        audit_unreturned_pairs=len(collector.contents) - len(pairs),
        audit_return_control_bytes=collector.returns.control_bytes,
        audit_durable_storage_bytes=getattr(collector.audit, "storage_bytes", 0),
        audit_durable_storage_capacity_bytes=getattr(
            collector.audit, "durable_log_capacity_bytes", 0
        ),
        audit_durable_storage_peak_bytes=getattr(
            collector.audit, "peak_storage_bytes", 0
        ),
        audit_inclusion_proof_bytes=getattr(collector.audit, "proof_bytes", 0),
        audit_collector_log=collector.log.audit(now=final_epoch),
        receipt_returned=len(collector.returns.delivered),
        receipt_buffer_dropped=len(collector.returns.dropped),
        receipt_expired=len(collector.returns.expired),
        receipt_pending=len(collector.returns.pending),
        receipt_peak_buffer_bytes=collector.returns.peak_buffer,
        receipt_reassembly_peak_bytes=max(
            (receiver.peak_buffer for receiver in collector.receivers.values()), default=0
        ),
        receipt_durable_storage_bytes=sum(
            len(payload)
            for receiver in collector.receivers.values()
            for payload in receiver.completed.values()
        ),
        receipt_scope=(
            "capacity admission before signed acceptance; terminal proofs, "
            "return payloads and ACKs consume finite reverse-contact capacity"
        ),
    )
    return result, collector


def integrated_transport_reviewer(
    trace,
    observations,
    config,
    *,
    policy="combined",
    fairness=True,
    allocation_policy=None,
):
    """Run raw, reserved release, allocation and audit queues on one trace."""
    collector = ReviewerIntegratedCollector(
        config,
        trace.nodes,
        fairness=fairness,
        allocation_policy=allocation_policy,
    )
    result = simulate_transport(
        trace,
        observations,
        config,
        policy,
        collector_selector=collector.allocate,
        raw_acceptor=collector.accept,
        receipt_return=collector.return_receipt,
    )
    return _finalize_metrics(result, collector, trace)

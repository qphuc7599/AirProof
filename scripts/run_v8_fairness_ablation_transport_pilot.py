#!/usr/bin/env python3
"""Exposed one-world check that the Gate-A no-fairness ablation is complete."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airproof.metrics import coverage_metrics  # noqa: E402
from airproof.simulator import generate_world  # noqa: E402
from airproof.v6_mobility import coupled_public_trace  # noqa: E402
from airproof.v7_shared_execution import integrated_transport_reviewer  # noqa: E402
from scripts.run_v7_shared_resource_confirmation import _config  # noqa: E402


def main() -> None:
    registration = json.loads(
        (ROOT / "configs/v8/reviewer_public_checkpoint_development_v1.json").read_text()
    )
    seed = 6251100
    config = _config(registration, "severe_clean")
    world = generate_world(config, seed)
    trace, _ = coupled_public_trace(config, seed, world.observations)
    output = ROOT / "reports/v8/fairness_ablation_transport_pilot_v2"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, policy, fairness in (
        ("FULL_FAIRNESS", "combined_group_fair", True),
        ("NO_FAIRNESS", "combined_no_service_priority", False),
    ):
        started = time.perf_counter()
        transport, collector = integrated_transport_reviewer(
            trace,
            world.observations,
            config,
            policy=policy,
            fairness=fairness,
            allocation_policy=registration["execution"]["allocation_policy"],
        )
        burn = int(config["world"]["burn_in_steps"])
        stop = int(config["world"]["steps"])
        selected = [item for item in collector.selected if burn <= item.epoch < stop]
        rows.append(
            {
                "arm": name,
                "policy": policy,
                "fairness": fairness,
                "trace_hash": trace.trace_hash,
                "coverage": coverage_metrics(selected, int(config["world"]["groups"])),
                "selected_count": len(selected),
                "raw_timely_delivered": transport.metrics["raw_timely_delivered"],
                "total_wire_bytes": transport.metrics["total_wire_bytes"],
                "control_bytes": transport.metrics["control_bytes"],
                "token_violations": transport.metrics["token_violations"],
                "audit_publicly_checkpointed": transport.metrics["audit_publicly_checkpointed"],
                "receipt_issued": transport.metrics["receipt_issued"],
                "runtime_seconds": time.perf_counter() - started,
            }
        )
    fair, nofair = rows
    result = {
        "schema_version": 1,
        "role": "exposed mechanism diagnostic only",
        "seed": seed,
        "same_trace": fair["trace_hash"] == nofair["trace_hash"],
        "coverage_gap_reduction": 1.0 - fair["coverage"]["coverage_gap"] / nofair["coverage"]["coverage_gap"],
        "arms": rows,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

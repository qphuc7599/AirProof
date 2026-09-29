"""Descriptive statistics of the completed 3-block/30-anchor public-network sample."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign
    anchors = json.loads((root / "anchors.json").read_text(encoding="utf-8"))
    completion = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(anchors) != 30 or completion["anchors"] != 30 or len({row["transaction_hash"] for row in anchors}) != 30:
        raise ValueError("missing or duplicated public transaction evidence")
    if sorted(row["replicate"] for row in anchors) != list(range(30)):
        raise ValueError("replicate mismatch")
    if {block: sum(row["time_block"] == block for row in anchors) for block in range(3)} != {0: 10, 1: 10, 2: 10}:
        raise ValueError("three time blocks with ten anchors each required")
    if any(row["status"] != 1 or not row["readback_verified"] or row["reverted"] for row in anchors):
        raise ValueError("a transaction/contract readback did not succeed")
    if any(row["confirmations"] != 2 or row["time_to_confirmation_seconds"] < row["time_to_receipt_seconds"] for row in anchors):
        raise ValueError("confirmation-clock mismatch")
    metrics = ("gas_used", "time_to_receipt_seconds", "time_to_confirmation_seconds", "fee_test_eth")
    def describe(rows):
        result = {}
        for name in metrics:
            values = np.array([float(row[name]) for row in rows])
            result[name] = {"mean": float(values.mean()), "median": float(np.median(values)),
                            "p95": float(np.quantile(values, .95)), "min": float(values.min()),
                            "max": float(values.max())}
        return result
    deployments = [event for event in events if event.get("event") == "receipt" and event.get("label") == "deployment"]
    if len(deployments) != 1 or deployments[0]["status"] != 1 or deployments[0]["reverted"]:
        raise ValueError("one successful deployment receipt required")
    summary = {"role": "public-network-feasibility-not-throughput-or-finality-latency-benchmark",
               "anchors": 30, "time_blocks": 3, "all_readbacks_verified": True,
               "contract_address": completion["contract_address"],
               "overall": describe(anchors),
               "by_block": {str(block): describe([row for row in anchors if row["time_block"] == block]) for block in range(3)},
               "anchor_fees_total_test_eth": float(sum(float(row["fee_test_eth"]) for row in anchors)),
               "deployment_receipts": deployments,
               "rpc_failures": completion["rpc_failures"], "reverted_transactions": completion["reverted_transactions"],
               "confirmations": 2, "anchors_consensus_finalized_at_measurement_end": completion["anchors_finalized_at_end"],
               "timing_scope": "receipt availability and two block confirmations; neither is consensus finalization time",
               "statistical_scope": "descriptive sample across three time blocks, not thirty independent network regimes",
               "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("anchors.json", "summary.json", "events.jsonl")}}
    (root / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"anchors": 30, "overall": summary["overall"], "rpc_failures": summary["rpc_failures"]}, indent=2))


if __name__ == "__main__":
    main()

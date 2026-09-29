#!/usr/bin/env python3
"""Create a compact table source for the shared-resource reviewer experiment."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/v7/reviewer_revision/shared_resource_confirmation_v2"
OUTPUT = BASE / "analysis_detail.json"


def _mean(rows: list[dict], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def execute() -> dict:
    result_paths = sorted(BASE.glob("jobs/*/*/result.json"))
    rows = [json.loads(path.read_text()) for path in result_paths]
    if len(rows) != 24:
        raise ValueError(f"expected 24 completed jobs, found {len(rows)}")
    cells: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cells[row["cell"]].append(row)
    cell_summary = {}
    for cell, subset in sorted(cells.items()):
        methods: dict[str, list[dict]] = defaultdict(list)
        for row in subset:
            for method in row["method_rows"]:
                methods[method["method"]].append(method)
        cell_summary[cell] = {
            "methods": {
                name: {
                    "rmse_mean": float(np.mean([item["rmse"] for item in values])),
                    "live_rmse_mean": float(np.mean([item["live_rmse"] for item in values])),
                    "event_recall_mean": float(np.mean([item["event_recall"] for item in values])),
                    "worst_group_rmse_mean": float(np.mean([item["worst_group_rmse"] for item in values])),
                }
                for name, values in sorted(methods.items())
            },
            "resource_means": {
                "raw_generated": _mean(subset, ("transport", "raw_generated")),
                "raw_timely_delivered": _mean(subset, ("transport", "raw_timely_delivered")),
                "release_delivered": _mean(subset, ("transport", "release_delivered")),
                "selected_numerical_records": _mean(subset, ("selected_count",)),
                "receipt_issued": _mean(subset, ("transport", "receipt_issued")),
                "receipt_returned_frames": _mean(subset, ("transport", "receipt_returned")),
                "receipt_verified_pairs": _mean(subset, ("transport", "audit_verified_return_pairs")),
                "total_wire_bytes": _mean(subset, ("transport", "total_wire_bytes")),
            },
            "fairness_means": {
                "coverage_gap": _mean(subset, ("fairness", "coverage_gap")),
                "globally_feasible_epochs": _mean(subset, ("fairness", "globally_feasible_epochs")),
                "max_avoidable_deficit": _mean(subset, ("fairness", "mean_max_avoidable_deficit")),
                "max_unavoidable_deficit": _mean(subset, ("fairness", "mean_max_unavoidable_deficit")),
            },
            "lifetime_means": {
                "effective_records": _mean(subset, ("lifetime_exposure", "effective_records")),
                "dropped_exhausted_records": _mean(subset, ("lifetime_exposure", "dropped_exhausted_records")),
                "maximum_user_exposure": _mean(subset, ("lifetime_exposure", "maximum_user_lifetime_exposure")),
            },
        }
    paired = [json.loads(line) for line in (BASE / "paired_worlds.jsonl").read_text().splitlines() if line]
    attack = {
        "worlds": len(paired),
        "sq_positive_growth_worlds": sum(row["hotspot_growth_sq"] > 0 for row in paired),
        "sq_zero_growth_worlds": sum(row["hotspot_growth_sq"] == 0 for row in paired),
        "sq_negative_growth_worlds": sum(row["hotspot_growth_sq"] < 0 for row in paired),
        "mean_sq_growth": float(np.mean([row["hotspot_growth_sq"] for row in paired])),
        "mean_ap_growth": float(np.mean([row["hotspot_growth_ap"] for row in paired])),
        "diagnosis": "first-arrival B=7 exposure is exhausted before the late hotspot in most worlds",
    }
    result = {
        "schema_version": 1,
        "role": "compact manuscript table source for one shared-resource execution",
        "source_result_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in result_paths},
        "registered_analysis": json.loads((BASE / "analysis.json").read_text()),
        "cells": cell_summary,
        "late_attack_diagnosis": attack,
        "privacy_comparison": json.loads((BASE / "privacy_comparison.json").read_text()),
        "matched_no_fairness": json.loads(
            (ROOT / "reports/v7/reviewer_revision/shared_fairness_control_v1/analysis.json").read_text()
        ),
    }
    write_json(OUTPUT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))

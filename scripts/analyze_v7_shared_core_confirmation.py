#!/usr/bin/env python3
"""Build the compact manuscript table source for shared-core confirmation v3."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/v7/reviewer_revision/shared_resource_core_confirmation_v3"
NO_FAIR = ROOT / "reports/v7/reviewer_revision/shared_core_fairness_control_v1"
OUTPUT = BASE / "analysis_detail.json"


def _bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
    replicates: int = 20_000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(numerator), size=(replicates, len(numerator)))
    sampled = numerator[index].mean(axis=1) / denominator[index].mean(axis=1)
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def _method(row: dict, name: str) -> dict:
    return next(item for item in row["method_rows"] if item["method"] == name)


def _means(rows: list[dict], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def execute() -> dict:
    result_paths = sorted(BASE.glob("jobs/*/*/result.json"))
    if len(result_paths) != 24:
        raise ValueError(f"expected 24 completed jobs, found {len(result_paths)}")
    if not (BASE / "completion.json").is_file():
        raise ValueError("shared-core confirmation has no completion marker")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    indexed = {(int(row["seed"]), row["cell"]): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    if any((seed, cell) not in indexed for seed in seeds for cell in ("severe_clean", "severe_hotspot")):
        raise ValueError("shared-core result matrix is incomplete")

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell"]].append(row)
    cell_summary = {}
    for cell, subset in sorted(by_cell.items()):
        names = sorted({item["method"] for row in subset for item in row["method_rows"]})
        cell_summary[cell] = {
            "methods": {
                name: {
                    metric: float(np.mean([_method(row, name)[metric] for row in subset]))
                    for metric in (
                        "rmse",
                        "live_rmse",
                        "event_recall",
                        "worst_group_rmse",
                        "maximum_correction",
                    )
                }
                for name in names
            },
            "resource_means": {
                key: _means(subset, ("transport", key))
                for key in (
                    "raw_generated",
                    "raw_timely_delivered",
                    "release_delivered",
                    "receipt_issued",
                    "receipt_returned",
                    "audit_verified_return_pairs",
                    "audit_durable_storage_bytes",
                    "audit_inclusion_proof_bytes",
                    "total_wire_bytes",
                )
            },
            "selected_count_mean": _means(subset, ("selected_count",)),
            "fairness_means": {
                key: _means(subset, ("fairness", key))
                for key in (
                    "coverage_gap",
                    "globally_feasible_epochs",
                    "mean_max_avoidable_deficit",
                    "mean_max_unavoidable_deficit",
                )
            },
        }

    paired = []
    for seed in seeds:
        clean = indexed[seed, "severe_clean"]
        attack = indexed[seed, "severe_hotspot"]
        clean_ap = _method(clean, "AP_SHARED_CORE")
        clean_sq = _method(clean, "SQ")
        clean_huber = _method(clean, "HUBER")
        attack_ap = _method(attack, "AP_SHARED_CORE")
        attack_sq = _method(attack, "SQ")
        attack_huber = _method(attack, "HUBER")
        paired.append(
            {
                "seed": seed,
                "clean_ap": clean_ap["rmse"],
                "clean_sq": clean_sq["rmse"],
                "clean_huber": clean_huber["rmse"],
                "attack_ap": attack_ap["rmse"],
                "attack_sq": attack_sq["rmse"],
                "attack_huber": attack_huber["rmse"],
                "growth_ap": attack_ap["rmse"] - clean_ap["rmse"],
                "growth_sq": attack_sq["rmse"] - clean_sq["rmse"],
                "growth_huber": attack_huber["rmse"] - clean_huber["rmse"],
                "ap_cap_active_clean": clean_ap["maximum_correction"] >= 8.0 - 1e-8,
                "ap_cap_active_attack": attack_ap["maximum_correction"] >= 8.0 - 1e-8,
                "trace_match": clean["trace_hash"] == attack["trace_hash"],
            }
        )
    arrays = {
        key: np.asarray([row[key] for row in paired], dtype=float)
        for key in (
            "clean_ap",
            "clean_sq",
            "clean_huber",
            "attack_ap",
            "attack_sq",
            "attack_huber",
            "growth_ap",
            "growth_sq",
            "growth_huber",
        )
    }
    component_attribution = {
        "all_trace_pairs_match": all(row["trace_match"] for row in paired),
        "clean_ap_sq_ratio": float(arrays["clean_ap"].mean() / arrays["clean_sq"].mean()),
        "clean_ap_huber_ratio": float(
            arrays["clean_ap"].mean() / arrays["clean_huber"].mean()
        ),
        "hotspot_growth_means": {
            name: float(arrays[f"growth_{name}"].mean()) for name in ("ap", "sq", "huber")
        },
        "hotspot_attenuation_vs_sq": float(
            1.0 - arrays["growth_ap"].mean() / arrays["growth_sq"].mean()
        ),
        "huber_attenuation_vs_sq": float(
            1.0 - arrays["growth_huber"].mean() / arrays["growth_sq"].mean()
        ),
        "cap_active_worlds": {
            "clean": sum(row["ap_cap_active_clean"] for row in paired),
            "hotspot": sum(row["ap_cap_active_attack"] for row in paired),
        },
        "paired_ratio_ci95": {
            "clean_ap_sq": _bootstrap_ratio(
                arrays["clean_ap"], arrays["clean_sq"], seed=6256101
            ),
            "clean_ap_huber": _bootstrap_ratio(
                arrays["clean_ap"], arrays["clean_huber"], seed=6256102
            ),
        },
        "interpretation": (
            "HUBER isolates the bounded-score loss; divergence of AP_SHARED_CORE from HUBER "
            "isolates the final cap on the same observations and graph."
        ),
    }

    clean_rows = by_cell["severe_clean"]
    residual = np.asarray(
        [row["privacy_release"]["residual"]["rmse"] for row in clean_rows], dtype=float
    )
    raw = np.asarray(
        [row["privacy_release"]["raw_nonnegative"]["rmse"] for row in clean_rows],
        dtype=float,
    )
    public = np.asarray(
        [
            row["privacy_release"]["residual"]["same_support_public_rmse"]
            for row in clean_rows
        ],
        dtype=float,
    )
    privacy = {
        "worlds": len(clean_rows),
        "means": {
            "residual_release_rmse": float(residual.mean()),
            "nonnegative_raw_release_rmse": float(raw.mean()),
            "same_support_public_rmse": float(public.mean()),
        },
        "residual_vs_nonnegative_raw": {
            "ratio_of_means": float(residual.mean() / raw.mean()),
            "relative_reduction": float(1.0 - residual.mean() / raw.mean()),
            "ratio_bootstrap_ci95": _bootstrap_ratio(residual, raw, seed=6256103),
        },
        "residual_vs_public": {
            "ratio_of_means": float(residual.mean() / public.mean()),
            "ratio_bootstrap_ci95": _bootstrap_ratio(residual, public, seed=6256104),
        },
    }

    no_fair = None
    no_fair_path = NO_FAIR / "analysis.json"
    if no_fair_path.is_file():
        no_fair = json.loads(no_fair_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "role": "compact manuscript table source for registered shared-core confirmation v3",
        "source_result_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in result_paths
        },
        "registered_analysis": json.loads((BASE / "analysis.json").read_text(encoding="utf-8")),
        "cells": cell_summary,
        "component_attribution": component_attribution,
        "privacy_comparison": privacy,
        "matched_no_fairness": no_fair,
    }
    write_json(OUTPUT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))

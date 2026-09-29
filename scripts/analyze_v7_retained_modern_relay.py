#!/usr/bin/env python3
"""Expose the retained equal-resource AirProof-versus-PRoPHET comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/v4_validation/empirical_contacts_8050_8054/results.jsonl"
OUTPUT = ROOT / "reports/v7/reviewer_revision/retained_modern_relay_comparison"
FIELDS = (
    "deadline_delivery_ratio",
    "restricted_mean_delay_seconds",
    "group_delivery_gap",
    "transmissions_per_delivered",
)


def paired_interval(values, *, seed: int, replicates: int = 20_000):
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ValueError("at least two finite paired values required")
    rng = np.random.default_rng(seed)
    sampled = data[rng.integers(0, len(data), size=(replicates, len(data)))].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "conditional_role_traffic_replications": len(data),
    }


def analyze(rows):
    policies = {row["policy"] for row in rows}
    if not {"airproof_deadline", "prophet_gtmx"}.issubset(policies):
        raise ValueError("AirProof and PRoPHET rows required")
    cells = sorted({(row["dataset"], float(row["probability"])) for row in rows})
    output = {}
    direction = {"delivery": 0, "delay": 0, "gap": 0, "transmissions": 0}
    for cell_index, (dataset, probability) in enumerate(cells):
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset and float(row["probability"]) == probability
        ]
        indexed = {(int(row["seed"]), row["policy"]): row for row in selected}
        seeds = sorted({int(row["seed"]) for row in selected})
        if any((seed, policy) not in indexed for seed in seeds for policy in policies):
            raise ValueError("incomplete paired policy matrix")
        def vector(policy, field, indexed=indexed, seeds=seeds):
            return np.asarray(
                [indexed[seed, policy][field] for seed in seeds], dtype=float
            )

        contrasts = {
            "delivery_difference_percentage_points": 100
            * (
                vector("airproof_deadline", "deadline_delivery_ratio")
                - vector("prophet_gtmx", "deadline_delivery_ratio")
            ),
            "relative_delay_reduction": 1
            - vector("airproof_deadline", "restricted_mean_delay_seconds")
            / vector("prophet_gtmx", "restricted_mean_delay_seconds"),
            "group_gap_reduction_percentage_points": 100
            * (
                vector("prophet_gtmx", "group_delivery_gap")
                - vector("airproof_deadline", "group_delivery_gap")
            ),
            "relative_transmission_reduction": 1
            - vector("airproof_deadline", "transmissions_per_delivered")
            / vector("prophet_gtmx", "transmissions_per_delivered"),
        }
        for key, short in (
            ("delivery_difference_percentage_points", "delivery"),
            ("relative_delay_reduction", "delay"),
            ("group_gap_reduction_percentage_points", "gap"),
            ("relative_transmission_reduction", "transmissions"),
        ):
            direction[short] += int(float(np.mean(contrasts[key])) > 0)
        output[f"{dataset}:{probability:g}"] = {
            "seeds": seeds,
            "means": {
                policy: {
                    field: float(
                        np.mean([indexed[seed, policy][field] for seed in seeds])
                    )
                    for field in FIELDS
                }
                for policy in ("airproof_deadline", "prophet_gtmx")
            },
            "paired_airproof_minus_or_improvement_over_prophet": {
                key: paired_interval(value, seed=6254000 + 10 * cell_index + offset)
                for offset, (key, value) in enumerate(contrasts.items())
            },
        }
    return {
        "schema_version": 1,
        "comparison": "AirProof deadline relay versus RFC-6693-equation PRoPHET GTMX under identical retained resources and streams",
        "cells": output,
        "favorable_point_direction_cells_of_six": direction,
        "experimental_unit": "three measured contact collections; five synthetic role/traffic replications conditionally paired per collection/load",
        "inference_scope": "descriptive conditional bootstrap; the five seeds do not create independent measured networks",
    }


def main():
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    result = analyze(rows)
    result["source"] = SOURCE.relative_to(ROOT).as_posix()
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["favorable_point_direction_cells_of_six"], indent=2))


if __name__ == "__main__":
    main()

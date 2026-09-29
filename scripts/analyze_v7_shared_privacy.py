#!/usr/bin/env python3
"""Paired privacy-control contrasts for the registered shared-resource v2 run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from airproof.v6_covariance_forcing_inputs import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/v7/reviewer_revision/shared_resource_confirmation_v2/paired_worlds.jsonl"
OUTPUT = ROOT / "reports/v7/reviewer_revision/shared_resource_confirmation_v2/privacy_comparison.json"


def _bootstrap_ratio(left: np.ndarray, right: np.ndarray, *, seed: int, replicates: int):
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(left), size=(replicates, len(left)))
    ratio = left[index].mean(axis=1) / right[index].mean(axis=1)
    return np.quantile(ratio, [0.025, 0.975]).tolist()


def execute() -> dict:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    residual = np.asarray([row["release_rmse"] for row in rows], dtype=float)
    raw = np.asarray([row["release_nonnegative_raw_rmse"] for row in rows], dtype=float)
    public = np.asarray([row["release_public_rmse"] for row in rows], dtype=float)
    residual_raw_ratio = float(residual.mean() / raw.mean())
    residual_public_ratio = float(residual.mean() / public.mean())
    result = {
        "schema_version": 1,
        "role": "prespecified nonnegative raw control and matched-support public attribution",
        "worlds": len(rows),
        "source_sha256": sha256_file(SOURCE),
        "means": {
            "residual_release_rmse": float(residual.mean()),
            "nonnegative_raw_release_rmse": float(raw.mean()),
            "same_support_public_rmse": float(public.mean()),
        },
        "residual_vs_nonnegative_raw": {
            "ratio_of_means": residual_raw_ratio,
            "relative_reduction": 1.0 - residual_raw_ratio,
            "ratio_bootstrap_ci95": _bootstrap_ratio(
                residual, raw, seed=6254099, replicates=20_000
            ),
        },
        "residual_vs_public": {
            "ratio_of_means": residual_public_ratio,
            "relative_increase": residual_public_ratio - 1.0,
            "ratio_bootstrap_ci95": _bootstrap_ratio(
                residual, public, seed=6254100, replicates=20_000
            ),
        },
        "scope": (
            "Descriptive paired-world comparison under identical scheduled release support; "
            "the public control contains no protected citizen value."
        ),
    }
    write_json(OUTPUT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))

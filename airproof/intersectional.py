"""Outcome audit of overlapping spatial/availability strata, not new hard floors."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np

from .records import Observation
from .simulator import SyntheticWorld


def intersection_masks(
    world: SyntheticWorld, public_fields: np.ndarray, *, calibration_epochs: int,
    fixed_lag: int,
) -> tuple[dict[str, np.ndarray], dict]:
    fields = np.asarray(public_fields, dtype=float)
    steps, cells = world.truth.shape
    side = int(np.sqrt(cells))
    if side * side != cells or fields.shape != (steps, cells) or not 1 <= calibration_epochs < steps:
        raise ValueError("complete public grid and a completed calibration prefix required")
    if not np.isfinite(fields[:calibration_epochs]).all():
        raise ValueError("non-finite public calibration fields")
    coordinates = np.indices((side, side)).reshape(2, -1)
    edge_width = max(1, side // 8)
    peripheral = np.minimum.reduce((coordinates[0], coordinates[1],
                                    side - 1 - coordinates[0], side - 1 - coordinates[1])) < edge_width
    exposure = fields[:calibration_epochs].mean(axis=0)
    ingress = np.zeros(cells, dtype=int)
    seen = set()
    for record in world.observations:
        arrival = record.relay_arrival
        if (arrival is not None and record.epoch <= arrival < calibration_epochs
                and arrival - record.epoch <= fixed_lag and record.nullifier not in seen):
            seen.add(record.nullifier)
            ingress[record.cell] += 1
    quarter = int(np.ceil(cells / 4))
    high_exposure, low_ingress = np.zeros(cells, dtype=bool), np.zeros(cells, dtype=bool)
    high_exposure[np.lexsort((np.arange(cells), exposure))[-quarter:]] = True
    low_ingress[np.lexsort((np.arange(cells), ingress))[:quarter]] = True
    attributes = {"peripheral": peripheral, "high_public_exposure": high_exposure,
                  "low_prefix_ingress": low_ingress}
    names = tuple(attributes)
    masks = {}
    for count in range(1, len(names) + 1):
        for subset in itertools.combinations(names, count):
            masks[" & ".join(subset)] = np.logical_and.reduce([attributes[name] for name in subset])
    for group in np.unique(world.cell_groups):
        for bits in itertools.product((0, 1), repeat=len(names)):
            mask = world.cell_groups == group
            for name, flag in zip(names, bits, strict=True):
                mask = mask & (attributes[name] if flag else ~attributes[name])
            label = f"zone={group}|" + "|".join(f"{name}={flag}" for name, flag in zip(names, bits, strict=True))
            masks[label] = mask
    digest = hashlib.sha256()
    for name, mask in masks.items():
        digest.update(name.encode())
        digest.update(np.packbits(mask).tobytes())
    return masks, {
        "mask_sha256": digest.hexdigest(), "calibration_epochs": calibration_epochs,
        "attributes": {
            "peripheral": f"outer {edge_width} grid-cell band, fixed by geometry",
            "high_public_exposure": "top quarter of public-field prefix means, ties by cell id",
            "low_prefix_ingress": "bottom quarter of unique physically arrived-within-lag prefix records, ties by cell id",
        },
        "label_scope": "synthetic spatial/availability strata, not inferred income or demographic attributes",
        "privacy_scope": "internal experimental outcome audit; ingress metadata and these metrics are not part of the protected DP release interface",
        "selection_independence": "masks use geometry and completed raw/public prefix, never allocator output, scoring-period targets or future ingress",
        "hard_floor_scope": "allocation-zone partition only; no hard guarantee is asserted for overlapping audit strata",
    }


def audit_intersections(
    world: SyntheticWorld, reconstruction: np.ndarray, live_prediction: np.ndarray,
    accepted: list[Observation], public_fields: np.ndarray, *, burn_in: int, fixed_lag: int,
    minimum_cells: int = 8,
) -> dict:
    masks, metadata = intersection_masks(world, public_fields, calibration_epochs=burn_in,
                                          fixed_lag=fixed_lag)
    if minimum_cells < 1 or reconstruction.shape != world.truth.shape or live_prediction.shape != world.truth.shape:
        raise ValueError("invalid audit predictions/support")
    error = (reconstruction[burn_in:] - world.truth[burn_in:])**2
    live_error = (live_prediction[burn_in:] - world.truth[burn_in:])**2
    scored = [record for record in accepted if record.epoch >= burn_in]
    covered = np.zeros(error.shape, dtype=bool)
    for record in scored:
        covered[record.epoch - burn_in, record.cell] = True
    rows = {}
    for name, mask in masks.items():
        size = int(mask.sum())
        members = [record for record in scored if mask[record.cell]] if size else []
        rows[name] = {
            "cells": size, "supported_for_worst_stratum": size >= minimum_cells,
            "reconstruction_rmse": float(np.sqrt(error[:, mask].mean())) if size else None,
            "live_rmse": float(np.sqrt(live_error[:, mask].mean())) if size else None,
            "accepted_unique_contributors": len({record.user_id for record in members}),
            "accepted_records_per_100_cell_epochs": 100 * len(members) / (size * len(error)) if size else None,
            "citizen_covered_cell_epoch_fraction": float(covered[:, mask].mean()) if size else None,
        }
    supported = [row for row in rows.values() if row["supported_for_worst_stratum"]]
    return {"definition": metadata, "minimum_cells_for_worst_stratum": minimum_cells,
            "strata": rows, "supported_strata": len(supported),
            "worst_supported_reconstruction_rmse": max((row["reconstruction_rmse"] for row in supported), default=None),
            "worst_supported_live_rmse": max((row["live_rmse"] for row in supported), default=None)}

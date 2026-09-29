"""Integrity helpers for the uncertainty-v3 PUBLIC prerequisite generator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = ("live", "reconstructed", "truth", "public", "cell_groups", "scales")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def artifact_identity(*, seed: int, cell: str, source_digest: str,
                      configuration_hash: str, array_digest: str) -> str:
    return canonical_hash({"schema": 1, "role": "uncertainty-v3-public-prerequisite",
                           "seed": int(seed), "cell": str(cell),
                           "source_digest": source_digest,
                           "configuration_hash": configuration_hash,
                           "array_digest": array_digest})


def array_digest(arrays) -> str:
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        value = np.asarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def load_public_arrays(path: Path):
    with np.load(path, allow_pickle=False) as data:
        missing = set(ARRAY_KEYS) - set(data.files)
        if missing:
            raise ValueError(f"{path} misses {sorted(missing)}")
        return {key: data[key] for key in ARRAY_KEYS}


def prove_historical_cell_invariance(root: Path, seeds, cells):
    """Exact array proof on the retained 6201000-series artifacts."""
    input_hashes, seed_digests = {}, {}
    for seed in seeds:
        reference = None
        for cell in cells:
            path = root / f"{seed}_{cell}" / "PUBLIC.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            arrays = load_public_arrays(path)
            digest = array_digest(arrays)
            input_hashes[path.as_posix()] = sha256(path)
            if reference is None:
                reference = digest
            elif digest != reference:
                raise ValueError(f"PUBLIC arrays differ by physical cell for seed {seed}")
        seed_digests[str(seed)] = reference
    return {"seeds": list(map(int, seeds)), "cells": list(cells),
            "array_keys": list(ARRAY_KEYS), "exact_equal": True,
            "seed_array_digests": seed_digests, "input_hashes": input_hashes}


def target_inventory(root: Path, seeds, cells):
    paths = [root / f"{seed}_{cell}" / "PUBLIC.npz" for seed in seeds for cell in cells]
    present = [path for path in paths if path.exists()]
    return paths, present


def require_unambiguous_publish_state(expected, present, staging: Path):
    if present and len(present) != len(expected):
        raise RuntimeError(f"ambiguous partial final prerequisite set: {len(present)}/{len(expected)}")
    if staging.exists():
        raise RuntimeError(f"staging path already exists and will not be overwritten: {staging}")
    return "complete" if present else "empty"

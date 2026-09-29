from __future__ import annotations

import hashlib

import numpy as np

STREAMS = (
    "field",
    "mobility",
    "participation",
    "contact",
    "outage",
    "measurement",
    "reference",
    "attacker",
    "privacy",
    "public_meteorology",
)


def stream_seed(master_seed: int, name: str) -> int:
    if name not in STREAMS:
        raise KeyError(f"Unknown RNG stream {name!r}; expected one of {STREAMS}")
    digest = hashlib.sha256(f"airproof:{master_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def rng_streams(master_seed: int) -> dict[str, np.random.Generator]:
    return {name: np.random.default_rng(stream_seed(master_seed, name)) for name in STREAMS}


def seed_map(master_seed: int) -> dict[str, int]:
    return {name: stream_seed(master_seed, name) for name in STREAMS}

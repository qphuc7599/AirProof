"""Fail-closed provenance guards for the v6 covariance/forcing development family."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import re

import numpy as np


FORBIDDEN_PREDICTION_TOKENS = frozenset({
    "truth", "target", "targets", "attack", "attack_flag", "attack_label",
    "event_mask", "future", "future_public", "future_observation", "future_contact",
})


def _tokens(name: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    pieces = set(normalized.split("_"))
    pieces.add(normalized)
    return pieces


def assert_prediction_schema(fields: Iterable[str]) -> tuple[str, ...]:
    """Reject target, attack, event, or future semantics at the estimator boundary."""
    names = tuple(str(field) for field in fields)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("prediction input fields must be unique nonempty names")
    bad = sorted(name for name in names if _tokens(name) & FORBIDDEN_PREDICTION_TOKENS)
    if bad:
        raise ValueError(f"forbidden prediction inputs: {bad}")
    return names


def assert_causal_availability(
    prediction_epochs: np.ndarray,
    latest_input_epochs: np.ndarray,
) -> None:
    """Require every producer input to exist no later than its prediction epoch."""
    prediction, latest = np.broadcast_arrays(
        np.asarray(prediction_epochs, dtype=float),
        np.asarray(latest_input_epochs, dtype=float),
    )
    if not np.isfinite(prediction).all() or not np.isfinite(latest).all():
        raise ValueError("finite availability epochs required")
    if np.any(latest > prediction):
        raise ValueError("future producer input crosses the prediction boundary")


def assert_producer_manifest(manifest: Mapping) -> None:
    """Validate the minimum causal manifest without inferring missing provenance."""
    required = {
        "producer_id", "output_fields", "prediction_epochs", "latest_input_epochs",
        "source_hashes", "uses_hidden_truth", "uses_attack_flags",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"producer manifest missing fields: {missing}")
    assert_prediction_schema(manifest["output_fields"])
    if manifest["uses_hidden_truth"] is not False:
        raise ValueError("hidden truth is forbidden at prediction time")
    if manifest["uses_attack_flags"] is not False:
        raise ValueError("attack flags are forbidden at prediction time")
    if not manifest["producer_id"] or not manifest["source_hashes"]:
        raise ValueError("producer identity and source hashes are required")
    assert_causal_availability(manifest["prediction_epochs"], manifest["latest_input_epochs"])


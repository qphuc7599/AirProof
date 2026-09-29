"""Frozen clock-specific intervals and held-out evaluation; no online test fitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrozenIntervals:
    clock: str
    split_id: str
    training_end: float
    pooled_radii: tuple[float, float]
    strata: tuple[tuple[str, tuple[float, float]], ...]
    count: int
    stratum_counts: tuple[tuple[str, int], ...]
    fallback_strata: tuple[str, ...]
    selection_id: str
    selection_end: float
    evaluation_split_id: str
    evaluation_start: float
    levels: tuple[float, float] = (.9, .95)

    def predict(self, predictions, *, epochs, evaluation_split_id, public_strata=None):
        pred = np.asarray(predictions, float)
        times = np.broadcast_to(np.asarray(epochs, float), pred.shape)
        if (evaluation_split_id != self.evaluation_split_id
                or not np.isfinite(pred).all() or (pred < 0).any()
                or not np.isfinite(times).all() or (times < self.evaluation_start).any()):
            raise ValueError("bound evaluation split and finite predictions after calibration required")
        radii = np.broadcast_to(self.pooled_radii, (*pred.shape, 2)).copy()
        if public_strata is not None:
            labels = np.broadcast_to(np.asarray(public_strata).astype(str), pred.shape)
            for label, values in self.strata:
                radii[labels == label] = values
        return np.maximum(pred[..., None]-radii, 0), pred[..., None]+radii


def fit_frozen_intervals(predictions, targets, *, clock, split_id, training_end,
                         target_available_at, role="calibration", public_strata=None,
                         minimum_stratum_size=100, selection_id, selection_end,
                         evaluation_split_id, evaluation_start):
    """Independent post-selection calibration only; callers authenticate split provenance."""
    pred, truth = np.asarray(predictions, float), np.asarray(targets, float)
    availability = np.broadcast_to(np.asarray(target_available_at, float), pred.shape)
    if (pred.shape != truth.shape or pred.size < 20 or clock not in ("live", "reconstructed")
            or not split_id or role != "calibration" or not isinstance(minimum_stratum_size, int)
            or minimum_stratum_size < 100
            or not np.isfinite(training_end) or not np.isfinite(availability).all()
            or not selection_id or not evaluation_split_id
            or len({selection_id,split_id,evaluation_split_id}) != 3
            or not np.isfinite(selection_end) or not np.isfinite(evaluation_start)
            or not selection_end < training_end < evaluation_start
            or (availability <= selection_end).any()
            or (availability > training_end).any()
            or not np.isfinite(pred).all() or not np.isfinite(truth).all()):
        raise ValueError("valid independent calibration split and available targets required")
    errors = abs(pred-truth)

    def radii(values):
        ordered = np.sort(values.ravel())
        return tuple(float(ordered[min(int(np.ceil((len(ordered)+1)*q))-1, len(ordered)-1)])
                     for q in (.9, .95))

    groups,counts,fallback = [],[],[]
    if public_strata is not None:
        labels = np.broadcast_to(np.asarray(public_strata).astype(str), pred.shape)
        for label in np.unique(labels):
            selected = errors[labels == label]
            counts.append((str(label),int(selected.size)))
            if selected.size >= minimum_stratum_size:
                groups.append((str(label), radii(selected)))
            else:
                fallback.append(str(label))
    return FrozenIntervals(clock, split_id, float(training_end), radii(errors), tuple(groups), pred.size,
                           tuple(counts),tuple(fallback),selection_id,float(selection_end),
                           evaluation_split_id,float(evaluation_start))


def declared_uncertainty_strata(groups, ages, reference_contexts):
    """Bind calibration labels to declared public group, age and context only."""
    g=np.asarray(groups);a=np.asarray(ages);c=np.asarray(reference_contexts).astype(str)
    try:g,a,c=np.broadcast_arrays(g,a,c)
    except ValueError as exc:raise ValueError("broadcast-compatible declared strata required") from exc
    if (not np.issubdtype(g.dtype,np.integer) or (g<0).any()
            or not np.issubdtype(a.dtype,np.integer) or (a<0).any()
            or np.any(np.char.str_len(c)==0)):
        raise ValueError("nonnegative group/age and nonempty public context required")
    labels=np.empty(g.shape,dtype=object)
    for index in np.ndindex(g.shape):
        labels[index]=f"group={int(g[index])}|age={int(a[index])}|context={c[index]}"
    return labels.astype(str)


def evaluate_intervals(targets, lower, upper, *, public_strata=None, event_mask=None,
                        block_length=24, bootstrap_replicates=1000, seed=0):
    """Moving-block bootstrap samples whole time rows, preserving spatial co-movement.

    Outputs empirical percentile CIs, not exchangeability/conditional coverage claims.
    Empty event/stratum support remains unavailable, never a successful zero.
    """
    truth, low, high = (np.asarray(x, float) for x in (targets, lower, upper))
    if (truth.ndim != 2 or min(truth.shape) < 1 or low.shape != (*truth.shape, 2)
            or high.shape != low.shape or not all(np.isfinite(x).all() for x in (truth, low, high))
            or (low > high).any() or not isinstance(block_length, int)
            or not 1 <= block_length <= len(truth) or not isinstance(bootstrap_replicates, int)
            or bootstrap_replicates < 1):
        raise ValueError("valid time-by-cell intervals and bootstrap settings required")
    levels = np.array([.9, .95])
    covered = (low <= truth[..., None]) & (truth[..., None] <= high)
    width = high-low
    score = width + 2/(1-levels)*(np.maximum(low-truth[..., None], 0)
                                + np.maximum(truth[..., None]-high, 0))
    masks = {"all": np.ones(truth.shape, bool)}
    if public_strata is not None:
        labels = np.broadcast_to(np.asarray(public_strata).astype(str), truth.shape)
        masks.update({"stratum:"+str(label): labels == label for label in np.unique(labels)})
    if event_mask is not None:
        masks["event"] = np.broadcast_to(np.asarray(event_mask, bool), truth.shape)
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(truth)/block_length))
    samples = [np.concatenate([np.arange(s, s+block_length) for s in
               rng.integers(0, len(truth)-block_length+1, size=blocks)])[:len(truth)]
               for _ in range(bootstrap_replicates)]
    output = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if not count:
            output[name] = {"n": 0, "status": "unavailable"}
            continue
        metrics = {}
        for label, array in (("coverage", covered), ("width", width), ("interval_score", score)):
            means = [array[indices][mask[indices]].mean(axis=0)
                     for indices in samples if mask[indices].any()]
            metrics[label] = {"estimate": array[mask].mean(axis=0).tolist(),
                              "ci95": (np.quantile(means, [.025, .975], axis=0).T.tolist()
                                       if means else None),
                              "bootstrap_valid": len(means)}
        output[name] = {"n": count, "status": "available", **metrics}
    return {"levels": levels.tolist(), "block_length": block_length,
            "bootstrap_replicates": bootstrap_replicates, "seed": seed, "groups": output,
            "inference_scope":"descriptive moving-block bootstrap under temporal dependence",
            "coverage_claim":"empirical marginal coverage only; no exact conformal or conditional guarantee"}


def evaluate_estimator_intervals(result, calibrators, targets, *, epochs,
                                  evaluation_split_id, public_strata=None, event_mask=None, **bootstrap_kwargs):
    output = {}
    for clock in ("live", "reconstructed"):
        calibration = calibrators[clock]
        if calibration.clock != clock:
            raise ValueError("interval calibration clock mismatch")
        lower, upper = calibration.predict(getattr(result, clock), epochs=epochs,
                                           evaluation_split_id=evaluation_split_id,
                                           public_strata=public_strata)
        output[clock] = evaluate_intervals(targets, lower, upper, public_strata=public_strata,
                                           event_mask=event_mask, **bootstrap_kwargs)
    return output

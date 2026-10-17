from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .records import Observation


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray, cell_groups: np.ndarray) -> dict[str, float]:
    residual = prediction - truth
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan")
    group_rmse = {
        int(group): float(np.sqrt(np.mean(residual[:, cell_groups == group] ** 2)))
        for group in np.unique(cell_groups)
    }
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "worst_group_rmse": max(group_rmse.values()),
        "worst_median_error_ratio": max(group_rmse.values()) / max(np.median(list(group_rmse.values())), 1e-12),
        **{f"group_{group}_rmse": value for group, value in group_rmse.items()},
    }


def network_metrics(
    selected: Iterable[Observation],
    accepted: Iterable[Observation],
    arrivals: dict[str, int],
    horizon: int,
) -> dict[str, float | bool]:
    selected = list(selected)
    accepted = list(accepted)
    delivered = {item.nullifier for item in accepted}
    ages: list[float] = []
    for item in selected:
        if item.nullifier in delivered:
            ages.append(float(arrivals[item.nullifier] - item.epoch))
        else:
            ages.append(float(max(0, horizon - item.epoch)))
    ratio = len(accepted) / len(selected) if selected else 0.0
    aoi95 = float(np.percentile(ages, 95)) if ages else float(horizon)
    return {
        "selected_records": len(selected),
        "delivered_records": len(accepted),
        "delivery_ratio": ratio,
        "aoi_median": float(np.median(ages)) if ages else float(horizon),
        "aoi_p95": aoi95,
        "aoi_p95_gt_horizon": ratio < 0.95,
        "restricted_mean_aoi": float(np.mean(ages)) if ages else float(horizon),
        "queue_drop_records": len(selected) - len(accepted),
    }


def coverage_metrics(selected: Iterable[Observation], groups: int) -> dict[str, float]:
    users: dict[int, set[int]] = defaultdict(set)
    infeasible = 0
    for item in selected:
        users[item.group].add(item.user_id)
    counts = np.array([len(users[group]) for group in range(groups)], dtype=float)
    maximum = max(float(np.max(counts)), 1.0)
    normalized = counts / maximum
    if np.any(counts == 0):
        infeasible = int(np.sum(counts == 0))
    return {
        "coverage_gap": float(np.max(normalized) - np.min(normalized)),
        "coverage_p10": float(np.percentile(normalized, 10)),
        "infeasible_group_rate": infeasible / groups,
        **{f"group_{group}_contributors": int(counts[group]) for group in range(groups)},
    }


from __future__ import annotations

import numpy as np

from scripts.analyze_v7_lifetime_retained import _aggregate_profiles, _paired_summary


def test_profile_aggregation_ignores_missing_epoch_denominators():
    rows = [
        {
            "epoch": 0,
            "absolute_epoch": 48,
            "input_records": 0,
            "effective_records": 0,
            "dropped_exhausted_records": 0,
            "active_users": 0,
            "granted_active_users": 0,
            "active_user_fraction": np.nan,
            "mean_effective_quality": np.nan,
            "users_seen": 0,
            "mean_remaining_budget": np.nan,
            "median_remaining_budget": np.nan,
            "exhausted_user_fraction": np.nan,
            "cumulative_input_records": 0,
            "cumulative_effective_records": 0,
            "cumulative_dropped_exhausted_records": 0,
            "cumulative_requested_exposure": 0.0,
            "cumulative_granted_exposure": 0.0,
        }
    ]
    result = _aggregate_profiles(rows)
    assert result[0]["mean_active_user_fraction"] is None
    assert result[0]["mean_cumulative_granted_exposure"] == 0.0


def test_paired_summary_preserves_world_pairing():
    rows = []
    values = {
        1: {"AP_LIFETIME7": 1.0, "PUBLIC": 2.0, "SQ": 0.9},
        2: {"AP_LIFETIME7": 3.0, "PUBLIC": 2.0, "SQ": 3.1},
    }
    for seed, methods in values.items():
        for method, rmse in methods.items():
            rows.append(
                {
                    "seed": seed,
                    "scenario": "clean",
                    "phase": "late",
                    "method": method,
                    "rmse": rmse,
                }
            )
    result = _paired_summary(rows)[0]
    assert result["ap_minus_public_mean"] == 0.0
    assert result["ap_better_than_public_worlds"] == 1
    assert result["ap_better_than_sq_worlds"] == 1

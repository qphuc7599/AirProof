"""Frozen contract and gate evaluation for history-normalized Huber v3."""
from __future__ import annotations

import math
from collections.abc import Mapping


def assert_registration(registration: Mapping) -> None:
    if (registration.get("schema_version") != 1
            or registration.get("registration")
            != "v6-history-normalized-huber-mechanistic-development-v3"
            or registration.get("status")
            != "prospective_protocol_frozen_after_v2_closure_before_v3_outcomes"):
        raise ValueError("v3 identity or frozen status changed")
    inputs = registration["inputs"]
    if (inputs["development_seeds"]
            != [6241020, 6241021, 6241022, 6241023,
                6241024, 6241025, 6241026, 6241027]
            or inputs["cells"] != ["severe_clean", "severe_drift", "severe_hotspot"]
            or len(set(inputs["development_seeds"])) != 8):
        raise ValueError("v3 development worlds or cells changed")
    candidates = registration["candidates"]
    if ([row["id"] for row in candidates]
            != ["history_budget1", "history_budget2", "history_budget4"]
            or [row["per_user_weight_budget"] for row in candidates] != [1.0, 2.0, 4.0]):
        raise ValueError("v3 must retain the previously registered {1,2,4} budgets")
    fixed = registration["fixed_estimator"]
    expected_fixed = {
        "innovation_covariance": "world_cross_fitted_covariance",
        "residual_forcing": "producer_exact_causal",
        "cap": 8.0,
        "output_cap": True,
        "input_clip": False,
        "loss": "huber",
        "huber_delta": 1.345,
        "lag": 6,
        "lambda_zero": 0.05,
        "lambda_temporal": 0.05,
        "lambda_spatial": 0.02,
        "time_step": 1.0,
        "tolerance": 1e-7,
        "max_iterations": 1000,
        "numerical_refinements": 2,
        "history_normalized_huber": True,
    }
    if dict(fixed) != expected_fixed:
        raise ValueError("fixed v3 estimator contract changed")
    controls = registration["controls"]
    if (controls["source_campaign"]
            != "reports/v6/innovation_covariance_forcing_development_v2/outcomes"
            or controls["source_candidate"]
            != "world_cross_fitted_covariance__producer_exact_causal"
            or controls["reused_methods"] != ["PUBLIC", "SQ", "HUBER", "CANDIDATE"]
            or controls["renamed_base_method"] != "BASE_BOUNDED_HUBER"):
        raise ValueError("v3 locked controls changed")
    gates = registration["joint_gates"]
    expected_gates = {
        "aggregation_unit": "paired exposed development world",
        "clean_candidate_to_SQ_mean_rmse_ratio_max": 1.05,
        "drift_mean_excess_rmse_attenuation_min": 0.20,
        "hotspot_mean_excess_rmse_attenuation_min": 0.20,
        "event_recall_mean_loss_vs_SQ_max_percentage_points": 5.0,
        "all_solver_failure_rates_max": 0.0,
        "all_projected_gradient_inf_max": 1e-6,
        "all_candidate_absolute_corrections_max": 8.0000001,
        "all_outputs_finite": True,
        "all_information_fingerprints_equal_within_comparison": True,
        "all_resource_invariants": True,
    }
    if dict(gates) != expected_gates:
        raise ValueError("original scientific margins changed")


def _mean(rows: list[Mapping], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"finite nonempty {field} required")
    return sum(values) / len(values)


def evaluate(registration: Mapping, rows: list[Mapping]) -> dict:
    assert_registration(registration)
    seeds = list(map(int, registration["inputs"]["development_seeds"]))
    cells = registration["inputs"]["cells"]
    candidates = registration["candidates"]
    methods = ("CANDIDATE", "PUBLIC", "SQ", "HUBER")
    by_key = {(int(row["seed"]), row["candidate"], row["scenario"], row["method"]): row
              for row in rows}
    expected = len(seeds) * len(candidates) * len(cells) * len(methods)
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError("incomplete or duplicate v3 comparison matrix")
    gates = registration["joint_gates"]
    decisions = []
    for candidate in candidates:
        name = candidate["id"]
        invariants = {}
        for seed in seeds:
            for cell in cells:
                group = [by_key[seed, name, cell, method] for method in methods]
                invariants[f"fingerprint:{seed}:{cell}"] = len({
                    row["information_fingerprint"] for row in group}) == 1
                for row in group:
                    prefix = f"{seed}:{cell}:{row['method']}"
                    invariants[f"finite:{prefix}"] = (row.get("finite_outputs") is True
                        and all(math.isfinite(float(row[field])) for field in
                                ("rmse", "event_recall", "wall_seconds",
                                 "process_cpu_seconds", "peak_rss_bytes")))
                    invariants[f"resource:{prefix}"] = row.get("resource_invariants_pass") is True
                    if row["method"] != "PUBLIC":
                        invariants[f"solver:{prefix}"] = (
                            float(row["solver_failure_rate"])
                            <= gates["all_solver_failure_rates_max"]
                            and float(row["projected_gradient_inf"])
                            <= gates["all_projected_gradient_inf_max"])
                    if row["method"] == "CANDIDATE":
                        invariants[f"cap:{prefix}"] = (
                            float(row["maximum_absolute_correction"])
                            <= gates["all_candidate_absolute_corrections_max"])
                        invariants[f"user_budget:{prefix}"] = (
                            float(row["maximum_user_window_weight"])
                            <= float(candidate["per_user_weight_budget"]) + 1e-7)
        clean_candidate = [by_key[seed, name, "severe_clean", "CANDIDATE"]
                           for seed in seeds]
        clean_sq = [by_key[seed, name, "severe_clean", "SQ"] for seed in seeds]
        clean_ratio = _mean(clean_candidate, "rmse") / _mean(clean_sq, "rmse")
        attenuation = {}
        event_loss = {}
        for cell in ("severe_drift", "severe_hotspot"):
            attack_candidate = [by_key[seed, name, cell, "CANDIDATE"] for seed in seeds]
            attack_sq = [by_key[seed, name, cell, "SQ"] for seed in seeds]
            candidate_excess = (_mean(attack_candidate, "rmse")
                                - _mean(clean_candidate, "rmse"))
            sq_excess = _mean(attack_sq, "rmse") - _mean(clean_sq, "rmse")
            attenuation[cell] = 1 - candidate_excess / sq_excess if sq_excess > 0 else -math.inf
            event_loss[cell] = 100 * (_mean(attack_sq, "event_recall")
                                      - _mean(attack_candidate, "event_recall"))
        scientific = {
            "clean": clean_ratio <= gates["clean_candidate_to_SQ_mean_rmse_ratio_max"],
            "drift": attenuation["severe_drift"]
            >= gates["drift_mean_excess_rmse_attenuation_min"],
            "hotspot": attenuation["severe_hotspot"]
            >= gates["hotspot_mean_excess_rmse_attenuation_min"],
            "drift_event": event_loss["severe_drift"]
            <= gates["event_recall_mean_loss_vs_SQ_max_percentage_points"],
            "hotspot_event": event_loss["severe_hotspot"]
            <= gates["event_recall_mean_loss_vs_SQ_max_percentage_points"],
        }
        normalized = max(
            clean_ratio / gates["clean_candidate_to_SQ_mean_rmse_ratio_max"],
            (1 - attenuation["severe_drift"])
            / (1 - gates["drift_mean_excess_rmse_attenuation_min"]),
            (1 - attenuation["severe_hotspot"])
            / (1 - gates["hotspot_mean_excess_rmse_attenuation_min"]),
            max(0.0, event_loss["severe_drift"])
            / gates["event_recall_mean_loss_vs_SQ_max_percentage_points"],
            max(0.0, event_loss["severe_hotspot"])
            / gates["event_recall_mean_loss_vs_SQ_max_percentage_points"],
        )
        passed = all(invariants.values()) and all(scientific.values())
        decisions.append({
            "candidate": name, "per_user_weight_budget": candidate["per_user_weight_budget"],
            "pass": passed, "scientific_checks": scientific,
            "all_invariants_pass": all(invariants.values()),
            "failed_invariants": sorted(key for key, value in invariants.items() if not value),
            "metrics": {"clean_ratio": clean_ratio, "attenuation": attenuation,
                        "event_loss_percentage_points": event_loss,
                        "worst_normalized_condition_loss": normalized},
            "mean_runtime_seconds": _mean([row for row in rows
                if row["candidate"] == name and row["method"] == "CANDIDATE"],
                "wall_seconds"),
        })
    feasible = [row for row in decisions if row["pass"]]
    selected = min(feasible, key=lambda row: (
        row["metrics"]["worst_normalized_condition_loss"],
        row["per_user_weight_budget"], row["mean_runtime_seconds"],
        row["candidate"]))["candidate"] if feasible else None
    return {
        "candidates": decisions,
        "eligible": [row["candidate"] for row in feasible],
        "selected_candidate": selected,
        "family_closed": selected is None,
        "extension_authorized": False,
        "scope": "sequential exposed mechanistic-synthetic development; no p-values",
    }

"""Frozen contract and gates for exposed lifetime-budget repair development v5."""
from __future__ import annotations

import math
from collections.abc import Mapping

REGISTRATION_ID = "v6-causal-lifetime-budget-repair-exposed-development-v5"
SEEDS = list(range(6241200, 6241212))
CELLS = ["anchor_clean", "outage_clean", "severe_clean", "severe_drift", "severe_hotspot"]
CANDIDATES = ["lifetime_budget7_repair", "lifetime_budget14_repair"]
CONTROL_METHODS = ["AP_LIFETIME28", "PUBLIC", "SQ", "HUBER"]


def assert_registration(registration: Mapping) -> None:
    expected_sections = {
        "schema_version", "registration", "status", "role", "objective",
        "failure_lineage", "input_reuse", "fixed_estimator", "candidates",
        "candidate_derivation", "controls", "development_gates",
        "selection_and_exit", "runtime", "artifacts", "protected_namespaces",
    }
    if set(registration) != expected_sections:
        raise ValueError("repair registration sections changed")
    if (registration.get("schema_version") != 1
            or registration.get("registration") != REGISTRATION_ID
            or registration.get("status") != (
                "prospective_candidate_protocol_frozen_after_validation_v1_failure_"
                "before_candidate_predictions")):
        raise ValueError("repair registration identity changed")
    reuse = registration["input_reuse"]
    if (reuse["seeds"] != SEEDS or reuse["cells"] != CELLS
            or reuse["clean_cells"] != CELLS[:3]
            or reuse["attack_pairs"] != {
                "severe_drift": "severe_clean", "severe_hotspot": "severe_clean"}
            or reuse["bundle"] != "reports/v6/causal_lifetime_exposure_validation_v1/inputs"
            or reuse["input_lock"] != "reports/v6/causal_lifetime_exposure_validation_v1/input_lock.json"
            or reuse["input_manifest"] != "reports/v6/causal_lifetime_exposure_validation_v1/inputs/manifest.json"
            or reuse["source_validation_outcomes"] != "reports/v6/causal_lifetime_exposure_validation_v1/outcomes"
            or reuse["new_world_or_transport_generation"] is not False
            or reuse["prediction_before_scoring"] is not True):
        raise ValueError("repair input reuse changed")
    expected_fixed = {
        "innovation_covariance": "world_cross_fitted_covariance",
        "innovation_calibration": "reports/v6/innovation_covariance_forcing_development_v2/inputs/global_calibration.json",
        "residual_forcing": "producer_exact_causal", "cap": 8.0,
        "output_cap": True, "input_clip": False, "loss": "huber",
        "huber_delta": 1.345, "lag": 6, "lambda_zero": 0.05,
        "lambda_temporal": 0.05, "lambda_spatial": 0.02,
        "time_step": 1.0, "tolerance": 1e-7, "max_iterations": 1000,
        "numerical_refinements": 2, "causal_lifetime_exposure": True,
        "maximum_record_uses": 7,
    }
    if dict(registration["fixed_estimator"]) != expected_fixed:
        raise ValueError("repair estimator changed")
    if registration["candidates"] != [
        {"id": CANDIDATES[0], "per_user_exposure_budget": 7.0},
        {"id": CANDIDATES[1], "per_user_exposure_budget": 14.0},
    ]:
        raise ValueError("repair candidates changed")
    controls = registration["controls"]
    if (controls["reused_methods"] != CONTROL_METHODS
            or controls["recompute"] is not False
            or controls["parent_failure_retained"] is not True):
        raise ValueError("repair controls changed")
    gates = registration["development_gates"]
    expected_gates = {
        "aggregation_unit": "paired exposed world",
        "clean_candidate_to_SQ_mean_rmse_ratio_max": 1.05,
        "attack_target_attenuation": 0.20,
        "attack_gate": (
            "For each attack, require mean SQ excess > 0 and mean[0.8*SQ excess "
            "- candidate excess] >= 0; this is algebraically the unchanged "
            "20-percent target without division."),
        "event_recall_mean_loss_vs_SQ_max_percentage_points": 5.0,
        "severe_clean_candidate_to_PUBLIC_mean_rmse_ratio_max": 1.0,
        "all_solver_failure_rates_max": 0.0,
        "all_projected_gradient_inf_max": 1e-6,
        "all_candidate_absolute_corrections_max": 8.0000001,
        "all_lifetime_exposures_at_most_candidate_budget": True,
        "all_outputs_finite": True,
        "all_information_fingerprints_equal": True,
        "all_resource_invariants": True,
    }
    if dict(gates) != expected_gates:
        raise ValueError("repair gates changed")
    if (registration["runtime"] != {
            "workers": 4, "blas_threads_per_worker": 1,
            "runtime_smoke_seeds": [SEEDS[0]],
            "resume": "only missing hash-identical predictions; no replacement seed"}
            or registration["protected_namespaces"] != [
                "data/external/epa", "reports/v6/primary",
                "reports/v6/confirmation", "reports/v6/test"]):
        raise ValueError("repair runtime or protected scope changed")


def _mean(rows: list[Mapping], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"finite nonempty {field} required")
    return sum(values) / len(values)


def evaluate(registration: Mapping, rows: list[Mapping]) -> dict:
    """Evaluate the complete exposed matrix with the registered direct IUT."""
    assert_registration(registration)
    by_key = {(int(row["seed"]), row["candidate"], row["scenario"], row["method"]): row
              for row in rows}
    methods = ["CANDIDATE", *CONTROL_METHODS]
    expected = len(SEEDS) * len(CELLS) * len(CANDIDATES) * len(methods)
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError("incomplete or duplicate repair comparison matrix")
    gates = registration["development_gates"]
    decisions = []
    for candidate_spec in registration["candidates"]:
        candidate, budget = candidate_spec["id"], float(candidate_spec["per_user_exposure_budget"])
        invariants = {}
        for seed in SEEDS:
            for cell in CELLS:
                group = [by_key[seed, candidate, cell, method] for method in methods]
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
                        invariants[f"budget:{prefix}"] = (
                            float(row["maximum_user_lifetime_exposure"]) <= budget + 1e-7)
        clean_ratios = {}
        for cell in registration["input_reuse"]["clean_cells"]:
            cand = [by_key[s, candidate, cell, "CANDIDATE"] for s in SEEDS]
            sq = [by_key[s, candidate, cell, "SQ"] for s in SEEDS]
            clean_ratios[cell] = _mean(cand, "rmse") / _mean(sq, "rmse")
        severe = [by_key[s, candidate, "severe_clean", "CANDIDATE"] for s in SEEDS]
        severe_public = [by_key[s, candidate, "severe_clean", "PUBLIC"] for s in SEEDS]
        citizen_ratio = _mean(severe, "rmse") / _mean(severe_public, "rmse")
        attacks, event_loss = {}, {}
        severe_sq = [by_key[s, candidate, "severe_clean", "SQ"] for s in SEEDS]
        for cell in registration["input_reuse"]["attack_pairs"]:
            attacked = [by_key[s, candidate, cell, "CANDIDATE"] for s in SEEDS]
            attacked_sq = [by_key[s, candidate, cell, "SQ"] for s in SEEDS]
            candidate_excess = _mean(attacked, "rmse") - _mean(severe, "rmse")
            sq_excess = _mean(attacked_sq, "rmse") - _mean(severe_sq, "rmse")
            margin = 0.8 * sq_excess - candidate_excess
            attacks[cell] = {
                "sq_excess_rmse": sq_excess,
                "candidate_excess_rmse": candidate_excess,
                "direct_margin": margin,
                "attenuation": 1 - candidate_excess / sq_excess if sq_excess > 0 else None,
                "pass": sq_excess > 0 and margin >= 0,
            }
            event_loss[cell] = 100 * (_mean(attacked_sq, "event_recall")
                                      - _mean(attacked, "event_recall"))
        scientific = {
            **{f"clean:{cell}": ratio <= gates[
                "clean_candidate_to_SQ_mean_rmse_ratio_max"]
               for cell, ratio in clean_ratios.items()},
            "attack:severe_drift": attacks["severe_drift"]["pass"],
            "attack:severe_hotspot": attacks["severe_hotspot"]["pass"],
            "event:severe_drift": event_loss["severe_drift"] <= gates[
                "event_recall_mean_loss_vs_SQ_max_percentage_points"],
            "event:severe_hotspot": event_loss["severe_hotspot"] <= gates[
                "event_recall_mean_loss_vs_SQ_max_percentage_points"],
            "citizen": citizen_ratio <= gates[
                "severe_clean_candidate_to_PUBLIC_mean_rmse_ratio_max"],
        }
        passed = all(invariants.values()) and all(scientific.values())
        decisions.append({
            "candidate": candidate, "per_user_exposure_budget": budget,
            "pass": passed, "scientific_checks": scientific,
            "all_invariants_pass": all(invariants.values()),
            "failed_invariants": sorted(k for k, value in invariants.items() if not value),
            "metrics": {"clean_ratios": clean_ratios, "attacks": attacks,
                        "event_loss_percentage_points": event_loss,
                        "citizen_ratio": citizen_ratio,
                        "minimum_attack_attenuation": min(
                            value["attenuation"] if value["attenuation"] is not None
                            else -math.inf for value in attacks.values())},
            "mean_runtime_seconds": _mean([row for row in rows
                if row["candidate"] == candidate and row["method"] == "CANDIDATE"],
                "wall_seconds"),
        })
    eligible = [row for row in decisions if row["pass"]]
    selected = sorted(eligible, key=lambda row: (
        -row["metrics"]["minimum_attack_attenuation"],
        row["metrics"]["clean_ratios"]["severe_clean"],
        row["per_user_exposure_budget"], row["mean_runtime_seconds"],
        row["candidate"]))[0]["candidate"] if eligible else None
    return {"candidates": decisions, "eligible": [row["candidate"] for row in eligible],
            "selected_candidate": selected, "family_closed": selected is None,
            "validation_v1_rescued": False, "extension_authorized": False,
            "scope": "sequential exposed development; descriptive point gates; no p-values"}

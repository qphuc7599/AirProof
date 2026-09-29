"""Frozen contract and simultaneous gate evaluation for validation v1."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

REGISTRATION_ID = "v6-causal-lifetime-exposure-independent-synthetic-validation-v1"
SEEDS = list(range(6241200, 6241212))
CELLS = ["anchor_clean", "outage_clean", "severe_clean", "severe_drift", "severe_hotspot"]
METHODS = ["AP_LIFETIME28", "PUBLIC", "SQ", "HUBER"]
BOOTSTRAP_SEED = 6241299
BOOTSTRAP_NAMESPACE = "lifetime_validation_bootstrap"

EXPECTED_NOMINATION = {
    "development_registration": "configs/v6/causal_lifetime_exposure_development_v4.json",
    "development_analysis": (
        "reports/v6/causal_lifetime_exposure_development_v4/outcomes/analysis.json"
    ),
    "selected_candidate": "lifetime_budget28",
    "selection_rule_applied_once": True,
    "no_validation_outcome_seen": True,
}
EXPECTED_HISTORICAL_REUSE = {
    "v4_primary": "reports/v4_primary/core_final_7600_7629_20260903/analysis.json",
    "v4_reuse_lock": "configs/v6/v4_reuse_hash_lock.json",
    "rerun": False,
    "scope": (
        "v4 H1-H7 remain historical; validation computes only the changed robust-twin "
        "arms on new worlds"
    ),
}
EXPECTED_CLOCK = {
    "scored_acquisition_half_open": [48, 672],
    "prediction_half_open": [48, 678],
    "rmse_attenuation_and_citizen_value": (
        "lag-6 reconstructed estimate scored on acquisition epochs [48,672)"
    ),
    "event_recall": "immutable live estimate at each acquisition epoch in [48,672)",
    "lag_tail_and_transport_drain": (
        "generated for causal reconstruction/resource completeness but unscored"
    ),
    "event_threshold_source": (
        "per-world truth prefix [0,48) for each clean base cell; severe attacks reuse "
        "the severe_clean threshold"
    ),
    "event_threshold_quantile": 0.95,
    "event_comparison": ">=",
    "event_support_required_positive": True,
}
EXPECTED_ATTACK_CONTRACT = {
    "severe_drift": {
        "kind": "adversarial_drift",
        "fraction": 0.2,
        "amplitude": 12.0,
        "start_epoch": 336,
    },
    "severe_hotspot": {
        "kind": "hotspot_suppression",
        "fraction": 0.2,
        "amplitude": 12.0,
        "start_epoch": 336,
    },
    "matched_fields": [
        "truth",
        "physical_transition",
        "physical_forcing",
        "public_center",
        "candidate_identity",
        "arrival_clock",
        "selection_order",
    ],
    "allowed_difference": (
        "observation value and corrupted flag only after start_epoch for the registered "
        "attacker set"
    ),
}


def assert_registration(registration: Mapping) -> None:
    expected_sections = {
        "schema_version", "registration", "status", "role", "objective", "nomination",
        "historical_reuse", "worlds", "evaluation_clock", "attack_contract",
        "fixed_estimator", "methods", "same_information", "input_producer",
        "executable_sources", "scientific_gates", "decision", "runtime", "artifacts",
        "protected_namespaces",
    }
    if set(registration) != expected_sections:
        raise ValueError("validation registration sections changed")
    if (registration.get("schema_version") != 1
            or registration.get("registration") != REGISTRATION_ID
            or registration.get("status")
            != "prospective_protocol_frozen_after_development_nomination_before_validation_input_generation"):
        raise ValueError("validation identity or prospective status changed")
    if registration.get("role") != (
        "independent synthetic mechanism validation; not primary, EPA test, or confirmation"
    ) or registration.get("objective") != (
        "Validate the single development-nominated causal lifetime-exposure estimator on "
        "new worlds and all five physical cells before any integrated campaign."
    ):
        raise ValueError("validation role changed")
    if dict(registration.get("nomination", {})) != EXPECTED_NOMINATION:
        raise ValueError("development nomination contract changed")
    if dict(registration.get("historical_reuse", {})) != EXPECTED_HISTORICAL_REUSE:
        raise ValueError("historical reuse contract changed")
    worlds = registration["worlds"]
    if (worlds["seeds"] != SEEDS or worlds["cells"] != CELLS
            or worlds["clean_cells"] != CELLS[:3]
            or worlds["attack_pairs"] != {
                "severe_drift": "severe_clean", "severe_hotspot": "severe_clean"}
            or worlds["seed_registry"] != "configs/v6/seed_registry.json"
            or worlds["seed_namespace"] != "lifetime_estimator_validation"
            or len(set(worlds["seeds"])) != 12
            or [worlds[key] for key in ("agents", "grid_side", "acquisition_epochs",
                                        "burn_in_epochs", "fixed_lag_epochs",
                                        "transport_drain_epochs")]
            != [1000, 32, 672, 48, 6, 24]):
        raise ValueError("validation world contract changed")
    if dict(registration.get("evaluation_clock", {})) != EXPECTED_CLOCK:
        raise ValueError("validation evaluation clock changed")
    if dict(registration.get("attack_contract", {})) != EXPECTED_ATTACK_CONTRACT:
        raise ValueError("validation attack matching changed")
    fixed = registration["fixed_estimator"]
    expected_fixed = {
        "innovation_covariance": "world_cross_fitted_covariance",
        "innovation_calibration": "reports/v6/innovation_covariance_forcing_development_v2/inputs/global_calibration.json",
        "residual_forcing": "producer_exact_causal",
        "cap": 8.0, "output_cap": True, "input_clip": False, "loss": "huber",
        "huber_delta": 1.345, "lag": 6, "lambda_zero": 0.05,
        "lambda_temporal": 0.05, "lambda_spatial": 0.02, "time_step": 1.0,
        "tolerance": 1e-7, "max_iterations": 1000,
        "numerical_refinements": 2, "causal_lifetime_exposure": True,
        "per_user_exposure_budget": 28.0, "maximum_record_uses": 7,
    }
    if dict(fixed) != expected_fixed or registration["methods"] != METHODS:
        raise ValueError("selected estimator or controls changed")
    if registration.get("same_information") != (
        "For every world/cell all methods hash the same raw public center, selected "
        "records, arrivals, transition, covariance and residual forcing before "
        "method-specific ignoring or weighting."
    ):
        raise ValueError("same-information control changed")
    expected_input_producer = {
        "bundle": "reports/v6/causal_lifetime_exposure_validation_v1/inputs",
        "source_lock": "reports/v6/causal_lifetime_exposure_validation_v1/source_lock.json",
        "input_lock": "reports/v6/causal_lifetime_exposure_validation_v1/input_lock.json",
        "readiness": "reports/v6/causal_lifetime_exposure_validation_v1/readiness.json",
        "truth_boundary": (
            "prediction and scoring capabilities are separate; every prediction is "
            "written and hashed before scoring is loaded"
        ),
    }
    if dict(registration.get("input_producer", {})) != expected_input_producer:
        raise ValueError("validation input or truth boundary changed")
    expected_sources = {
        "base_configuration": (
            "reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml"
        ),
        "protocol": "configs/v6/protocol.yaml",
        "expected_config_hash": {
            "anchor_clean": "b2b9328c85bc7b5a8e4f604be7208aae99bf1eb75ceb051dea4225455fc70380",
            "outage_clean": "26770ed4f29033755c4ffe12d9dc9d39524477b3565a592460ed3ea3568e482b",
            "severe_clean": "e3109978621347ee42467c0a863d159e8023b50b2676e9e25288bb08750599f0",
            "severe_drift": "6ab77d4ade6738e8cd0511f13374424268abc0203aa4ab6c8c67b29d2bdd35b1",
            "severe_hotspot": "ca7838b0c16f7336ea1e726b09ec326574ea1c9e184472ac47b827dc7cd9fc29",
        },
    }
    if dict(registration.get("executable_sources", {})) != expected_sources:
        raise ValueError("validation executable source contract changed")
    gates = registration["scientific_gates"]
    expected_gates = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": 20000,
        "confidence_level": 0.95,
        "simultaneous_family_size": 8,
        "simultaneous_method": "paired-world percentile bootstrap with one-sided Bonferroni alpha/8 bounds",
        "clean_AP_to_SQ_ratio_upper_max": 1.05,
        "drift_excess_RMSE_attenuation_lower_min": 0.20,
        "hotspot_excess_RMSE_attenuation_lower_min": 0.20,
        "event_recall_loss_vs_SQ_upper_max_percentage_points": 5.0,
        "severe_clean_AP_to_PUBLIC_ratio_upper_max": 1.0,
        "all_solver_failure_rates_max": 0.0,
        "all_projected_gradient_inf_max": 1e-6,
        "AP_LIFETIME28_absolute_correction_max": 8.0000001,
        "all_lifetime_exposures_max": 28.0000001,
        "all_information_fingerprints_equal": True,
        "all_resource_invariants": True,
        "all_outputs_finite": True,
    }
    if dict(gates) != expected_gates:
        raise ValueError("validation scientific gates changed")
    expected_decision = {
        "pass": (
            "Every registered one-sided bootstrap bound and invariant must pass; no "
            "cell may be dropped."
        ),
        "fail": (
            "Retain the adverse validation and do not open integrated primary/confirmation "
            "or EPA tests."
        ),
        "success": (
            "Freeze one integrated common-resource validation using AP_LIFETIME28; "
            "success is not primary or confirmation."
        ),
    }
    expected_runtime = {
        "workers": 4,
        "blas_threads_per_worker": 1,
        "runtime_smoke_seeds": [SEEDS[0]],
        "resume": "only missing hash-identical inputs or predictions; no replacement seed",
    }
    expected_artifacts = {
        "directory": "reports/v6/causal_lifetime_exposure_validation_v1",
        "outcomes": "reports/v6/causal_lifetime_exposure_validation_v1/outcomes",
    }
    expected_protected = [
        "data/external/epa",
        "reports/v6/epa_2025_protocol",
        "reports/v6/test",
        "reports/v6/primary",
        "reports/v6/confirmation",
    ]
    if (dict(registration.get("decision", {})) != expected_decision
            or dict(registration.get("runtime", {})) != expected_runtime
            or dict(registration.get("artifacts", {})) != expected_artifacts
            or registration.get("protected_namespaces") != expected_protected):
        raise ValueError("validation decision, runtime, artifact, or protected scope changed")


def assert_seed_registry(registration: Mapping, registry: Mapping) -> None:
    """Require exclusive world and bootstrap namespaces in the shared seed registry."""
    assert_registration(registration)
    namespaces = registry.get("namespaces")
    if not isinstance(namespaces, Mapping):
        raise TypeError("seed registry namespaces missing")
    world_namespace = registration["worlds"]["seed_namespace"]
    if namespaces.get(world_namespace) != SEEDS:
        raise ValueError("validation world seed namespace changed")
    if namespaces.get(BOOTSTRAP_NAMESPACE) != [BOOTSTRAP_SEED]:
        raise ValueError("validation bootstrap seed namespace changed")
    owners: dict[int, list[str]] = {}
    for name, values in namespaces.items():
        if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
            raise ValueError(f"invalid seed registry namespace: {name}")
        for value in values:
            owners.setdefault(value, []).append(str(name))
    collisions = {seed: names for seed, names in owners.items() if len(names) != 1}
    if collisions:
        raise ValueError(f"seed registry contains collisions: {collisions}")
    expected = {seed: [world_namespace] for seed in SEEDS}
    expected[BOOTSTRAP_SEED] = [BOOTSTRAP_NAMESPACE]
    if any(owners.get(seed) != owner for seed, owner in expected.items()):
        raise ValueError("validation seed or bootstrap namespace collides")


def _bounds(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("finite bootstrap samples required")
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def evaluate(registration: Mapping, rows: list[Mapping]) -> dict:
    assert_registration(registration)
    by_key = {(int(row["seed"]), row["scenario"], row["method"]): row for row in rows}
    expected = len(SEEDS) * len(CELLS) * len(METHODS)
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError("incomplete or duplicate validation matrix")
    gates = registration["scientific_gates"]
    invariants = {}
    for seed in SEEDS:
        for cell in CELLS:
            group = [by_key[seed, cell, method] for method in METHODS]
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
                if row["method"] == "AP_LIFETIME28":
                    invariants[f"cap:{prefix}"] = (
                        float(row["maximum_absolute_correction"])
                        <= gates["AP_LIFETIME28_absolute_correction_max"])
                    invariants[f"lifetime:{prefix}"] = (
                        float(row["maximum_user_lifetime_exposure"])
                        <= gates["all_lifetime_exposures_max"])
    rng = np.random.default_rng(int(gates["bootstrap_seed"]))
    draws = int(gates["bootstrap_replicates"])
    indices = rng.integers(0, len(SEEDS), size=(draws, len(SEEDS)))
    alpha = ((1 - float(gates["confidence_level"]))
             / int(gates["simultaneous_family_size"]))

    def values(cell: str, method: str, field: str) -> np.ndarray:
        return np.asarray([float(by_key[seed, cell, method][field]) for seed in SEEDS])

    tests = {}
    for cell in CELLS[:3]:
        ap, sq = values(cell, "AP_LIFETIME28", "rmse"), values(cell, "SQ", "rmse")
        estimate = float(ap.mean() / sq.mean())
        samples = ap[indices].mean(axis=1) / sq[indices].mean(axis=1)
        lower, upper = _bounds(samples, alpha)
        tests[f"clean:{cell}"] = {"estimate": estimate, "lower": lower, "upper": upper,
            "threshold": gates["clean_AP_to_SQ_ratio_upper_max"],
            "pass": upper <= gates["clean_AP_to_SQ_ratio_upper_max"]}
    clean_ap = values("severe_clean", "AP_LIFETIME28", "rmse")
    clean_sq = values("severe_clean", "SQ", "rmse")
    for attack in CELLS[3:]:
        attack_ap = values(attack, "AP_LIFETIME28", "rmse")
        attack_sq = values(attack, "SQ", "rmse")
        numerator = attack_ap[indices].mean(axis=1) - clean_ap[indices].mean(axis=1)
        denominator = attack_sq[indices].mean(axis=1) - clean_sq[indices].mean(axis=1)
        samples = np.where(denominator > 0, 1 - numerator / denominator, -np.inf)
        finite = samples[np.isfinite(samples)]
        lower = float(np.quantile(finite, alpha)) if len(finite) == len(samples) else -math.inf
        estimate_denominator = float(attack_sq.mean() - clean_sq.mean())
        estimate = (float(1 - (attack_ap.mean() - clean_ap.mean()) / estimate_denominator)
                    if estimate_denominator > 0 else -math.inf)
        threshold = gates[f"{attack.removeprefix('severe_')}_excess_RMSE_attenuation_lower_min"]
        tests[f"attack:{attack}"] = {"estimate": estimate, "lower": lower,
            "upper": float(np.quantile(finite, 1 - alpha)) if len(finite) else math.inf,
            "threshold": threshold, "pass": lower >= threshold,
            "positive_denominator_draws": int(np.isfinite(samples).sum())}
        ap_event = values(attack, "AP_LIFETIME28", "event_recall")
        sq_event = values(attack, "SQ", "event_recall")
        event_samples = 100 * (sq_event[indices].mean(axis=1)
                               - ap_event[indices].mean(axis=1))
        event_lower, event_upper = _bounds(event_samples, alpha)
        tests[f"event:{attack}"] = {
            "estimate": float(100 * (sq_event.mean() - ap_event.mean())),
            "lower": event_lower, "upper": event_upper,
            "threshold": gates[
                "event_recall_loss_vs_SQ_upper_max_percentage_points"],
            "pass": event_upper <= gates[
                "event_recall_loss_vs_SQ_upper_max_percentage_points"],
        }
    public = values("severe_clean", "PUBLIC", "rmse")
    samples = clean_ap[indices].mean(axis=1) / public[indices].mean(axis=1)
    lower, upper = _bounds(samples, alpha)
    tests["citizen_value:severe_clean"] = {
        "estimate": float(clean_ap.mean() / public.mean()), "lower": lower,
        "upper": upper,
        "threshold": gates["severe_clean_AP_to_PUBLIC_ratio_upper_max"],
        "pass": upper <= gates["severe_clean_AP_to_PUBLIC_ratio_upper_max"],
    }
    passed = all(invariants.values()) and all(row["pass"] for row in tests.values())
    return {
        "pass": passed,
        "tests": tests,
        "all_invariants_pass": all(invariants.values()),
        "failed_invariants": sorted(key for key, value in invariants.items() if not value),
        "bootstrap": {"seed": gates["bootstrap_seed"], "replicates": draws,
                      "one_sided_alpha_per_gate": alpha, "family_size": 8,
                      "method": gates["simultaneous_method"]},
        "scope": registration["role"],
    }

"""Outcome-blind registration and frozen aggregate gate evaluation for v2."""
from __future__ import annotations

import math
from collections.abc import Mapping


def candidate_pairs(registration: Mapping) -> set[tuple[str, str]]:
    return {(row["innovation_covariance"], row["residual_forcing"])
            for row in registration["candidates"]}


def assert_exact_family(registration: Mapping) -> None:
    axes = registration["family_axes"]
    expected = {(a, b) for a in axes["innovation_covariance"]
                for b in axes["residual_forcing"]}
    rows = registration["candidates"]
    ids = [row["id"] for row in rows]
    if (len(axes["innovation_covariance"]) != 2
            or len(axes["residual_forcing"]) != 2
            or len(rows) != 4 or len(set(ids)) != 4
            or candidate_pairs(registration) != expected):
        raise ValueError("candidate family must be the exact unique Cartesian 2x2")


def assert_fixed_contract(registration: Mapping) -> None:
    fixed = registration["fixed_estimator"]
    if not (fixed["cap"] == 8.0 and fixed["output_cap"] is True
            and fixed["input_clip"] is False and fixed["loss"] == "huber"
            and fixed["huber_delta"] == 1.345 and fixed["lag"] == 6):
        raise ValueError("cap8/delta1.345/lag6 fixed estimator contract changed")
    if registration["same_information_controls"]["names"] != ["PUBLIC", "SQ", "HUBER"]:
        raise ValueError("PUBLIC/SQ/HUBER are the registered controls")
    gates = registration["joint_gates"]
    if not (gates["clean_candidate_to_SQ_mean_rmse_ratio_max"] == 1.05
            and gates["drift_mean_excess_rmse_attenuation_min"] == .20
            and gates["hotspot_mean_excess_rmse_attenuation_min"] == .20
            and gates["event_recall_mean_loss_vs_SQ_max_percentage_points"] == 5.0):
        raise ValueError("original scientific margins changed")


def registered_seeds(registration: Mapping) -> tuple[set[int], set[int]]:
    calibration = set(map(int, registration["world_roles"]["calibration"]["seeds"]))
    development = set(map(int, registration["world_roles"]["development"]["seeds"]))
    smoke = set(map(int, registration["world_roles"]["runtime_smoke"]["seeds"]))
    lo, hi = map(int, registration["seed_namespace"]["half_open"])
    if (len(calibration) != 8 or len(development) != 8 or calibration & development
            or not smoke <= development or any(not lo <= seed < hi
                                               for seed in calibration | development)):
        raise ValueError("invalid/disjoint calibration-development seed roles")
    return calibration, development


def registry_seed_values(value) -> set[int]:
    found: set[int] = set()
    if isinstance(value, bool):
        return found
    if isinstance(value, int):
        found.add(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            if isinstance(item, (int, Mapping, list, tuple)) and not isinstance(item, bool):
                found |= registry_seed_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= registry_seed_values(item)
    return found


def assert_seed_registry(registration: Mapping, registry: Mapping) -> None:
    calibration, development = registered_seeds(registration)
    namespaces = registry.get("namespaces", {})
    own_calibration = set(map(int, namespaces.get("covariance_forcing_v2_calibration", [])))
    own_development = set(map(int, namespaces.get("covariance_forcing_v2_development", [])))
    if own_calibration != calibration or own_development != development:
        raise ValueError("seed registry does not contain the exact v2 reservations")
    other = {key: value for key, value in namespaces.items()
             if key not in {"covariance_forcing_v2_calibration",
                            "covariance_forcing_v2_development"}}
    collisions = sorted((calibration | development) & registry_seed_values(other))
    if collisions:
        raise ValueError(f"v2 seed namespace collision: {collisions}")


def _mean(rows: list[Mapping], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"finite nonempty {field} values required")
    return sum(values) / len(values)


def evaluate_joint_gates(registration: Mapping, rows: list[Mapping],
                         innovation_scales: Mapping[str, float] | None = None) -> dict:
    """Evaluate frozen aggregate development gates; no family extension is possible."""
    assert_exact_family(registration)
    assert_fixed_contract(registration)
    gates = registration["joint_gates"]
    seeds = sorted(registered_seeds(registration)[1])
    cells = registration["world_roles"]["development"]["cells"]
    candidates = registration["candidates"]
    by_key = {(int(row["seed"]), row["candidate"], row["scenario"], row["method"]): row
              for row in rows}
    expected_count = len(seeds) * len(candidates) * len(cells) * 4
    if len(rows) != expected_count or len(by_key) != expected_count:
        raise ValueError("incomplete or duplicate registered outcome matrix")
    decisions = []
    for candidate in candidates:
        candidate_id = candidate["id"]
        invariants: dict[str, bool] = {}
        for seed in seeds:
            for cell in cells:
                group = [by_key[seed, candidate_id, cell, method]
                         for method in ("CANDIDATE", "PUBLIC", "SQ", "HUBER")]
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
                            float(row["solver_failure_rate"]) <= gates["all_solver_failure_rates_max"]
                            and float(row["projected_gradient_inf"]) <= gates["all_projected_gradient_inf_max"])
                    if row["method"] == "CANDIDATE":
                        invariants[f"cap:{prefix}"] = float(row["maximum_absolute_correction"]) <= \
                            gates["all_candidate_absolute_corrections_max"]
        clean_candidate = [by_key[s, candidate_id, "severe_clean", "CANDIDATE"] for s in seeds]
        clean_sq = [by_key[s, candidate_id, "severe_clean", "SQ"] for s in seeds]
        clean_ratio = _mean(clean_candidate, "rmse") / _mean(clean_sq, "rmse")
        attenuation = {}
        event_loss = {}
        for cell, gate_name in (("severe_drift", "drift_mean_excess_rmse_attenuation_min"),
                                ("severe_hotspot", "hotspot_mean_excess_rmse_attenuation_min")):
            attack_candidate = [by_key[s, candidate_id, cell, "CANDIDATE"] for s in seeds]
            attack_sq = [by_key[s, candidate_id, cell, "SQ"] for s in seeds]
            candidate_excess = _mean(attack_candidate, "rmse") - _mean(clean_candidate, "rmse")
            sq_excess = _mean(attack_sq, "rmse") - _mean(clean_sq, "rmse")
            attenuation[cell] = (1 - candidate_excess / sq_excess
                                 if sq_excess > 0 else -math.inf)
            event_loss[cell] = 100 * (_mean(attack_sq, "event_recall")
                                      - _mean(attack_candidate, "event_recall"))
        scientific = {
            "clean": clean_ratio <= gates["clean_candidate_to_SQ_mean_rmse_ratio_max"],
            "drift": attenuation["severe_drift"] >= gates["drift_mean_excess_rmse_attenuation_min"],
            "hotspot": attenuation["severe_hotspot"] >= gates["hotspot_mean_excess_rmse_attenuation_min"],
            "drift_event": event_loss["severe_drift"] <= gates["event_recall_mean_loss_vs_SQ_max_percentage_points"],
            "hotspot_event": event_loss["severe_hotspot"] <= gates["event_recall_mean_loss_vs_SQ_max_percentage_points"],
        }
        passed = all(invariants.values()) and all(scientific.values())
        normalized = max(
            clean_ratio / gates["clean_candidate_to_SQ_mean_rmse_ratio_max"],
            (1 - attenuation["severe_drift"]) / (1 - gates["drift_mean_excess_rmse_attenuation_min"]),
            (1 - attenuation["severe_hotspot"]) / (1 - gates["hotspot_mean_excess_rmse_attenuation_min"]),
            max(0.0, event_loss["severe_drift"]) / gates["event_recall_mean_loss_vs_SQ_max_percentage_points"],
            max(0.0, event_loss["severe_hotspot"]) / gates["event_recall_mean_loss_vs_SQ_max_percentage_points"],
        )
        decisions.append({"candidate": candidate_id, "pass": passed,
            "scientific_checks": scientific, "all_invariants_pass": all(invariants.values()),
            "failed_invariants": sorted(key for key, value in invariants.items() if not value),
            "metrics": {"clean_ratio": clean_ratio, "attenuation": attenuation,
                        "event_loss_percentage_points": event_loss,
                        "worst_normalized_condition_loss": normalized},
            "mean_runtime_seconds": _mean([row for row in rows
                if row["candidate"] == candidate_id], "wall_seconds")})
    feasible = [row for row in decisions if row["pass"]]
    selected = None
    if feasible:
        scales = innovation_scales or {}
        selected = min(feasible, key=lambda row: (
            row["metrics"]["worst_normalized_condition_loss"],
            float(scales.get(next(item["innovation_covariance"] for item in candidates
                                  if item["id"] == row["candidate"]), math.inf)),
            row["mean_runtime_seconds"], row["candidate"]))["candidate"]
    return {"candidates": decisions, "eligible": [row["candidate"] for row in feasible],
            "selected_candidate": selected, "family_closed": selected is None,
            "extension_authorized": False,
            "scope": "exposed mechanistic synthetic development; no confirmatory p-values"}

import copy
import json
from pathlib import Path

import pytest

from airproof.v6_covariance_forcing_registration import (
    assert_exact_family,
    assert_fixed_contract,
    assert_seed_registry,
    evaluate_joint_gates,
    registered_seeds,
)

ROOT = Path(__file__).resolve().parents[1]


def registration():
    return json.loads((ROOT / "configs/v6/innovation_covariance_forcing_development_v2.json").read_text())


def test_registration_is_exact_and_preserves_claim_gates_and_roles():
    value = registration()
    assert_exact_family(value)
    assert_fixed_contract(value)
    calibration, development = registered_seeds(value)
    assert len(calibration) == len(development) == 8
    assert not calibration & development
    assert value["scale"] == {"agents": 1000, "grid_side": 32, "groups": 4,
        "acquisition_epochs": 672, "burn_in_epochs": 48, "fixed_lag_epochs": 6,
        "transport_drain_epochs": 24, "epoch_hours": 1}
    assert value["historical_reuse"]["rerun"] is False
    assert "residual_identity" in value["synthetic_boundary"]


def test_duplicate_or_extra_candidate_is_rejected():
    value = registration()
    value["candidates"].append(dict(value["candidates"][0], id="extra"))
    with pytest.raises(ValueError, match="exact unique Cartesian 2x2"):
        assert_exact_family(value)


def test_seed_registry_contains_exact_disjoint_reservation():
    value = registration()
    registry = json.loads((ROOT / "configs/v6/seed_registry.json").read_text())
    assert_seed_registry(value, registry)
    broken = copy.deepcopy(registry)
    broken["namespaces"]["development"].append(6241007)
    with pytest.raises(ValueError, match="collision"):
        assert_seed_registry(value, broken)


def complete_passing_rows(value):
    rows = []
    for seed in value["world_roles"]["development"]["seeds"]:
        for candidate in value["candidates"]:
            for cell in value["world_roles"]["development"]["cells"]:
                for method in ("CANDIDATE", "PUBLIC", "SQ", "HUBER"):
                    if cell == "severe_clean":
                        rmse = {"CANDIDATE": 1.04, "SQ": 1.0,
                                "PUBLIC": 1.2, "HUBER": 1.03}[method]
                    else:
                        rmse = {"CANDIDATE": 1.40, "SQ": 1.50,
                                "PUBLIC": 1.2, "HUBER": 1.42}[method]
                    rows.append({"seed": seed, "candidate": candidate["id"],
                        "scenario": cell, "method": method, "rmse": rmse,
                        "event_recall": .90 if method == "CANDIDATE" else .92,
                        "information_fingerprint": f"{seed}:{candidate['id']}:{cell}",
                        "wall_seconds": 1.0, "process_cpu_seconds": .9,
                        "peak_rss_bytes": 1000, "solver_failure_rate": 0.0,
                        "projected_gradient_inf": 1e-8,
                        "maximum_absolute_correction": 8.0,
                        "finite_outputs": True, "resource_invariants_pass": True})
    return rows


def test_joint_gate_uses_complete_aggregate_matrix_and_all_invariants():
    value = registration()
    rows = complete_passing_rows(value)
    result = evaluate_joint_gates(value, rows, {"independence": 2.0,
        "world_cross_fitted_covariance": 1.8})
    assert len(result["eligible"]) == 4
    assert result["selected_candidate"] is not None
    assert result["family_closed"] is False
    rows.pop()
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_joint_gates(value, rows)


def test_one_mismatched_information_fingerprint_disqualifies_candidate():
    value = registration()
    rows = complete_passing_rows(value)
    target = value["candidates"][0]["id"]
    for row in rows:
        if (row["candidate"] == target and row["seed"] == 6241020
                and row["scenario"] == "severe_clean" and row["method"] == "PUBLIC"):
            row["information_fingerprint"] = "different"
    result = evaluate_joint_gates(value, rows)
    decision = next(item for item in result["candidates"] if item["candidate"] == target)
    assert decision["pass"] is False
    assert decision["failed_invariants"]


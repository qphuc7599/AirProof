"""Small contract tests; no registered v3 outcome or protected source is opened."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from airproof import v6_history_normalized_runner as runner
from airproof.v6_history_normalized_registration import assert_registration, evaluate

ROOT = Path(__file__).resolve().parents[1]


def registration():
    return json.loads((ROOT / "configs/v6/history_normalized_huber_development_v3.json")
                      .read_text(encoding="utf-8"))


def matrix(*, drift_candidate=1.6):
    reg = registration()
    rows = []
    for seed in reg["inputs"]["development_seeds"]:
        for candidate in reg["candidates"]:
            for scenario in reg["inputs"]["cells"]:
                for method in ("CANDIDATE", "PUBLIC", "SQ", "HUBER"):
                    attacked = scenario != "severe_clean"
                    rmse = (drift_candidate if method == "CANDIDATE" and scenario == "severe_drift"
                            else 1.6 if method == "CANDIDATE" and attacked
                            else 2.0 if method == "SQ" and attacked else 1.0)
                    rows.append({
                        "seed": seed, "candidate": candidate["id"], "scenario": scenario,
                        "method": method, "information_fingerprint": f"same:{seed}:{scenario}",
                        "rmse": rmse, "event_recall": .90 if method == "CANDIDATE" else .92,
                        "wall_seconds": 1., "process_cpu_seconds": .5, "peak_rss_bytes": 1000,
                        "finite_outputs": True, "resource_invariants_pass": True,
                        "solver_failure_rate": 0., "projected_gradient_inf": 0.,
                        "maximum_absolute_correction": 1.,
                        "maximum_user_window_weight": candidate["per_user_weight_budget"],
                    })
    return reg, rows


def test_frozen_budgets_and_scientific_margins_are_immutable():
    reg = registration()
    assert_registration(reg)
    for mutation in (lambda x: x["candidates"][0].update(per_user_weight_budget=.5),
                     lambda x: x["joint_gates"].update(
                         drift_mean_excess_rmse_attenuation_min=.19),
                     lambda x: x["fixed_estimator"].update(cap=9.0),
                     lambda x: x["fixed_estimator"].update(lambda_zero=999.0),
                     lambda x: x["inputs"].update(cells=["severe_clean"]),
                     lambda x: x["inputs"].update(
                         development_seeds=x["inputs"]["development_seeds"][:-1]
                         + [x["inputs"]["development_seeds"][0]]),
                     lambda x: x["controls"].update(reused_methods=["SQ"])):
        changed = copy.deepcopy(reg)
        mutation(changed)
        with pytest.raises(ValueError):
            assert_registration(changed)


def test_synthetic_joint_gate_pass_and_drift_fail_are_diagnostic():
    reg, passing = matrix(drift_candidate=1.6)
    result = evaluate(reg, passing)
    assert result["eligible"] == [row["id"] for row in reg["candidates"]]
    assert result["selected_candidate"] == "history_budget1"
    _, failing = matrix(drift_candidate=1.9)
    result = evaluate(reg, failing)
    assert result["eligible"] == [] and result["family_closed"] is True
    assert all(not row["scientific_checks"]["drift"] for row in result["candidates"])
    assert all(row["scientific_checks"]["clean"] for row in result["candidates"])


def test_information_or_user_budget_tamper_fails_candidate():
    reg, rows = matrix()
    rows[0]["information_fingerprint"] = "tampered"
    rows[1]["maximum_user_window_weight"] = 999
    result = evaluate(reg, rows)
    candidate = next(row for row in result["candidates"]
                     if row["candidate"] == rows[0]["candidate"])
    assert candidate["pass"] is False
    assert any(name.startswith(("fingerprint:", "user_budget:"))
               for name in candidate["failed_invariants"])


def test_job_writes_all_predictions_before_loading_scoring(tmp_path, monkeypatch):
    reg = registration()
    reg["inputs"]["campaign_bundle"] = "bundle"
    reg["inputs"]["input_lock"] = "input_lock.json"
    reg["artifacts"]["outcomes"] = "outcomes"
    reg["controls"]["source_campaign"] = "v2-outcomes"
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "scoring" / "1").mkdir(parents=True)
    np.savez_compressed(tmp_path / "bundle" / "scoring" / "1" / "scoring_only.npz",
                        fixture=np.array([1]))
    (tmp_path / "input_lock.json").write_text(
        json.dumps({"input_manifest_sha256": "a" * 64}), encoding="utf-8")
    (tmp_path / "bundle" / "global_calibration.json").write_text(json.dumps({
        "innovation_scales": {"world_cross_fitted_covariance": 1.},
        "innovation_variances": {"world_cross_fitted_covariance": 1.},
    }), encoding="utf-8")
    observations = ()
    view = {"public_center": np.ones((2, 2)), "observations": observations,
            "arrival_map": {}, "operator": None, "residual_forcing": np.zeros((2, 2))}
    monkeypatch.setattr(runner, "load_prediction_view", lambda *a, **k: view)
    monkeypatch.setattr(runner, "information_fingerprint", lambda *a, **k: "fingerprint")
    v2_target = tmp_path / "v2-outcomes" / "jobs" / "1" / "severe_clean"
    v2_target.mkdir(parents=True)
    v2_target.joinpath("result.json").write_text(json.dumps({"rows": [{
        "candidate": reg["controls"]["source_candidate"], "method": method,
        "resource_invariants_pass": True, "information_fingerprint": "fingerprint",
    } for method in ("PUBLIC", "SQ", "HUBER", "CANDIDATE")]}), encoding="utf-8")
    monkeypatch.setattr(runner, "_predict_one", lambda *a, **k: (
        np.ones((2, 2)), np.ones((2, 2)), {"epoch_terms": [],
        "solver_failure_rate": 0., "maximum_correction": 0.}))

    def scoring(*args, **kwargs):
        target = tmp_path / "outcomes" / "jobs" / "1" / "severe_clean"
        assert all((target / row["id"] / "prediction.npz").is_file()
                   for row in reg["candidates"])
        return {"evaluation_truth": np.ones((2, 2)), "event_mask": np.ones((2, 2), bool),
                "event_threshold": .5, "cell_groups": np.array([0, 1])}

    monkeypatch.setattr(runner, "load_scoring_view", scoring)
    runner._job((str(tmp_path), reg, 1, "severe_clean", "b" * 64))
    target = tmp_path / "outcomes" / "jobs" / "1" / "severe_clean"
    assert (target / "result.json").is_file()
    manifest = target / reg["candidates"][0]["id"] / "prediction_manifest.json"
    saved = manifest.read_bytes()
    manifest.unlink()
    with pytest.raises(ValueError, match="one-sided"):
        runner._job((str(tmp_path), reg, 1, "severe_clean", "b" * 64))
    manifest.write_bytes(saved)
    result = json.loads((target / "result.json").read_text(encoding="utf-8"))
    result["rows"][0]["prediction_sha256"] = "tampered"
    (target / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="tampered v3 result"):
        runner._job((str(tmp_path), reg, 1, "severe_clean", "b" * 64))

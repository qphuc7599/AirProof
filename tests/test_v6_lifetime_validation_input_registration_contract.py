"""Validation input/registration contracts; no registered outcome or protected data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from airproof import v6_lifetime_validation_runner as validation_runner
from airproof.records import canonical_json
from airproof.v6_covariance_forcing_inputs import save_csr, sha256_file
from airproof.v6_lifetime_validation_inputs import (
    load_prediction_view,
    load_scoring_view,
    verify_bundle,
)
from airproof.v6_lifetime_validation_registration import (
    CELLS,
    METHODS,
    SEEDS,
    assert_registration,
    evaluate,
)
from airproof.v6_lifetime_validation_runner import planned_jobs
from scripts.run_v6_lifetime_validation_v1 import require_ready

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs/v6/causal_lifetime_exposure_validation_v1.json"


def registration() -> dict:
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))


def rows(*, attack_ap: float = 1.5) -> list[dict]:
    result = []
    for seed in SEEDS:
        for cell in CELLS:
            for method in METHODS:
                attacked = cell in ("severe_drift", "severe_hotspot")
                rmse = (attack_ap if method == "AP_LIFETIME28" and attacked
                        else .9 if method == "AP_LIFETIME28"
                        else 2.0 if method == "SQ" and attacked
                        else 1.1 if method == "PUBLIC" else 1.0)
                result.append({
                    "seed": seed, "scenario": cell, "method": method,
                    "information_fingerprint": f"same:{seed}:{cell}",
                    "rmse": rmse, "event_recall": .90 if method == "AP_LIFETIME28" else .92,
                    "wall_seconds": 1., "process_cpu_seconds": .5, "peak_rss_bytes": 1024,
                    "finite_outputs": True, "resource_invariants_pass": True,
                    "solver_failure_rate": 0., "projected_gradient_inf": 0.,
                    "maximum_absolute_correction": 1.,
                    "maximum_user_lifetime_exposure": 28.,
                })
    return result


def test_exact_validation_seed_cell_method_and_gate_contract():
    reg = registration()
    assert_registration(reg)
    assert reg["worlds"]["seeds"] == list(range(6241200, 6241212))
    assert reg["worlds"]["cells"] == CELLS
    assert reg["methods"] == METHODS
    mutations = (
        lambda value: value["worlds"].update(seeds=SEEDS[:-1]),
        lambda value: value["worlds"].update(cells=CELLS[:-1]),
        lambda value: value["scientific_gates"].update(
            drift_excess_RMSE_attenuation_lower_min=.19),
        lambda value: value["scientific_gates"].update(bootstrap_seed=6241300),
    )
    for mutate in mutations:
        changed = copy.deepcopy(reg)
        mutate(changed)
        with pytest.raises(ValueError):
            assert_registration(changed)
    assert len(planned_jobs(reg, smoke=False)) == 60
    assert len(planned_jobs(reg, smoke=False)) * len(METHODS) == 240
    assert planned_jobs(reg, smoke=True) == [(6241200, cell) for cell in CELLS]


def test_bootstrap_fixture_passes_and_isolates_attack_failure():
    reg = registration()
    passing = evaluate(reg, rows(attack_ap=1.5))
    assert passing["pass"] is True
    assert passing["bootstrap"] == {
        "seed": 6241299, "replicates": 20000,
        "one_sided_alpha_per_gate": pytest.approx(.00625), "family_size": 8,
        "method": reg["scientific_gates"]["simultaneous_method"],
    }
    failing = evaluate(reg, rows(attack_ap=1.9))
    assert failing["pass"] is False
    assert failing["tests"]["attack:severe_drift"]["pass"] is False
    assert failing["tests"]["attack:severe_hotspot"]["pass"] is False
    assert all(failing["tests"][f"clean:{cell}"]["pass"]
               for cell in ("anchor_clean", "outage_clean", "severe_clean"))


def test_attack_prediction_and_scoring_reuse_clean_context_and_truth(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    prediction = bundle / "prediction" / str(SEEDS[0])
    scoring = bundle / "scoring" / str(SEEDS[0])
    mechanism = bundle / "mechanism"
    prediction.mkdir(parents=True)
    scoring.mkdir(parents=True)
    mechanism.mkdir(parents=True)
    public = np.arange(8, dtype=float).reshape(2, 4)
    residual = np.ones_like(public)
    np.savez_compressed(prediction / "context_severe_clean.npz",
        public_center=public, prediction_epochs=np.array([48, 49]),
        public_center_available_at=np.array([48, 49]), physical_forcing=np.ones_like(public),
        physical_forcing_available_at=np.array([48, 49]), residual_forcing=residual,
        previous_public_center=np.zeros(4))
    truth = public + 2
    np.savez_compressed(scoring / "scoring_severe_clean.npz",
        evaluation_truth=truth, event_mask=truth > 4, event_threshold=np.asarray(4.),
        cell_groups=np.arange(4))
    (prediction / "observations_severe_drift.npz").write_bytes(b"opaque")
    save_csr(mechanism / "physical_operator.npz", sparse.eye(4, format="csr"))
    manifest = {"seeds": SEEDS, "cells": CELLS,
                "context_base": {cell: "severe_clean" if cell.startswith("severe_")
                                 else cell for cell in CELLS}}
    monkeypatch.setattr("airproof.v6_lifetime_validation_inputs.verify_bundle",
                        lambda *args, **kwargs: manifest)
    monkeypatch.setattr("airproof.v6_lifetime_validation_inputs.restore_records",
                        lambda path: (("attacked-record",), {"n": 49}))
    view = load_prediction_view(tmp_path, bundle, SEEDS[0], "severe_drift",
                                expected_manifest_sha256="0" * 64)
    score = load_scoring_view(bundle, SEEDS[0], "severe_drift")
    np.testing.assert_array_equal(view["public_center"], public)
    np.testing.assert_array_equal(view["residual_forcing"], residual)
    np.testing.assert_array_equal(score["evaluation_truth"], truth)
    assert view["observations"] == ("attacked-record",)
    assert not ({"evaluation_truth", "event_mask", "attack_label"} & set(view))


def test_prediction_boundary_rejects_scoring_field(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    path = bundle / "prediction" / str(SEEDS[0])
    path.mkdir(parents=True)
    np.savez_compressed(path / "context_severe_clean.npz",
                        public_center=np.ones((1, 1)), truth=np.ones((1, 1)))
    manifest = {"seeds": SEEDS, "cells": CELLS,
                "context_base": {"severe_drift": "severe_clean"}}
    monkeypatch.setattr("airproof.v6_lifetime_validation_inputs.verify_bundle",
                        lambda *args, **kwargs: manifest)
    with pytest.raises(ValueError, match="scoring field"):
        load_prediction_view(tmp_path, bundle, SEEDS[0], "severe_drift",
                             expected_manifest_sha256="0" * 64)


def test_bundle_and_external_manifest_tamper_fail_closed(tmp_path, monkeypatch):
    root, bundle = tmp_path / "root", tmp_path / "bundle"
    config = root / "configs/v6"
    config.mkdir(parents=True)
    reg = registration()
    config.joinpath("causal_lifetime_exposure_validation_v1.json").write_text(
        json.dumps(reg), encoding="utf-8")
    source_lock = root / reg["input_producer"]["source_lock"]
    source_lock.parent.mkdir(parents=True)
    source_lock.write_text("locked", encoding="utf-8")
    payload = bundle / "prediction/payload.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"original")
    payloads = {"prediction/payload.bin": sha256_file(payload)}
    monkeypatch.setattr(
        "airproof.v6_lifetime_validation_inputs._expected_payloads",
        lambda _registration: set(payloads),
    )
    monkeypatch.setattr(
        "airproof.v6_lifetime_validation_inputs._source_lineage",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        "airproof.v6_lifetime_validation_inputs._validate_bundle_semantics",
        lambda *_args: None,
    )
    manifest = {
        "schema_version": 1,
        "role": "hash-bound independent synthetic validation inputs; zero outcomes",
        "registration": "configs/v6/causal_lifetime_exposure_validation_v1.json",
        "registration_sha256": sha256_file(
            config / "causal_lifetime_exposure_validation_v1.json"),
        "source_lock_sha256": sha256_file(source_lock), "scientific_outcomes": 0,
        "protected_namespaces_opened": False, "seeds": SEEDS, "cells": CELLS,
        "context_base": {
            cell: ("severe_clean" if cell in ("severe_drift", "severe_hotspot") else cell)
            for cell in CELLS
        },
        "calibration_source": reg["fixed_estimator"]["innovation_calibration"],
        "calibration_source_sha256": "0" * 64,
        "source_lineage": {},
        "expected_payload_count": 1,
        "payload_sha256": payloads,
        "payload_root_sha256": hashlib.sha256(canonical_json(payloads)).hexdigest(),
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    external = sha256_file(manifest_path)
    assert verify_bundle(root, bundle, expected_manifest_sha256=external) == manifest
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload hash"):
        verify_bundle(root, bundle, expected_manifest_sha256=external)
    payload.write_bytes(b"original")
    manifest["scientific_outcomes"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        verify_bundle(root, bundle, expected_manifest_sha256=external)


def test_controller_requires_exact_zero_outcome_240_row_readiness(tmp_path):
    root = tmp_path
    config = root / "configs/v6"
    config.mkdir(parents=True)
    reg = registration()
    config.joinpath("causal_lifetime_exposure_validation_v1.json").write_text(
        json.dumps(reg), encoding="utf-8")
    source = root / reg["input_producer"]["source_lock"]
    inputs = root / reg["input_producer"]["input_lock"]
    readiness = root / reg["input_producer"]["readiness"]
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    inputs.write_text("inputs", encoding="utf-8")
    ready = {
        "pass": True, "expected_jobs": 60, "expected_rows": 240,
        "scientific_outcomes_read_or_generated": 0,
        "source_lock_sha256": sha256_file(source),
        "input_lock_sha256": sha256_file(inputs),
    }
    readiness.write_text(json.dumps(ready), encoding="utf-8")
    assert require_ready(root) == ready
    for field, value in (("expected_rows", 239),
                         ("scientific_outcomes_read_or_generated", 1),
                         ("pass", False)):
        changed = dict(ready)
        changed[field] = value
        readiness.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(RuntimeError, match="readiness"):
            require_ready(root)


def test_dedicated_runner_hashes_all_predictions_before_scoring(tmp_path, monkeypatch):
    reg = registration()
    reg["input_producer"]["bundle"] = "bundle"
    reg["input_producer"]["input_lock"] = "input_lock.json"
    reg["artifacts"]["outcomes"] = "outcomes"
    bundle = tmp_path / "bundle"
    (bundle / "prediction" / str(SEEDS[0])).mkdir(parents=True)
    (bundle / "scoring" / str(SEEDS[0])).mkdir(parents=True)
    np.savez_compressed(bundle / "scoring" / str(SEEDS[0]) / "scoring_severe_clean.npz",
                        fixture=np.array([1]))
    (tmp_path / "input_lock.json").write_text(
        json.dumps({"manifest_sha256": "a" * 64}), encoding="utf-8")
    (bundle / "global_calibration.json").write_text(json.dumps({
        "innovation_scales": {"world_cross_fitted_covariance": 1.},
        "innovation_variances": {"world_cross_fitted_covariance": 1.},
    }), encoding="utf-8")
    (bundle / "prediction" / str(SEEDS[0]) / "summary.json").write_text(json.dumps({
        "cell_summary": {"severe_drift": {"resource_invariants_pass": True}}}),
        encoding="utf-8")
    epochs = np.arange(48, 678)
    view = {"public_center": np.ones((630, 1)), "prediction_epochs": epochs,
            "observations": (), "arrival_map": {}, "operator": sparse.eye(1),
            "residual_forcing": np.zeros((630, 1))}
    monkeypatch.setattr(validation_runner, "load_prediction_view", lambda *a, **k: view)
    monkeypatch.setattr(validation_runner, "information_fingerprint",
                        lambda *a, **k: "same-information")
    monkeypatch.setattr(validation_runner, "causal_lifetime_exposure_weights",
                        lambda *a, **k: ((), {"maximum_user_lifetime_exposure": 0.,
                            "effective_records": 0, "scaled_records": 0,
                            "dropped_exhausted_records": 0, "total_lifetime_exposure": 0.}))
    monkeypatch.setattr(validation_runner, "_predict_one", lambda *a, **k: (
        np.ones((630, 1)), np.ones((630, 1)), {"epoch_terms": [],
        "solver_failure_rate": 0., "maximum_correction": 0.}))

    def scoring(*args, **kwargs):
        target = tmp_path / "outcomes" / "jobs" / str(SEEDS[0]) / "severe_drift"
        for method in METHODS:
            prediction = target / method / "prediction.npz"
            manifest = target / method / "prediction_manifest.json"
            assert prediction.is_file() and manifest.is_file()
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            assert saved["prediction_sha256"] == sha256_file(prediction)
            assert saved["scoring_loaded_during_prediction"] is False
        return {"evaluation_truth": np.ones((624, 1)),
                "event_mask": np.ones((624, 1), bool), "event_threshold": .5,
                "cell_groups": np.array([0])}

    monkeypatch.setattr(validation_runner, "load_scoring_view", scoring)
    result = validation_runner._job((
        str(tmp_path), reg, SEEDS[0], "severe_drift", "b" * 64, "c" * 64))
    assert result["row_count"] == 4
    assert {row["rmse_clock"] for row in result["rows"]} == {
        "fixed_lag_reconstructed_after_registered_lag"}
    assert {tuple(row["prediction_horizon_half_open"]) for row in result["rows"]} == {
        (48, 678)}
    assert {tuple(row["scoring_horizon_half_open"]) for row in result["rows"]} == {
        (48, 672)}


def test_analyzer_recomputes_metrics_instead_of_trusting_result_rows(tmp_path, monkeypatch):
    from scripts import analyze_v6_lifetime_validation_v1 as analyzer

    reg = registration()
    reg["worlds"]["cells"] = ["severe_drift"]
    reg["runtime"]["runtime_smoke_seeds"] = [SEEDS[0]]
    reg["input_producer"].update(
        bundle="bundle", source_lock="source.lock", input_lock="input.lock")
    reg["artifacts"]["outcomes"] = "outcomes"
    bundle = tmp_path / "bundle"
    target = tmp_path / "outcomes/jobs" / str(SEEDS[0]) / "severe_drift"
    scoring_dir = bundle / "scoring" / str(SEEDS[0])
    prediction_dir = bundle / "prediction" / str(SEEDS[0])
    target.mkdir(parents=True)
    scoring_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)
    (tmp_path / "source.lock").write_text("source", encoding="utf-8")
    (tmp_path / "input.lock").write_text(
        json.dumps({"manifest_sha256": "e" * 64}), encoding="utf-8")
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    truth = np.ones((2, 1))
    np.savez_compressed(scoring_dir / "scoring_severe_clean.npz",
        evaluation_truth=truth, event_mask=np.ones_like(truth, bool),
        event_threshold=np.asarray(.5), cell_groups=np.array([0]))
    prediction_dir.joinpath("summary.json").write_text(json.dumps({
        "cell_summary": {"severe_drift": {"resource_invariants_pass": True}}}),
        encoding="utf-8")
    result_rows = []
    for method in METHODS:
        arm = target / method
        arm.mkdir()
        prediction = arm / "prediction.npz"
        np.savez_compressed(prediction, live=truth, reconstructed=truth)
        manifest = {
            "scoring_loaded_during_prediction": False,
            "prediction_sha256": sha256_file(prediction),
            "information_fingerprint": "same", "solver_failure_rate": 0.,
            "projected_gradient_inf": 0., "maximum_absolute_correction": 0.,
            "maximum_user_lifetime_exposure": 28., "wall_seconds": 1.,
            "process_cpu_seconds": .5, "peak_rss_bytes": 1000, "identity": method,
        }
        arm.joinpath("prediction_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        result_rows.append({
            "seed": SEEDS[0], "scenario": "severe_drift", "method": method,
            "rmse": 99., "mae": 0., "worst_group_rmse": 0., "live_rmse": 0.,
            "event_recall": 1., "event_support": 2, "event_threshold": .5,
            "prediction_horizon_half_open": [48, 678],
            "scoring_horizon_half_open": [48, 672],
            "rmse_clock": "fixed_lag_reconstructed_after_registered_lag",
            "live_rmse_clock": "immutable_prediction_at_epoch",
            "event_recall_clock": "immutable_live_prediction_at_epoch",
            "information_fingerprint": "same", "solver_failure_rate": 0.,
            "projected_gradient_inf": 0., "maximum_absolute_correction": 0.,
            "maximum_user_lifetime_exposure": 28., "wall_seconds": 1.,
            "process_cpu_seconds": .5, "peak_rss_bytes": 1000,
            "prediction_identity": method, "prediction_sha256": manifest[
                "prediction_sha256"],
            "scoring_sha256": sha256_file(scoring_dir / "scoring_severe_clean.npz"),
            "resource_invariants_pass": True, "finite_outputs": True,
        })
    target.joinpath("result.json").write_text(json.dumps({
        "status": "complete", "seed": SEEDS[0], "cell": "severe_drift",
        "source_lock_sha256": sha256_file(tmp_path / "source.lock"),
        "input_lock_sha256": sha256_file(tmp_path / "input.lock"),
        "row_count": 4, "rows": result_rows,
    }), encoding="utf-8")
    monkeypatch.setattr(analyzer, "ROOT", tmp_path)
    monkeypatch.setattr(analyzer, "verify_source_lock", lambda *a: {})
    monkeypatch.setattr(analyzer, "verify_bundle",
                        lambda *a, **k: {"payload_root_sha256": "d" * 64})
    monkeypatch.setattr(analyzer, "load_scoring_view", lambda *a: {
        "evaluation_truth": truth, "event_mask": np.ones_like(truth, bool),
        "event_threshold": np.asarray(.5), "cell_groups": np.array([0])})
    rows_verified, audit = analyzer.collect(reg, smoke=True)
    assert rows_verified and rows_verified[0]["rmse"] == 0.
    assert any("recomputed_row" in failure and "rmse" in failure
               for failure in audit["failures"])

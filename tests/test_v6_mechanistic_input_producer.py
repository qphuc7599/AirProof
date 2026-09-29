import hashlib
import json

import numpy as np
import pytest
from scipy import sparse

from airproof.field import synthetic_field, synthetic_field_with_mechanism
from airproof.records import Observation
from airproof.v6_mechanistic_input_producer import (
    load_calibration_audit_bundle,
    load_producer_bundle,
    produce_provenance_complete_bundle,
)


def observations():
    rows = []
    for epoch in range(1, 7):
        for user in range(2):
            rows.append(Observation(
                user, epoch, user, user, 10.0 + epoch + 0.2 * user,
                1.0, 0.9, 512, f"n-{epoch}-{user}", epoch, epoch + user % 2))
    return rows


def produce(path):
    epochs = np.arange(1, 7)
    center = np.column_stack((10 + epochs, 11 + epochs)).astype(float)
    truth = center[:3] + np.array([[.5, -.5]])
    transitions = [sparse.eye(2, format="csr") for _ in epochs]
    physical = np.ones_like(center)
    return produce_provenance_complete_bundle(
        path, observations=observations(), public_center=center,
        prediction_epochs=epochs, public_center_available_at=epochs,
        transitions=transitions, transition_latest_input_at=epochs,
        physical_forcing=physical, physical_forcing_latest_input_at=epochs,
        previous_public_center=np.array([9., 10.]),
        calibration_half_open=(1, 4), evaluation_half_open=(4, 7),
        calibration_truth={"world-0": truth, "world-1": truth},
        calibration_truth_available_at={"world-0": np.array([1, 2, 3]),
                                        "world-1": np.array([1, 2, 3])},
        calibration_split_id="closed-cal", evaluation_split_id="exposed-eval",
        public_model_id="fixed-public-v2", sensor_model_id="fixed-sensor-v2",
        source_hashes={"fixture": "0" * 64}, source_roles={
            "citizen_observations": "exact_selected_citizen_channel",
            "public_center": "frozen_public_predictor",
            "transition": "declared_physical_operator",
            "physical_forcing": "declared_public_exogenous",
            "calibration_truth": "closed_calibration_labels",
        }, calibration_pair_context={f"n-{e}-{u}": {
            "calibration_world_id": f"world-{u}",
            "calibration_time_block_id": f"block-{e // 2}"}
            for e in range(1, 4) for u in range(2)},
        development_seed_id="seed-fixture", cross_fit_folds=2)


def test_bundle_is_complete_hash_bound_and_evaluation_blind(tmp_path):
    directory = tmp_path / "innovation_covariance_forcing_development_v2" / "producer_inputs"
    manifest = produce(directory)
    expected = hashlib.sha256((directory / "producer_manifest.json").read_bytes()).hexdigest()
    loaded = load_producer_bundle(directory, expected_manifest_sha256=expected)
    assert manifest["authorizes_outcome_execution"] is False
    assert len(loaded["evaluation_observations"]) == 6
    audit = load_calibration_audit_bundle(directory, expected_manifest_sha256=expected)
    assert len(audit["calibration_observations"]) == 6
    assert audit["calibration_statistics"]["full_calibration"]["psd_audit_pass"]
    evaluation_text = (directory / "citizen_evaluation_records.jsonl").read_text()
    assert "truth" not in evaluation_text and "attack" not in evaluation_text
    pairs = [json.loads(line) for line in (directory / "paired_errors_calibration.jsonl").read_text().splitlines()]
    assert {row["acquisition_epoch"] for row in pairs} == {1, 2, 3}
    assert {row["nullifier"] for row in pairs} == {f"n-{e}-{u}" for e in range(1, 4) for u in range(2)}


def test_bundle_rejects_future_physics_and_collision(tmp_path):
    directory = tmp_path / "bundle"
    produce(directory)
    with pytest.raises(FileExistsError):
        produce(directory)
    bad = tmp_path / "future"
    epochs = np.arange(1, 7)
    center = np.column_stack((10 + epochs, 11 + epochs)).astype(float)
    with pytest.raises(ValueError, match="future producer input"):
        produce_provenance_complete_bundle(
            bad, observations=observations(), public_center=center,
            prediction_epochs=epochs, public_center_available_at=epochs,
            transitions=[sparse.eye(2) for _ in epochs],
            transition_latest_input_at=np.array([1, 2, 3, 5, 5, 6]),
            physical_forcing=np.ones_like(center), physical_forcing_latest_input_at=epochs,
            previous_public_center=np.array([9., 10.]), calibration_half_open=(1, 4),
            evaluation_half_open=(4, 7), calibration_truth={
                "world-0": center[:3], "world-1": center[:3]},
            calibration_truth_available_at={"world-0": np.arange(1, 4),
                "world-1": np.arange(1, 4)}, calibration_split_id="cal",
            evaluation_split_id="eval", public_model_id="public", sensor_model_id="sensor",
            source_hashes={"fixture": "0" * 64}, source_roles={
                "citizen_observations": "exact_selected_citizen_channel",
                "public_center": "frozen_public_predictor",
                "transition": "declared_physical_operator",
                "physical_forcing": "declared_public_exogenous",
                "calibration_truth": "closed_calibration_labels",
            }, calibration_pair_context={f"n-{e}-{u}": {
                "calibration_world_id": f"world-{u}",
                "calibration_time_block_id": f"block-{e // 2}"}
                for e in range(1, 4) for u in range(2)}, development_seed_id="bad")


def test_hash_tampering_fails_closed(tmp_path):
    directory = tmp_path / "bundle"
    produce(directory)
    with (directory / "citizen_evaluation_records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_producer_bundle(directory, expected_manifest_sha256=hashlib.sha256(
            (directory / "producer_manifest.json").read_bytes()).hexdigest())


def test_manifest_digest_is_external_and_mandatory(tmp_path):
    directory = tmp_path / "bundle"
    produce(directory)
    manifest = directory / "producer_manifest.json"
    registered = hashlib.sha256(manifest.read_bytes()).hexdigest()
    value = json.loads(manifest.read_text())
    value["source_hashes"] = {"tampered": "claim"}
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="externally registered"):
        load_producer_bundle(directory, expected_manifest_sha256=registered)


def test_synthetic_field_mechanism_preserves_rng_and_excludes_process_noise():
    first = synthetic_field(3, 8, np.random.default_rng(42))
    second, mechanism = synthetic_field_with_mechanism(3, 8, np.random.default_rng(42))
    assert np.array_equal(first, second)
    residual = second[1:] - np.stack([mechanism.transition @ x for x in second[:-1]])
    process_innovation = residual - mechanism.exogenous_forcing[1:]
    assert np.any(np.abs(process_innovation) > 0)
    assert mechanism.source == "synthetic-public-plume-forcing-v2"

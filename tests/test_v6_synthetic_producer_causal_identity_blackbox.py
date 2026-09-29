"""Independent black-box contract tests for the synthetic v2 input producer.

These tests create only tiny temporary input bundles. They never invoke an estimator
runner or read a scientific outcome namespace.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from scipy import sparse

from airproof.field import synthetic_field_with_mechanism
from airproof.records import Observation
from airproof.v6_mechanistic_input_producer import (
    load_producer_bundle,
    produce_provenance_complete_bundle,
)
from airproof.v6_residual_dynamics import public_residual_forcing
from scripts.run_v6_covariance_forcing_development_v2 import planned_runs

SOURCE_ROLES = {
    "citizen_observations": "exact_selected_citizen_channel",
    "public_center": "frozen_public_predictor",
    "transition": "declared_physical_operator",
    "physical_forcing": "declared_public_exogenous",
    "calibration_truth": "closed_calibration_labels",
}


class _RecordingRng:
    """Supply deterministic draws while retaining calls outside the producer API."""

    def __init__(self) -> None:
        self.uniform_calls = 0
        self.normal_draws: list[np.ndarray] = []

    def uniform(self, low, high):
        self.uniform_calls += 1
        return 0.0 if self.uniform_calls == 1 else 14.0

    def normal(self, location, scale, size):
        count = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        if not self.normal_draws:
            draw = np.linspace(-30.0, 3.0, count)
        else:
            sign = -1.0 if len(self.normal_draws) % 2 else 1.0
            draw = np.full(count, sign * 40.0)
        draw = draw.reshape(size) if isinstance(size, tuple) else draw
        self.normal_draws.append(np.asarray(draw, dtype=float).copy())
        return draw


def _observations(epochs: np.ndarray) -> tuple[Observation, ...]:
    rows = []
    for epoch in epochs:
        for user in range(4):
            rows.append(Observation(
                user_id=user,
                epoch=int(epoch),
                cell=user % 2,
                group=user % 2,
                value=float(8 + epoch + user / 10),
                sigma=1.0,
                quality=0.9,
                size_bytes=512,
                nullifier=f"blackbox-{epoch}-{user}",
                direct_arrival=int(epoch),
                relay_arrival=int(epoch + user % 2),
            ))
    return tuple(rows)


def _produce_bundle(path):
    epochs = np.arange(1, 7)
    center = np.column_stack((10 + epochs, 11 + epochs)).astype(float)
    transition = sparse.csr_matrix([[0.8, 0.1], [0.2, 0.7]])
    physical = np.column_stack((1 + epochs / 10, 2 + epochs / 10))
    manifest = produce_provenance_complete_bundle(
        path,
        observations=_observations(epochs),
        public_center=center,
        prediction_epochs=epochs,
        public_center_available_at=epochs,
        transitions=[transition] * len(epochs),
        transition_latest_input_at=epochs,
        physical_forcing=physical,
        physical_forcing_latest_input_at=epochs,
        previous_public_center=np.array([10.0, 11.0]),
        calibration_half_open=(1, 4),
        evaluation_half_open=(4, 7),
        calibration_truth={f"world-{user}": center[:3] + np.array([[0.5, -0.5]])
                           for user in range(4)},
        calibration_truth_available_at={f"world-{user}": np.array([1, 2, 3])
                                        for user in range(4)},
        calibration_split_id="blackbox-closed-calibration",
        evaluation_split_id="blackbox-exposed-evaluation",
        public_model_id="blackbox-public-v2",
        sensor_model_id="blackbox-sensor-v2",
        source_hashes={"blackbox-source": "1" * 64},
        source_roles=SOURCE_ROLES,
        calibration_pair_context={f"blackbox-{epoch}-{user}": {
            "calibration_world_id": f"world-{user}",
            "calibration_time_block_id": f"block-{epoch // 2}"}
            for epoch in range(1, 4) for user in range(4)},
        development_seed_id="blackbox-seed",
        cross_fit_folds=2,
    )
    manifest_path = path / "producer_manifest.json"
    return manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_epoch_zero_is_initialization_not_a_transition_row(tmp_path):
    rng = _RecordingRng()
    baseline = 12.0
    truth, mechanism = synthetic_field_with_mechanism(
        2, 5, rng, process_noise=1.0, baseline=baseline)

    plume_zero = (mechanism.exogenous_forcing[0] - 0.10 * baseline) / 0.22
    raw_zero = baseline + rng.normal_draws[0] + plume_zero
    np.testing.assert_allclose(truth[0], np.maximum(raw_zero, 0.0), atol=1e-12)

    # A bundle row is a physical transition row and therefore cannot claim epoch 0.
    epochs = np.arange(0, 6)
    center = np.column_stack((10 + epochs, 11 + epochs)).astype(float)
    with pytest.raises(ValueError, match="epoch|initial|boundary"):
        produce_provenance_complete_bundle(
            tmp_path / "invalid-epoch-zero-bundle",
            observations=_observations(epochs),
            public_center=center,
            prediction_epochs=epochs,
            public_center_available_at=epochs,
            transitions=[sparse.eye(2)] * len(epochs),
            transition_latest_input_at=epochs,
            physical_forcing=np.ones_like(center),
            physical_forcing_latest_input_at=epochs,
            previous_public_center=np.array([9.0, 10.0]),
            calibration_half_open=(0, 3),
            evaluation_half_open=(3, 6),
            calibration_truth={f"world-{user}": center[:3] for user in range(4)},
            calibration_truth_available_at={f"world-{user}": np.arange(3)
                                             for user in range(4)},
            calibration_split_id="cal",
            evaluation_split_id="eval",
            public_model_id="public",
            sensor_model_id="sensor",
            source_hashes={"fixture": "0" * 64},
            source_roles=SOURCE_ROLES,
            calibration_pair_context={f"blackbox-{epoch}-{user}": {
                "calibration_world_id": f"world-{user}",
                "calibration_time_block_id": f"block-{epoch // 2}"}
                for epoch in range(3) for user in range(4)},
            development_seed_id="invalid",
        )


def test_post_initialization_physical_and_residual_identities_include_projection():
    rng = _RecordingRng()
    truth, mechanism = synthetic_field_with_mechanism(
        2, 6, rng, process_noise=1.0, baseline=12.0)
    public = np.arange(truth.size, dtype=float).reshape(truth.shape) / 7 + 5

    for epoch in range(1, len(truth)):
        innovation = rng.normal_draws[epoch]
        raw = (mechanism.transition @ truth[epoch - 1]
               + mechanism.exogenous_forcing[epoch] + innovation)
        projection_correction = np.maximum(raw, 0.0) - raw
        physical_noise = innovation + projection_correction
        np.testing.assert_allclose(
            truth[epoch],
            mechanism.transition @ truth[epoch - 1]
            + mechanism.exogenous_forcing[epoch] + physical_noise,
            atol=1e-12,
        )
        residual = truth[epoch] - public[epoch]
        residual_previous = truth[epoch - 1] - public[epoch - 1]
        residual_forcing = (mechanism.exogenous_forcing[epoch]
                            + mechanism.transition @ public[epoch - 1]
                            - public[epoch])
        np.testing.assert_allclose(
            residual,
            mechanism.transition @ residual_previous + residual_forcing + physical_noise,
            atol=1e-12,
        )


def test_loaded_bundle_has_epoch_scoped_causal_forcing_and_no_hidden_state(tmp_path):
    directory = tmp_path / "producer-inputs"
    _, manifest_digest = _produce_bundle(directory)
    loaded = load_producer_bundle(directory, expected_manifest_sha256=manifest_digest)

    with np.load(directory / "mechanistic_arrays.npz", allow_pickle=False) as arrays:
        epochs = arrays["prediction_epochs"]
        assert np.all(epochs >= 1)
        np.testing.assert_array_less(arrays["physical_forcing_latest_input_at"], epochs + 1)
        expected = public_residual_forcing(
            arrays["public_center"], loaded["transitions"], arrays["physical_forcing"],
            previous_public=arrays["previous_public_center"],
        )
        np.testing.assert_array_equal(loaded["residual_forcing"], expected)

        forbidden = {
            "truth", "evaluation_truth", "process_innovation", "innovation",
            "preprojection_state", "field_rng", "rng_state", "field_seed",
            "phase", "amplitude", "attack_label", "event_mask",
        }
        assert forbidden.isdisjoint(arrays.files)

    public_mechanism = synthetic_field_with_mechanism(
        2, 4, np.random.default_rng(4))[1]
    public_names = set(vars(public_mechanism))
    assert {"transition", "exogenous_forcing", "source"} <= public_names
    assert not any(token in name for name in public_names
                   for token in ("truth", "noise", "innovation", "rng", "seed", "phase", "amplitude"))

    evaluation_text = (directory / "citizen_evaluation_records.jsonl").read_text(encoding="utf-8")
    assert all(token not in evaluation_text for token in
               ("truth", "attack", "event_mask", "field_rng", "process_innovation"))


def test_one_forcing_payload_is_shared_by_every_registered_control(tmp_path):
    directory = tmp_path / "producer-inputs"
    manifest, manifest_digest = _produce_bundle(directory)
    loaded = load_producer_bundle(directory, expected_manifest_sha256=manifest_digest)
    registration = {
        "same_information_controls": {
            "names": ["PUBLIC", "SQ", "HUBER"],
            "required_fingerprint_fields": ["physical_forcing", "residual_forcing"],
        },
        "scenarios": ["clean"],
        "candidates": [{"id": "candidate"}],
    }
    plan = planned_runs(registration, [1])
    assert plan[0]["methods"] == ["candidate", "PUBLIC", "SQ", "HUBER"]
    assert {key for key in loaded if "forcing" in key} == {
        "physical_forcing", "residual_forcing"}
    assert manifest["producers"]["physical_forcing"]["output_fields"] == [
        "physical_forcing", "residual_forcing"]


@pytest.mark.parametrize("payload", ["mechanistic_arrays.npz", "causality_audit.json"])
def test_payload_tampering_fails_closed(tmp_path, payload):
    directory = tmp_path / payload.replace(".", "-")
    _, manifest_digest = _produce_bundle(directory)
    path = directory / payload
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_producer_bundle(directory, expected_manifest_sha256=manifest_digest)


def test_manifest_tampering_fails_against_external_digest(tmp_path):
    directory = tmp_path / "manifest-tamper"
    _, manifest_digest = _produce_bundle(directory)
    path = directory / "producer_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source_hashes"]["blackbox-source"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.*SHA-256|manifest hash"):
        load_producer_bundle(directory, expected_manifest_sha256=manifest_digest)


def test_manifest_digest_is_required(tmp_path):
    directory = tmp_path / "missing-external-digest"
    _produce_bundle(directory)
    with pytest.raises(TypeError):
        load_producer_bundle(directory)

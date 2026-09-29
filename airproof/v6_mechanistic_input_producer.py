"""Hash-bound causal inputs for the v6 covariance-by-forcing development family.

This module is deliberately an input producer, not an experiment runner.  Scoring
truth is accepted only for the closed calibration partition and is never written to
evaluation records or prediction arrays.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .records import Observation, canonical_json
from .v6_mechanistic_provenance import assert_causal_availability, assert_producer_manifest
from .v6_residual_dynamics import public_residual_forcing

SCHEMA_VERSION = 2
NAMESPACE_ROLE = "exposed-development-producer-inputs-only"
DEFAULT_NAMESPACE = "innovation_covariance_forcing_development_v2"
PAYLOAD_FILES = (
    "citizen_calibration_records.jsonl",
    "citizen_evaluation_records.jsonl",
    "arrival_maps.json",
    "paired_errors_calibration.jsonl",
    "calibration_statistics.json",
    "mechanistic_arrays.npz",
    "transition_sequence.json",
    "causality_audit.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + b"\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _observation_row(item: Observation, split_id: str) -> dict[str, Any]:
    # `corrupted` is a scoring-side attack label and must never cross this boundary.
    return {
        "schema_version": SCHEMA_VERSION,
        "split_id": split_id,
        "user_id": int(item.user_id),
        "acquisition_epoch": int(item.epoch),
        "cell": int(item.cell),
        "group": int(item.group),
        "exact_observation_value": float(item.value),
        "sigma": float(item.sigma),
        "quality": float(item.quality),
        "size_bytes": int(item.size_bytes),
        "nullifier": str(item.nullifier),
        "source_class": str(item.source_class),
    }


def _validate_observations(observations: Sequence[Observation], epoch_to_index: Mapping[int, int], cells: int) -> None:
    nullifiers: set[str] = set()
    for item in observations:
        if not isinstance(item, Observation) or item.source_class != "citizen":
            raise ValueError("exact authenticated citizen Observation records required")
        if (item.epoch not in epoch_to_index or not 0 <= item.cell < cells
                or item.nullifier in nullifiers or not item.nullifier
                or not np.isfinite((item.value, item.sigma, item.quality)).all()
                or item.sigma <= 0 or not 0 < item.quality <= 1
                or item.direct_arrival is not None and item.direct_arrival < item.epoch
                or item.relay_arrival is not None and item.relay_arrival < item.epoch):
            raise ValueError("invalid, duplicate, or noncausal citizen record")
        nullifiers.add(item.nullifier)


def _covariance_summary(sensor_error: np.ndarray, public_error: np.ndarray) -> dict[str, Any]:
    if len(sensor_error) < 2:
        raise ValueError("at least two paired calibration errors required")
    matrix = np.cov(np.column_stack((sensor_error, public_error)), rowvar=False, ddof=1)
    matrix = np.asarray(matrix, float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    sensor_variance = float(matrix[0, 0])
    public_variance = float(matrix[1, 1])
    covariance = float(matrix[0, 1])
    innovation = sensor_variance + public_variance - 2 * covariance
    if not np.isfinite(matrix).all() or eigenvalues.min() < -1e-10 or innovation < -1e-10:
        raise ValueError("paired-error covariance is not positive semidefinite")
    return {
        "pair_count": len(sensor_error),
        "sensor_error_variance": sensor_variance,
        "public_error_variance": public_variance,
        "sensor_public_error_covariance": covariance,
        "innovation_variance": float(max(innovation, 0.0)),
        "covariance_matrix_sensor_public": matrix.tolist(),
        "psd_eigenvalues": eigenvalues.tolist(),
        "psd_audit_pass": bool(eigenvalues.min() >= -1e-10),
    }


def produce_provenance_complete_bundle(
    output_dir: str | Path,
    *,
    observations: Sequence[Observation],
    public_center: np.ndarray,
    prediction_epochs: Sequence[int],
    public_center_available_at: Sequence[int],
    transitions: Sequence[sparse.spmatrix | np.ndarray],
    transition_latest_input_at: Sequence[int],
    physical_forcing: np.ndarray,
    physical_forcing_latest_input_at: Sequence[int],
    previous_public_center: np.ndarray,
    calibration_half_open: tuple[int, int],
    evaluation_half_open: tuple[int, int],
    calibration_truth: Mapping[str, np.ndarray],
    calibration_truth_available_at: Mapping[str, Sequence[int]],
    calibration_split_id: str,
    evaluation_split_id: str,
    public_model_id: str,
    sensor_model_id: str,
    source_hashes: Mapping[str, str],
    source_roles: Mapping[str, str],
    calibration_pair_context: Mapping[str, Mapping[str, str]],
    development_seed_id: str,
    cross_fit_folds: int = 2,
    producer_namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    """Write an immutable input bundle without evaluating any estimator outcome."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("producer namespace already exists; overwrite is prohibited")
    required_roles = {
        "citizen_observations": "exact_selected_citizen_channel",
        "public_center": "frozen_public_predictor",
        "transition": "declared_physical_operator",
        "physical_forcing": "declared_public_exogenous",
        "calibration_truth": "closed_calibration_labels",
    }
    valid_hashes = (isinstance(source_hashes, Mapping) and bool(source_hashes)
        and all(isinstance(key, str) and key and isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value.lower())
                for key, value in source_hashes.items()))
    if (producer_namespace != DEFAULT_NAMESPACE or not all(
            (calibration_split_id, evaluation_split_id, public_model_id, sensor_model_id))
            or not valid_hashes or dict(source_roles) != required_roles):
        raise ValueError("fixed development-v2 namespace and complete producer identities required")
    epochs = np.asarray(prediction_epochs, dtype=int)
    center = np.asarray(public_center, dtype=float)
    if (epochs.ndim != 1 or center.ndim != 2 or len(epochs) != len(center)
            or len(set(map(int, epochs))) != len(epochs) or epochs.min() < 1
            or np.any(np.diff(epochs) <= 0)
            or not np.isfinite(center).all() or (center < 0).any()):
        raise ValueError("ordered unique epochs and a finite nonnegative public center required")
    epoch_to_index = {int(epoch): index for index, epoch in enumerate(epochs)}
    calibration_indices = np.flatnonzero((epochs >= calibration_half_open[0]) & (epochs < calibration_half_open[1]))
    evaluation_indices = np.flatnonzero((epochs >= evaluation_half_open[0]) & (epochs < evaluation_half_open[1]))
    if (not len(calibration_indices) or not len(evaluation_indices)
            or calibration_half_open[1] > evaluation_half_open[0]
            or set(calibration_indices) & set(evaluation_indices)):
        raise ValueError("nonempty disjoint calibration and evaluation partitions required")
    calibration_epochs = epochs[calibration_indices]
    if (not isinstance(calibration_truth, Mapping) or not calibration_truth
            or set(calibration_truth) != set(calibration_truth_available_at)):
        raise ValueError("world-keyed calibration truth and availability required")
    truths, truth_availability = {}, {}
    for world_id, values in calibration_truth.items():
        world_truth = np.asarray(values, float)
        world_available = np.asarray(calibration_truth_available_at[world_id], int)
        if (not world_id or world_truth.shape != (len(calibration_indices), center.shape[1])
                or world_available.shape != (len(calibration_indices),)
                or not np.isfinite(world_truth).all()
                or np.any(world_available < calibration_epochs)
                or np.any(world_available >= evaluation_half_open[0])):
            raise ValueError("truth labels must cover each closed calibration world only")
        truths[world_id] = world_truth
        truth_availability[world_id] = world_available
    if cross_fit_folds < 2 or len(calibration_indices) < cross_fit_folds:
        raise ValueError("at least two nonempty fixed cross-fit folds required")

    observations = tuple(observations)
    _validate_observations(observations, epoch_to_index, center.shape[1])
    by_partition: dict[str, list[Observation]] = {"calibration": [], "evaluation": []}
    for item in observations:
        if calibration_half_open[0] <= item.epoch < calibration_half_open[1]:
            by_partition["calibration"].append(item)
        elif evaluation_half_open[0] <= item.epoch < evaluation_half_open[1]:
            by_partition["evaluation"].append(item)
        else:
            raise ValueError("citizen record lies outside the declared producer partitions")
    if not by_partition["calibration"] or not by_partition["evaluation"]:
        raise ValueError("both partitions require exact citizen records")

    operators = tuple(sparse.csr_matrix(op) for op in transitions)
    physical = np.asarray(physical_forcing, float)
    previous = np.asarray(previous_public_center, float)
    public_available = np.asarray(public_center_available_at, int)
    transition_available = np.asarray(transition_latest_input_at, int)
    forcing_available = np.asarray(physical_forcing_latest_input_at, int)
    if (len(operators) != len(epochs) or any(op.shape != (center.shape[1], center.shape[1])
            or not np.isfinite(op.data).all() for op in operators)
            or physical.shape != center.shape or previous.shape != (center.shape[1],)
            or any(x.shape != epochs.shape for x in (public_available, transition_available, forcing_available))
            or not all(np.isfinite(x).all() for x in (physical, previous))
            or not development_seed_id):
        raise ValueError("complete finite transition, forcing, and availability arrays required")
    for latest in (public_available, transition_available, forcing_available):
        assert_causal_availability(epochs, latest)
    residual = public_residual_forcing(center, operators, physical,
                                       previous_public=previous)

    calibration_epoch_offsets = {int(epochs[index]): offset
                                 for offset, index in enumerate(calibration_indices)}
    paired_rows: list[dict[str, Any]] = []
    sensor_errors: list[float] = []
    public_errors: list[float] = []
    for item in sorted(by_partition["calibration"], key=lambda x: (x.epoch, x.cell, x.nullifier)):
        context = calibration_pair_context.get(item.nullifier)
        if (not context or not context.get("calibration_world_id")
                or not context.get("calibration_time_block_id")):
            raise ValueError("every calibration pair requires world and time-block identity")
        world_id = context["calibration_world_id"]
        if world_id not in truths:
            raise ValueError("calibration pair references an undeclared truth world")
        offset = calibration_epoch_offsets[item.epoch]
        label = truths[world_id][offset]
        label_available_at = int(truth_availability[world_id][offset])
        if not 0 <= item.cell < center.shape[1] or label_available_at > calibration_half_open[1]:
            raise ValueError("calibration pair is out of bounds or label closes after calibration")
        index = epoch_to_index[item.epoch]
        sensor_error = float(item.value - label[item.cell])
        public_error = float(center[index, item.cell] - label[item.cell])
        unit = f'{context["calibration_world_id"]}:{context["calibration_time_block_id"]}'
        pair_id = hashlib.sha256(
            canonical_json([calibration_split_id, item.nullifier, item.epoch, item.cell])
        ).hexdigest()
        fold = int(hashlib.sha256(unit.encode()).hexdigest()[:16], 16) % cross_fit_folds
        paired_rows.append({
            "schema_version": SCHEMA_VERSION, "pair_id": pair_id,
            "closed_calibration_split": calibration_split_id, "cross_fit_fold": fold,
            "calibration_world_id": context["calibration_world_id"],
            "calibration_time_block_id": context["calibration_time_block_id"],
            "nullifier": item.nullifier, "user_id": int(item.user_id),
            "acquisition_epoch": int(item.epoch), "arrival_epoch": item.relay_arrival,
            "cell": int(item.cell), "exact_observation_value": float(item.value),
            "paired_truth_label": float(label[item.cell]),
            "truth_label_available_at": label_available_at,
            "public_prediction": float(center[index, item.cell]),
            "public_prediction_available_at": int(public_available[index]),
            "paired_sensor_error": sensor_error, "paired_public_error": public_error,
            "sensor_model_id": sensor_model_id, "public_model_id": public_model_id,
        })
        sensor_errors.append(sensor_error)
        public_errors.append(public_error)
    sensor_array, public_array = np.asarray(sensor_errors), np.asarray(public_errors)
    full_summary = _covariance_summary(sensor_array, public_array)
    fold_summaries = {}
    folds = np.array([row["cross_fit_fold"] for row in paired_rows])
    unit_folds: dict[tuple[str, str], set[int]] = {}
    for row in paired_rows:
        unit_folds.setdefault((row["calibration_world_id"], row["calibration_time_block_id"]), set()).add(row["cross_fit_fold"])
    if any(len(value) != 1 for value in unit_folds.values()):
        raise ValueError("world/time-block fold leakage")
    for fold in range(cross_fit_folds):
        training = folds != fold
        if training.sum() < 2 or not np.any(~training):
            raise ValueError("each cross-fit fold needs held rows and at least two training pairs")
        fold_summaries[str(fold)] = _covariance_summary(sensor_array[training], public_array[training])
    statistics = {
        "schema_version": SCHEMA_VERSION,
        "calibration_split_id": calibration_split_id,
        "evaluation_split_id": evaluation_split_id,
        "calibration_only": True,
        "full_calibration": full_summary,
        "cross_fit_training_excluding_held_fold": fold_summaries,
    }

    destination.mkdir(parents=True)
    _jsonl(destination / PAYLOAD_FILES[0], [_observation_row(x, calibration_split_id) for x in by_partition["calibration"]])
    _jsonl(destination / PAYLOAD_FILES[1], [_observation_row(x, evaluation_split_id) for x in by_partition["evaluation"]])
    arrivals = {
        "schema_version": SCHEMA_VERSION,
        "direct": {x.nullifier: x.direct_arrival for x in observations},
        "relay": {x.nullifier: x.relay_arrival for x in observations},
    }
    (destination / PAYLOAD_FILES[2]).write_bytes(canonical_json(arrivals) + b"\n")
    _jsonl(destination / PAYLOAD_FILES[3], paired_rows)
    (destination / PAYLOAD_FILES[4]).write_bytes(canonical_json(statistics) + b"\n")
    np.savez_compressed(destination / "mechanistic_arrays.npz", public_center=center,
        prediction_epochs=epochs, public_center_available_at=public_available,
        transition_latest_input_at=transition_available,
        physical_forcing=physical, physical_forcing_latest_input_at=forcing_available,
        residual_forcing=residual, previous_public_center=previous)
    operator_dir = destination / "transition_operators"
    operator_dir.mkdir()
    operator_ids, operator_hashes = [], {}
    unique: dict[str, str] = {}
    for operator in operators:
        fingerprint = hashlib.sha256(canonical_json({
            "shape": operator.shape, "indptr": operator.indptr.tolist(),
            "indices": operator.indices.tolist(), "data": operator.data.tolist()})).hexdigest()
        operator_id = unique.get(fingerprint)
        if operator_id is None:
            operator_id = f"operator_{len(unique):04d}"
            unique[fingerprint] = operator_id
            path = operator_dir / f"{operator_id}.npz"
            sparse.save_npz(path, operator, compressed=True)
            operator_hashes[f"transition_operators/{path.name}"] = _sha256(path)
        operator_ids.append(operator_id)
    sequence = {"schema_version": SCHEMA_VERSION, "operator_ids": operator_ids,
                "unique_operator_count": len(unique)}
    (destination / "transition_sequence.json").write_bytes(canonical_json(sequence) + b"\n")
    audit = {
        "schema_version": SCHEMA_VERSION, "pass": True,
        "prediction_epochs": epochs.tolist(),
        "latest_input_epochs": {
            "public_center": public_available.tolist(),
            "transition_operator": transition_available.tolist(),
            "physical_forcing": forcing_available.tolist(),
        },
        "all_prediction_inputs_available_by_prediction": True,
        "all_arrivals_at_or_after_acquisition": True,
        "all_calibration_labels_available_within_closed_calibration": True,
        "truth_scope": "paired calibration errors only",
        "evaluation_truth_exported": False,
        "attack_or_event_labels_exported": False,
        "residual_forcing_identity_verified": bool(np.allclose(
            residual, public_residual_forcing(center, operators, physical,
                                              previous_public=previous))),
    }
    (destination / "causality_audit.json").write_bytes(canonical_json(audit) + b"\n")
    payload_hashes = {name: _sha256(destination / name) for name in PAYLOAD_FILES}
    payload_hashes.update(operator_hashes)
    bundle_root = hashlib.sha256(canonical_json(payload_hashes)).hexdigest()
    producer_epochs = epochs.tolist()
    producer_latest = np.maximum.reduce((public_available, transition_available, forcing_available)).tolist()
    producers = {}
    for name, fields in {
        "public_error_variance": ["frozen_public_error_variance"],
        "sensor_error_variance": ["frozen_sensor_error_variance"],
        "error_covariance": ["frozen_sensor_public_error_covariance"],
        "transition": ["transition_operator"],
        "physical_forcing": ["physical_forcing", "residual_forcing"],
    }.items():
        producers[name] = {
            "producer_id": f"{producer_namespace}:{name}", "output_fields": fields,
            "prediction_epochs": producer_epochs, "latest_input_epochs": producer_latest,
            "source_hashes": dict(source_hashes), "uses_hidden_truth": False,
            "uses_attack_flags": False, "uses_calibration_labels": name in {
                "public_error_variance", "sensor_error_variance", "error_covariance"},
            "prediction_export": name not in {
                "public_error_variance", "sensor_error_variance", "error_covariance"},
        }
    manifest = {
        "schema_version": SCHEMA_VERSION, "namespace_role": NAMESPACE_ROLE,
        "producer_namespace": producer_namespace,
        "authorizes_outcome_execution": False,
        "contains_scoring_outcomes": False,
        "calibration_half_open": list(calibration_half_open),
        "evaluation_half_open": list(evaluation_half_open),
        "calibration_split_id": calibration_split_id,
        "evaluation_split_id": evaluation_split_id,
        "development_seed_id": development_seed_id,
        "public_model_id": public_model_id, "sensor_model_id": sensor_model_id,
        "source_hashes": dict(source_hashes), "source_roles": dict(source_roles),
        "payload_sha256": payload_hashes,
        "bundle_root_sha256": bundle_root, "producers": producers,
    }
    (destination / "producer_manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def _restore_observations(rows: Sequence[Mapping[str, Any]], arrivals: Mapping[str, Any]) -> tuple[Observation, ...]:
    restored = []
    for row in rows:
        forbidden = {"truth", "paired_truth_label", "attack", "attack_flag", "attack_label", "event_mask"}
        if forbidden & set(row):
            raise ValueError("scoring field found in prediction observation records")
        nullifier = row["nullifier"]
        restored.append(Observation(
            int(row["user_id"]), int(row["acquisition_epoch"]), int(row["cell"]),
            int(row["group"]), float(row["exact_observation_value"]), float(row["sigma"]),
            float(row["quality"]), int(row["size_bytes"]), str(nullifier),
            arrivals["direct"][nullifier], arrivals["relay"][nullifier], False,
            str(row["source_class"])))
    return tuple(restored)


def load_producer_bundle(bundle_dir: str | Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
    """Verify every byte against an externally registered manifest digest."""
    source = Path(bundle_dir)
    manifest_path = source / "producer_manifest.json"
    if (not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64
            or _sha256(manifest_path) != expected_manifest_sha256):
        raise ValueError("producer manifest does not match externally registered SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("namespace_role") != NAMESPACE_ROLE
            or manifest.get("producer_namespace") != DEFAULT_NAMESPACE
            or manifest.get("authorizes_outcome_execution") is not False
            or manifest.get("contains_scoring_outcomes") is not False):
        raise ValueError("wrong or unsafe producer namespace manifest")
    required_producers = {
        "public_error_variance", "sensor_error_variance", "error_covariance",
        "transition", "physical_forcing",
    }
    if set(manifest.get("producers", {})) != required_producers:
        raise ValueError("producer manifest inventory mismatch")
    for producer in manifest["producers"].values():
        assert_producer_manifest(producer)
    hashes = manifest.get("payload_sha256", {})
    if not set(PAYLOAD_FILES).issubset(hashes) or not any(
            name.startswith("transition_operators/") for name in hashes):
        raise ValueError("producer payload inventory mismatch")
    actual = {name: _sha256(source / name) for name in hashes}
    if actual != hashes or hashlib.sha256(canonical_json(actual)).hexdigest() != manifest.get("bundle_root_sha256"):
        raise ValueError("producer payload hash mismatch")
    audit = json.loads((source / "causality_audit.json").read_text(encoding="utf-8"))
    if audit.get("pass") is not True or audit.get("evaluation_truth_exported") is not False:
        raise ValueError("producer causality audit failed")
    arrivals = json.loads((source / "arrival_maps.json").read_text(encoding="utf-8"))
    evaluation_rows = _read_jsonl(source / "citizen_evaluation_records.jsonl")
    arrays = np.load(source / "mechanistic_arrays.npz", allow_pickle=False)
    epochs = arrays["prediction_epochs"]
    for latest_name in ("public_center_available_at", "transition_latest_input_at",
                        "physical_forcing_latest_input_at"):
        assert_causal_availability(epochs, arrays[latest_name])
    producer_latest = np.maximum.reduce((arrays["public_center_available_at"],
        arrays["transition_latest_input_at"], arrays["physical_forcing_latest_input_at"])).tolist()
    for producer in manifest.get("producers", {}).values():
        if (producer.get("prediction_epochs") != epochs.tolist()
                or producer.get("latest_input_epochs") != producer_latest
                or producer.get("source_hashes") != manifest.get("source_hashes")):
            raise ValueError("producer manifest clocks or source hashes do not match payload")
    sequence = json.loads((source / "transition_sequence.json").read_text(encoding="utf-8"))
    transitions = tuple(sparse.load_npz(source / "transition_operators" / f"{operator_id}.npz")
                        for operator_id in sequence["operator_ids"])
    recomputed = public_residual_forcing(arrays["public_center"], transitions,
        arrays["physical_forcing"], previous_public=arrays["previous_public_center"])
    if not np.array_equal(recomputed, arrays["residual_forcing"]):
        raise ValueError("residual forcing identity mismatch")
    return {
        "manifest": manifest,
        "evaluation_observations": _restore_observations(evaluation_rows, arrivals),
        "arrival_maps": {name: {item.nullifier: arrivals[name][item.nullifier]
                         for item in _restore_observations(evaluation_rows, arrivals)}
                         for name in ("direct", "relay")},
        "frozen_innovation_scales": json.loads((source / "calibration_statistics.json").read_text(encoding="utf-8"))["cross_fit_training_excluding_held_fold"],
        "public_center": arrays["public_center"].copy(),
        "prediction_epochs": epochs.copy(),
        "transitions": transitions,
        "physical_forcing": arrays["physical_forcing"].copy(),
        "residual_forcing": arrays["residual_forcing"].copy(),
    }


def load_calibration_audit_bundle(bundle_dir: str | Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
    """Privileged audit view; never pass this object to an estimator runner."""
    source = Path(bundle_dir)
    load_producer_bundle(source, expected_manifest_sha256=expected_manifest_sha256)
    manifest = json.loads((source / "producer_manifest.json").read_text(encoding="utf-8"))
    pairs = _read_jsonl(source / "paired_errors_calibration.jsonl")
    if any(row.get("closed_calibration_split") != manifest["calibration_split_id"] for row in pairs):
        raise ValueError("paired errors escaped the closed calibration split")
    units: dict[tuple[str, str], set[int]] = {}
    for row in pairs:
        units.setdefault((row["calibration_world_id"], row["calibration_time_block_id"]), set()).add(row["cross_fit_fold"])
    if any(len(folds) != 1 for folds in units.values()):
        raise ValueError("world/time-block fold leakage")
    return {"manifest": manifest, "paired_errors_calibration": pairs,
            "calibration_observations": _restore_observations(
                _read_jsonl(source / "citizen_calibration_records.jsonl"),
                json.loads((source / "arrival_maps.json").read_text(encoding="utf-8"))),
            "calibration_statistics": json.loads(
                (source / "calibration_statistics.json").read_text(encoding="utf-8"))}

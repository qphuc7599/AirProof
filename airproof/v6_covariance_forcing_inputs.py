"""Provenance-complete inputs for the registered v6 covariance/forcing campaign.

Calibration labels and development scoring truth live in separate capabilities.
Prediction loaders cannot return either.  The physical operator is stored once in
CSR form; a dense time-by-cell-by-cell transition tensor is never materialized.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .config import config_hash, load_config
from .experiment import _v4_public_components
from .field import grid_laplacian
from .records import Observation, canonical_json
from .simulator import generate_world
from .v5_experiment import cell_configuration
from .v6_experiment import integrated_transport
from .v6_mobility import coupled_public_trace
from .v6_residual_dynamics import public_residual_forcing

SCHEMA_VERSION = 1
REGISTRATION = "configs/v6/innovation_covariance_forcing_development_v2.json"
BASE_CONFIG = "reports/v4_primary/core_final_7600_7629_20260903/base_configuration.yaml"
PROTOCOL = "configs/v6/protocol.yaml"
PREDICTION_FORBIDDEN = frozenset({
    "truth", "calibration_truth", "evaluation_truth", "event_mask", "attack_label",
    "corrupted", "paired_sensor_error", "paired_public_error", "process_innovation",
    "field_rng_state", "plume_phase", "plume_amplitude",
})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json(dict(value)) + b"\n")


def load_registration(root: str | Path) -> dict[str, Any]:
    return json.loads((Path(root) / REGISTRATION).read_text(encoding="utf-8"))


def campaign_config(root: str | Path, cell: str) -> dict[str, Any]:
    root = Path(root)
    cfg = load_config(root / BASE_CONFIG)
    protocol = load_config(root / PROTOCOL)
    cfg["transport"] = dict(protocol["transport"])
    cfg["transport"].update(drain_epochs=24, useful_lag_epochs=6)
    cfg = cell_configuration(cfg, cell)
    registration = load_registration(root)
    scale = registration["scale"]
    observed = {
        "agents": int(cfg["world"]["agents"]),
        "grid_side": int(cfg["world"]["grid_side"]),
        "groups": int(cfg["world"]["groups"]),
        "acquisition_epochs": int(cfg["world"]["steps"]),
        "burn_in_epochs": int(cfg["world"]["burn_in_steps"]),
        "fixed_lag_epochs": int(cfg["twin"]["fixed_lag"]),
        "transport_drain_epochs": int(cfg["transport"]["drain_epochs"]),
        "epoch_hours": int(cfg["world"].get("epoch_hours", 1)),
    }
    if observed != scale:
        raise ValueError(f"registered scale differs from executable configuration: {observed}")
    return cfg


def expected_physical_operator(side: int) -> sparse.csr_matrix:
    return (0.90 * (sparse.eye(side * side, format="csr")
                    - 0.08 * grid_laplacian(side))).tocsr()


def save_csr(path: str | Path, operator: sparse.spmatrix) -> None:
    matrix = sparse.csr_matrix(operator, dtype=np.float64)
    np.savez_compressed(path, data=matrix.data, indices=matrix.indices,
                        indptr=matrix.indptr, shape=np.asarray(matrix.shape, dtype=np.int64))


def load_csr(path: str | Path) -> sparse.csr_matrix:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"data", "indices", "indptr", "shape"}:
            raise ValueError("CSR payload schema mismatch")
        shape = tuple(map(int, arrays["shape"]))
        matrix = sparse.csr_matrix((arrays["data"], arrays["indices"], arrays["indptr"]),
                                   shape=shape)
    if matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix.data).all():
        raise ValueError("finite square CSR physical operator required")
    return matrix


def _public_context(world, cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    public, operators, diagnostics = _v4_public_components(world, cfg)
    lag = int(cfg["twin"]["fixed_lag"])
    tail: list[np.ndarray] = []
    last = public[-1].copy()
    for _ in range(lag):
        last = last.copy() if operators is None else np.maximum(operators[-1] @ last, 0)
        tail.append(last.copy())
    extended = np.concatenate((public, np.asarray(tail))) if lag else public
    return extended, diagnostics


def _run_common_execution(world, cfg: Mapping[str, Any]):
    trace, _ = coupled_public_trace(cfg, world.seed, world.observations)
    transport, collector = integrated_transport(trace, world.observations, cfg)
    selected = tuple(sorted(collector.selected,
                            key=lambda item: (item.epoch, item.cell, item.user_id,
                                              item.nullifier)))
    arrivals = {item.nullifier: int(collector.selection_times[item.nullifier])
                for item in selected}
    if len(arrivals) != len(selected):
        raise AssertionError("selected nullifier identity is not unique")
    resource_violations = {
        key: value for key, value in transport.metrics.items()
        if key.endswith(("violation_count", "violations"))
    }
    if any(value for value in resource_violations.values() if isinstance(value, (int, float))):
        raise AssertionError(f"common execution resource violation: {resource_violations}")
    return selected, arrivals, trace, transport, collector


def _record_arrays(records: Iterable[Observation], arrivals: Mapping[str, int], *,
                   burn: int) -> dict[str, np.ndarray]:
    items = tuple(records)
    if not items:
        raise ValueError("at least one selected observation required")
    shifted_epoch = np.asarray([item.epoch - burn for item in items], dtype=np.int16)
    shifted_arrival = np.asarray([arrivals[item.nullifier] - burn for item in items],
                                 dtype=np.int16)
    if shifted_epoch.min() < 0 or np.any(shifted_arrival < shifted_epoch):
        raise ValueError("noncausal shifted observation/arrival")
    return {
        "user_id": np.asarray([item.user_id for item in items], dtype=np.int32),
        "epoch": shifted_epoch,
        "cell": np.asarray([item.cell for item in items], dtype=np.int16),
        "group": np.asarray([item.group for item in items], dtype=np.int8),
        "value": np.asarray([item.value for item in items], dtype=np.float64),
        "sigma": np.asarray([item.sigma for item in items], dtype=np.float32),
        "quality": np.asarray([item.quality for item in items], dtype=np.float32),
        "size_bytes": np.asarray([item.size_bytes for item in items], dtype=np.int16),
        "nullifier": np.asarray([item.nullifier for item in items], dtype="U64"),
        "arrival": shifted_arrival,
    }


def restore_records(path: str | Path) -> tuple[tuple[Observation, ...], dict[str, int]]:
    with np.load(path, allow_pickle=False) as arrays:
        required = {"user_id", "epoch", "cell", "group", "value", "sigma",
                    "quality", "size_bytes", "nullifier", "arrival"}
        if set(arrays.files) != required:
            raise ValueError("prediction observation schema mismatch")
        values = {name: arrays[name].copy() for name in arrays.files}
    count = len(values["epoch"])
    if any(len(value) != count for value in values.values()):
        raise ValueError("unaligned observation arrays")
    records = tuple(Observation(
        int(values["user_id"][i]), int(values["epoch"][i]), int(values["cell"][i]),
        int(values["group"][i]), float(values["value"][i]), float(values["sigma"][i]),
        float(values["quality"][i]), int(values["size_bytes"][i]),
        str(values["nullifier"][i]), int(values["arrival"][i]),
        int(values["arrival"][i]), False, "citizen") for i in range(count))
    arrival_map = {record.nullifier: int(values["arrival"][i])
                   for i, record in enumerate(records)}
    if len(arrival_map) != count:
        raise ValueError("duplicate nullifier in observation payload")
    return records, arrival_map


def _moments(sensor: np.ndarray, public: np.ndarray) -> dict[str, float | int]:
    if sensor.shape != public.shape or sensor.ndim != 1 or len(sensor) < 2:
        raise ValueError("aligned paired errors required")
    if not np.isfinite(sensor).all() or not np.isfinite(public).all():
        raise ValueError("finite paired errors required")
    return {"n": len(sensor), "sum_s": float(sensor.sum()),
            "sum_p": float(public.sum()), "sum_ss": float(sensor @ sensor),
            "sum_pp": float(public @ public), "sum_sp": float(sensor @ public)}


def combine_moments(rows: Iterable[Mapping[str, float | int]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {key: 0 for key in
        ("n", "sum_s", "sum_p", "sum_ss", "sum_pp", "sum_sp")}
    for row in rows:
        for key in result:
            result[key] += row[key]
    return result


def covariance_from_moments(row: Mapping[str, float | int], *, tolerance: float) -> dict[str, Any]:
    n = int(row["n"])
    if n < 2:
        raise ValueError("covariance requires at least two pairs")
    mean_s = float(row["sum_s"]) / n
    mean_p = float(row["sum_p"]) / n
    vs = (float(row["sum_ss"]) - n * mean_s * mean_s) / (n - 1)
    vp = (float(row["sum_pp"]) - n * mean_p * mean_p) / (n - 1)
    cps = (float(row["sum_sp"]) - n * mean_s * mean_p) / (n - 1)
    matrix = np.asarray([[vs, cps], [cps, vp]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.isfinite(matrix).all() or eigenvalues.min() < -tolerance:
        raise ValueError("paired covariance is not PSD")
    return {"pair_count": n, "sensor_variance": float(vs),
            "public_variance": float(vp), "sensor_public_covariance": float(cps),
            "innovation_independence_variance": float(vs + vp),
            "innovation_cross_covariance_variance": float(vs + vp - 2 * cps),
            "covariance_matrix": matrix.tolist(), "psd_eigenvalues": eigenvalues.tolist(),
            "psd_audit_pass": True}


def build_calibration_world(root: str | Path, stage: str | Path, seed: int) -> dict[str, Any]:
    root, stage = Path(root), Path(stage)
    cfg = campaign_config(root, "severe_clean")
    world = generate_world(cfg, int(seed))
    selected, arrivals, trace, transport, _ = _run_common_execution(world, cfg)
    burn, steps = int(cfg["world"]["burn_in_steps"]), int(cfg["world"]["steps"])
    selected = tuple(item for item in selected if burn <= item.epoch < steps)
    public, public_diagnostics = _public_context(world, cfg)
    sensor = np.asarray([item.value - world.truth[item.epoch, item.cell]
                         for item in selected], dtype=np.float64)
    public_error = np.asarray([public[item.epoch, item.cell]
                               - world.truth[item.epoch, item.cell]
                               for item in selected], dtype=np.float64)
    target = stage / "calibration" / str(seed)
    target.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(target / "pairs.npz", sensor_error=sensor,
                        public_error=public_error,
                        acquisition_epoch=np.asarray([x.epoch for x in selected], dtype=np.int16),
                        cell=np.asarray([x.cell for x in selected], dtype=np.int16),
                        user_id=np.asarray([x.user_id for x in selected], dtype=np.int32),
                        nullifier=np.asarray([x.nullifier for x in selected], dtype="U64"),
                        arrival_epoch=np.asarray([arrivals[x.nullifier] for x in selected], dtype=np.int16))
    moment = _moments(sensor, public_error)
    summary = {"schema_version": SCHEMA_VERSION, "role": "closed_calibration_world",
        "seed": int(seed), "cell": "severe_clean", "config_hash": config_hash(cfg),
        "trace_hash": trace.trace_hash, "pair_moments": moment,
        "selected_records": len(selected), "public_diagnostics": public_diagnostics,
        "transport_metrics_hash": hashlib.sha256(canonical_json({
            k: v for k, v in transport.metrics.items() if k != "local_decisions"})).hexdigest(),
        "truth_exported_to_prediction": False, "scientific_outcome_computed": False}
    write_json(target / "summary.json", summary)
    return summary


def _assert_scenario_identity(clean_records: Mapping[str, Observation],
                              scenario_records: Mapping[str, Observation]) -> None:
    if set(clean_records) != set(scenario_records):
        raise AssertionError("scenario changed candidate observation identity")
    for nullifier, clean in clean_records.items():
        other = scenario_records[nullifier]
        left = (clean.user_id, clean.epoch, clean.cell, clean.group, clean.sigma,
                clean.quality, clean.size_bytes, clean.nullifier)
        right = (other.user_id, other.epoch, other.cell, other.group, other.sigma,
                 other.quality, other.size_bytes, other.nullifier)
        if left != right:
            raise AssertionError("scenario changed non-payload observation metadata")


def build_development_world(root: str | Path, stage: str | Path, seed: int) -> dict[str, Any]:
    root, stage = Path(root), Path(stage)
    registration = load_registration(root)
    cells = registration["world_roles"]["development"]["cells"]
    clean_cfg = campaign_config(root, "severe_clean")
    clean_world = generate_world(clean_cfg, int(seed))
    selected, arrivals, trace, transport, _ = _run_common_execution(clean_world, clean_cfg)
    burn, steps = int(clean_cfg["world"]["burn_in_steps"]), int(clean_cfg["world"]["steps"])
    lag = int(clean_cfg["twin"]["fixed_lag"])
    selected = tuple(item for item in selected if burn <= item.epoch < steps)
    selected_ids = {item.nullifier for item in selected}
    public, public_diagnostics = _public_context(clean_world, clean_cfg)
    mechanism = clean_world.physical_mechanism
    if mechanism is None or len(mechanism.exogenous_forcing) < steps + lag:
        raise ValueError("world lacks the registered physical mechanism horizon")
    expected = expected_physical_operator(int(clean_cfg["world"]["grid_side"]))
    if ((mechanism.transition - expected).nnz
            and np.max(np.abs((mechanism.transition - expected).data)) > 1e-14):
        raise AssertionError("world physical operator differs from registered operator")
    prediction_epochs = np.arange(burn, steps + lag, dtype=np.int16)
    center = public[burn:steps + lag]
    physical = mechanism.exogenous_forcing[burn:steps + lag]
    residual = public_residual_forcing(center, [expected] * len(center), physical,
                                       previous_public=public[burn - 1])
    prediction_dir = stage / "prediction" / str(seed)
    scoring_dir = stage / "scoring" / str(seed)
    prediction_dir.mkdir(parents=True, exist_ok=False)
    scoring_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(prediction_dir / "context.npz", public_center=center,
        prediction_epochs=prediction_epochs,
        public_center_available_at=prediction_epochs,
        physical_forcing=physical,
        physical_forcing_available_at=prediction_epochs,
        residual_forcing=residual,
        previous_public_center=public[burn - 1])
    threshold = float(np.quantile(clean_world.truth[:burn], .95))
    truth = clean_world.truth[burn:steps]
    np.savez_compressed(scoring_dir / "scoring_only.npz", evaluation_truth=truth,
        event_mask=truth >= threshold, event_threshold=np.asarray(threshold),
        cell_groups=clean_world.cell_groups)
    clean_map = {item.nullifier: item for item in clean_world.observations
                 if item.nullifier in selected_ids}
    scenario_counts = {}
    for cell in cells:
        if cell == "severe_clean":
            scenario_world = clean_world
        else:
            scenario_world = generate_world(campaign_config(root, cell), int(seed))
        scenario_map = {item.nullifier: item for item in scenario_world.observations
                        if item.nullifier in selected_ids}
        _assert_scenario_identity(clean_map, scenario_map)
        ordered = tuple(scenario_map[item.nullifier] for item in selected)
        np.savez_compressed(prediction_dir / f"observations_{cell}.npz",
                            **_record_arrays(ordered, arrivals, burn=burn))
        scenario_counts[cell] = len(ordered)
    summary = {"schema_version": SCHEMA_VERSION, "role": "development_prediction_and_separate_scoring",
        "seed": int(seed), "cells": cells, "config_hash": config_hash(clean_cfg),
        "trace_hash": trace.trace_hash, "selected_records_per_cell": scenario_counts,
        "public_diagnostics": public_diagnostics,
        "context_rows": len(center), "scoring_rows": len(truth),
        "truth_in_prediction_files": False, "scoring_loaded_during_prediction": False,
        "transport_metrics_hash": hashlib.sha256(canonical_json({
            k: v for k, v in transport.metrics.items() if k != "local_decisions"})).hexdigest(),
        "scientific_outcome_computed": False}
    write_json(prediction_dir / "summary.json", summary)
    return summary


def build_global_calibration(root: str | Path, stage: str | Path) -> dict[str, Any]:
    root, stage = Path(root), Path(stage)
    registration = load_registration(root)
    seeds = registration["world_roles"]["calibration"]["seeds"]
    tolerance = float(registration["covariance_calibration"]["psd_tolerance"])
    minimum = int(registration["covariance_calibration"]["minimum_pairs_per_training_fold"])
    by_world = {}
    for seed in seeds:
        summary = json.loads((stage / "calibration" / str(seed) / "summary.json").read_text())
        by_world[int(seed)] = summary["pair_moments"]
    full = covariance_from_moments(combine_moments(by_world.values()), tolerance=tolerance)
    folds = []
    total_held = sum(int(row["n"]) for row in by_world.values())
    weighted_independence = weighted_cross = 0.0
    for held_seed in seeds:
        training = [row for seed, row in by_world.items() if seed != held_seed]
        fitted = covariance_from_moments(combine_moments(training), tolerance=tolerance)
        if fitted["pair_count"] < minimum:
            raise ValueError("world-level covariance fold lacks registered support")
        held_count = int(by_world[int(held_seed)]["n"])
        weight = held_count / total_held
        weighted_independence += weight * fitted["innovation_independence_variance"]
        weighted_cross += weight * fitted["innovation_cross_covariance_variance"]
        folds.append({"held_world": int(held_seed), "training_worlds": [
            int(seed) for seed in seeds if seed != held_seed], "held_pair_count": held_count,
            "weight": weight, "training_summary": fitted})
    floor = float(registration["covariance_calibration"]["variance_floor"])
    variances = {"independence": float(max(weighted_independence, floor)),
                 "world_cross_fitted_covariance": float(max(weighted_cross, floor))}
    result = {"schema_version": SCHEMA_VERSION, "role": "frozen_global_calibration",
        "calibration_worlds": list(map(int, seeds)), "development_worlds_read": False,
        "fold_unit": "world", "fold_count": len(folds), "full_summary": full,
        "world_cross_fitted_folds": folds, "innovation_variances": variances,
        "innovation_scales": {key: float(np.sqrt(value)) for key, value in variances.items()},
        "variance_floor": floor, "psd_tolerance": tolerance,
        "candidate_selection_performed": False, "scientific_outcome_computed": False}
    write_json(stage / "global_calibration.json", result)
    return result


def source_inventory(root: str | Path) -> dict[str, str]:
    root = Path(root)
    paths = sorted([*root.glob("airproof/*.py"),
                    root / REGISTRATION,
                    root / "configs/v6/innovation_covariance_forcing_producer_development_v2.json",
                    root / PROTOCOL, root / BASE_CONFIG,
                    root / "scripts/build_v6_covariance_forcing_v2_inputs.py",
                    root / "scripts/lock_v6_covariance_forcing_v2_inputs.py",
                    root / "scripts/audit_v6_covariance_forcing_v2_readiness.py",
                    root / "scripts/run_v6_covariance_forcing_development_v2.py"])
    return {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in paths if path.is_file()}


def finalize_campaign_bundle(root: str | Path, stage: str | Path,
                             calibration_summaries: list[Mapping[str, Any]],
                             development_summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    root, stage = Path(root), Path(stage)
    registration = load_registration(root)
    inventory = source_inventory(root)
    source_digest = hashlib.sha256(canonical_json(inventory)).hexdigest()
    snapshot = stage / "source_snapshot"
    for relative in inventory:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    files = sorted(path for path in stage.rglob("*") if path.is_file()
                   and path.name != "manifest.json")
    payloads = {str(path.relative_to(stage)).replace("\\", "/"): sha256_file(path)
                for path in files}
    manifest = {"schema_version": SCHEMA_VERSION,
        "role": "hash-bound inputs only; zero estimator efficacy outcomes",
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(root / REGISTRATION),
        "source_files": inventory, "source_digest": source_digest,
        "calibration_seeds": registration["world_roles"]["calibration"]["seeds"],
        "development_seeds": registration["world_roles"]["development"]["seeds"],
        "development_cells": registration["world_roles"]["development"]["cells"],
        "calibration_world_count": len(calibration_summaries),
        "development_world_count": len(development_summaries),
        "operator_storage": "single CSR",
        "payload_sha256": payloads,
        "payload_root_sha256": hashlib.sha256(canonical_json(payloads)).hexdigest(),
        "scientific_outcomes": 0,
        "protected_namespaces_opened": False}
    write_json(stage / "manifest.json", manifest)
    return manifest


def verify_campaign_bundle(root: str | Path, input_dir: str | Path,
                           *, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    root, input_dir = Path(root), Path(input_dir)
    manifest_path = input_dir / "manifest.json"
    if expected_manifest_sha256 is not None and sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("campaign input manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("registration_sha256") != sha256_file(root / REGISTRATION)
            or manifest.get("scientific_outcomes") != 0
            or manifest.get("protected_namespaces_opened") is not False
            or manifest.get("operator_storage") != "single CSR"):
        raise ValueError("unsafe or stale campaign input manifest")
    expected = manifest["payload_sha256"]
    actual = {relative: sha256_file(input_dir / relative) for relative in expected}
    if actual != expected or hashlib.sha256(canonical_json(actual)).hexdigest() != manifest["payload_root_sha256"]:
        raise ValueError("campaign input payload hash mismatch")
    return manifest


def load_prediction_view(root: str | Path, input_dir: str | Path, seed: int, cell: str,
                         *, expected_manifest_sha256: str) -> dict[str, Any]:
    """Load only prediction-capability files; scoring/calibration paths are inaccessible."""
    root, input_dir = Path(root), Path(input_dir)
    manifest = verify_campaign_bundle(root, input_dir,
                                      expected_manifest_sha256=expected_manifest_sha256)
    if int(seed) not in manifest["development_seeds"] or cell not in manifest["development_cells"]:
        raise ValueError("unregistered development seed/cell")
    context_path = input_dir / "prediction" / str(seed) / "context.npz"
    observation_path = input_dir / "prediction" / str(seed) / f"observations_{cell}.npz"
    operator = load_csr(input_dir / "mechanism" / "physical_operator.npz")
    with np.load(context_path, allow_pickle=False) as arrays:
        if PREDICTION_FORBIDDEN & set(arrays.files):
            raise ValueError("scoring field crossed prediction context boundary")
        context = {name: arrays[name].copy() for name in arrays.files}
    records, arrivals = restore_records(observation_path)
    view = {"public_center": context["public_center"],
        "prediction_epochs": context["prediction_epochs"], "operator": operator,
        "physical_forcing": context["physical_forcing"],
        "residual_forcing": context["residual_forcing"],
        "observations": records, "arrival_map": arrivals}
    if PREDICTION_FORBIDDEN & set(view):
        raise AssertionError("prediction view exposes forbidden field")
    return view


def load_scoring_view(input_dir: str | Path, seed: int) -> dict[str, np.ndarray]:
    path = Path(input_dir) / "scoring" / str(int(seed)) / "scoring_only.npz"
    with np.load(path, allow_pickle=False) as arrays:
        required = {"evaluation_truth", "event_mask", "event_threshold", "cell_groups"}
        if set(arrays.files) != required:
            raise ValueError("scoring-only schema mismatch")
        result = {name: arrays[name].copy() for name in arrays.files}
    if result["event_mask"].dtype.kind != "b" or not np.isfinite(
            result["evaluation_truth"]).all():
        raise ValueError("invalid scoring-only artifact")
    return result

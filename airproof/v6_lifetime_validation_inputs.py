"""Source-bound five-cell inputs for independent lifetime-estimator validation."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import config_hash, load_config
from .records import canonical_json
from .simulator import generate_world
from .v5_experiment import cell_configuration
from .v6_covariance_forcing_inputs import (
    _assert_scenario_identity,
    _public_context,
    _record_arrays,
    _run_common_execution,
    expected_physical_operator,
    load_csr,
    restore_records,
    save_csr,
    sha256_file,
    write_json,
)
from .v6_residual_dynamics import public_residual_forcing

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = "configs/v6/causal_lifetime_exposure_validation_v1.json"
BASE_CELLS = ("anchor_clean", "outage_clean", "severe_clean")
ATTACK_CELLS = ("severe_drift", "severe_hotspot")
PREDICTION_FORBIDDEN = frozenset({
    "truth", "calibration_truth", "evaluation_truth", "event_mask", "attack_label",
    "corrupted", "paired_sensor_error", "paired_public_error", "process_innovation",
    "field_rng_state", "plume_phase", "plume_amplitude",
})
CONTEXT_FIELDS = frozenset({
    "public_center", "prediction_epochs", "public_center_available_at",
    "physical_forcing", "physical_forcing_available_at", "residual_forcing",
    "previous_public_center",
})
SCORING_FIELDS = frozenset({
    "evaluation_truth", "event_mask", "event_threshold", "cell_groups",
})
OBSERVATION_FIELDS = frozenset({
    "user_id", "epoch", "cell", "group", "value", "sigma", "quality",
    "size_bytes", "nullifier", "arrival",
})
MANIFEST_FIELDS = frozenset({
    "schema_version", "role", "registration", "registration_sha256",
    "source_lock_sha256", "seeds", "cells", "context_base",
    "calibration_source", "calibration_source_sha256", "source_lineage",
    "expected_payload_count", "payload_sha256", "payload_root_sha256",
    "scientific_outcomes", "protected_namespaces_opened",
})


def load_registration(root: str | Path = ROOT) -> dict[str, Any]:
    return json.loads((Path(root) / REGISTRATION).read_text(encoding="utf-8"))


def _cell_base(cell: str) -> str:
    return "severe_clean" if cell in ATTACK_CELLS else cell


def validation_campaign_config(root: str | Path, cell: str) -> dict[str, Any]:
    root = Path(root)
    registration = load_registration(root)
    sources = registration["executable_sources"]
    cfg = load_config(root / sources["base_configuration"])
    protocol = load_config(root / sources["protocol"])
    cfg["transport"] = dict(protocol["transport"])
    cfg["transport"].update(drain_epochs=24, useful_lag_epochs=6)
    cfg = cell_configuration(cfg, cell)
    worlds = registration["worlds"]
    observed = [int(cfg["world"][key]) for key in
                ("agents", "grid_side", "steps", "burn_in_steps")]
    if (observed != [worlds["agents"], worlds["grid_side"],
                     worlds["acquisition_epochs"], worlds["burn_in_epochs"]]
            or int(cfg["twin"]["fixed_lag"]) != worlds["fixed_lag_epochs"]
            or int(cfg["transport"]["drain_epochs"])
            != worlds["transport_drain_epochs"]):
        raise ValueError("validation registration differs from executable scale")
    if config_hash(cfg) != sources["expected_config_hash"][cell]:
        raise ValueError("validation executable configuration hash changed")
    return cfg


def _source_lineage(root: Path, registration: dict[str, Any]) -> dict[str, dict[str, str]]:
    sources = registration["executable_sources"]
    paths = {
        "registration": REGISTRATION,
        "seed_registry": registration["worlds"]["seed_registry"],
        "development_registration": registration["nomination"]["development_registration"],
        "development_analysis": registration["nomination"]["development_analysis"],
        "v4_reuse_lock": registration["historical_reuse"]["v4_reuse_lock"],
        "base_configuration": sources["base_configuration"],
        "protocol": sources["protocol"],
        "innovation_calibration": registration["fixed_estimator"]["innovation_calibration"],
        "innovation_input_lock": (
            "reports/v6/innovation_covariance_forcing_development_v2/input_lock.json"
        ),
    }
    missing = [relative for relative in paths.values() if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"validation lineage source missing: {missing}")
    return {
        name: {"path": relative, "sha256": sha256_file(root / relative)}
        for name, relative in paths.items()
    }


def _expected_payloads(registration: dict[str, Any]) -> set[str]:
    expected = {"global_calibration.json", "mechanism/physical_operator.npz"}
    for seed in registration["worlds"]["seeds"]:
        prefix = f"prediction/{seed}"
        score = f"scoring/{seed}"
        expected.add(f"{prefix}/summary.json")
        for cell in registration["worlds"]["cells"]:
            expected.add(f"{prefix}/observations_{cell}.npz")
        for cell in registration["worlds"]["clean_cells"]:
            expected.add(f"{prefix}/context_{cell}.npz")
            expected.add(f"{score}/scoring_{cell}.npz")
    return expected


def _load_exact_npz(path: Path, required: frozenset[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != required:
            raise ValueError(f"payload schema mismatch: {path}")
        return {name: arrays[name].copy() for name in arrays.files}


def _validate_context(
    arrays: dict[str, np.ndarray], registration: dict[str, Any], operator,
) -> None:
    worlds = registration["worlds"]
    clock = registration["evaluation_clock"]
    side = int(worlds["grid_side"])
    cells = side * side
    start, stop = map(int, clock["prediction_half_open"])
    rows = stop - start
    if (arrays["public_center"].shape != (rows, cells)
            or arrays["physical_forcing"].shape != (rows, cells)
            or arrays["residual_forcing"].shape != (rows, cells)
            or arrays["previous_public_center"].shape != (cells,)):
        raise ValueError("validation context shape mismatch")
    epochs = np.arange(start, stop, dtype=arrays["prediction_epochs"].dtype)
    for name in ("prediction_epochs", "public_center_available_at",
                 "physical_forcing_available_at"):
        if arrays[name].ndim != 1 or not np.array_equal(arrays[name], epochs):
            raise ValueError("validation context clock mismatch")
    numeric = (arrays["public_center"], arrays["physical_forcing"],
               arrays["residual_forcing"], arrays["previous_public_center"])
    if not all(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all()
               for value in numeric):
        raise ValueError("validation context requires finite numeric arrays")
    recomputed = public_residual_forcing(
        arrays["public_center"], [operator] * rows, arrays["physical_forcing"],
        previous_public=arrays["previous_public_center"],
    )
    if not np.array_equal(arrays["residual_forcing"], recomputed):
        raise ValueError("validation residual-forcing identity mismatch")


def _validate_scoring(arrays: dict[str, np.ndarray], registration: dict[str, Any]) -> None:
    worlds = registration["worlds"]
    start, stop = map(int, registration["evaluation_clock"][
        "scored_acquisition_half_open"])
    shape = (stop - start, int(worlds["grid_side"]) ** 2)
    truth, event = arrays["evaluation_truth"], arrays["event_mask"]
    threshold, groups = arrays["event_threshold"], arrays["cell_groups"]
    if (truth.shape != shape or event.shape != shape or threshold.shape != ()
            or groups.shape != (shape[1],)):
        raise ValueError("validation scoring shape mismatch")
    if (not np.issubdtype(truth.dtype, np.floating) or not np.isfinite(truth).all()
            or event.dtype.kind != "b" or not np.issubdtype(groups.dtype, np.integer)
            or not math.isfinite(float(threshold))):
        raise ValueError("validation scoring dtype or finiteness mismatch")
    expected = truth >= float(threshold)
    if not np.array_equal(event, expected) or not bool(event.any()):
        raise ValueError("validation event mask or support mismatch")
    if groups.min(initial=0) < 0 or groups.max(initial=0) >= 4:
        raise ValueError("validation scoring group labels outside registered support")


def _validate_observations(
    path: Path, registration: dict[str, Any], *, expected_count: int | None = None,
) -> tuple[tuple, dict[str, int]]:
    arrays = _load_exact_npz(path, OBSERVATION_FIELDS)
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise ValueError("validation observation arrays are empty or unaligned")
    count = next(iter(lengths))
    if expected_count is not None and count != expected_count:
        raise ValueError("validation paired observation count mismatch")
    epochs, arrivals = arrays["epoch"], arrays["arrival"]
    scored = registration["worlds"]["acquisition_epochs"] - registration["worlds"][
        "burn_in_epochs"]
    drain = registration["worlds"]["transport_drain_epochs"]
    if (not np.issubdtype(epochs.dtype, np.integer)
            or not np.issubdtype(arrivals.dtype, np.integer)
            or np.any(epochs < 0) or np.any(epochs >= scored)
            or np.any(arrivals < epochs) or np.any(arrivals >= scored + drain)
            or np.any(arrays["cell"] < 0)
            or np.any(arrays["cell"] >= registration["worlds"]["grid_side"] ** 2)
            or not all(np.isfinite(arrays[name]).all()
                       for name in ("value", "sigma", "quality"))):
        raise ValueError("validation observation support or clock mismatch")
    return restore_records(path)


def _observation_identity(record) -> tuple:
    return (
        record.user_id, record.epoch, record.cell, record.group, record.sigma,
        record.quality, record.size_bytes, record.nullifier, record.direct_arrival,
        record.relay_arrival, record.source_class,
    )


def _clipped_perturbation_consistent(clean: float, attacked: float, delta: float,
                                     clip: float, *, tolerance: float = 1e-12) -> bool:
    def preimage(value: float) -> tuple[float, float]:
        if math.isclose(value, -clip, rel_tol=0.0, abs_tol=tolerance):
            return -math.inf, -clip + tolerance
        if math.isclose(value, clip, rel_tol=0.0, abs_tol=tolerance):
            return clip - tolerance, math.inf
        return value - tolerance, value + tolerance

    clean_low, clean_high = preimage(clean)
    attack_low, attack_high = preimage(attacked)
    attack_low -= delta
    attack_high -= delta
    return max(clean_low, attack_low) <= min(clean_high, attack_high)


def _assert_attack_world_pair(clean_world, attack_world, cfg: dict[str, Any]) -> dict:
    if (not np.array_equal(clean_world.truth, attack_world.truth)
            or not np.array_equal(clean_world.cell_groups, attack_world.cell_groups)
            or clean_world.physical_mechanism is None
            or attack_world.physical_mechanism is None
            or not np.array_equal(clean_world.physical_mechanism.exogenous_forcing,
                                  attack_world.physical_mechanism.exogenous_forcing)
            or (clean_world.physical_mechanism.transition
                != attack_world.physical_mechanism.transition).nnz):
        raise AssertionError("attack changed registered latent physical world")
    clean_reference = [(_observation_identity(item), item.value)
                       for item in clean_world.reference_observations]
    attack_reference = [(_observation_identity(item), item.value)
                        for item in attack_world.reference_observations]
    if clean_reference != attack_reference:
        raise AssertionError("attack changed public reference observations")
    clean = {item.nullifier: item for item in clean_world.observations}
    attack = {item.nullifier: item for item in attack_world.observations}
    if set(clean) != set(attack):
        raise AssertionError("attack changed candidate observation identity")
    attack_cfg = cfg["attack"]
    kind = str(attack_cfg["kind"])
    amplitude = float(attack_cfg["amplitude"])
    start = int(attack_cfg["start_epoch"])
    baseline = float(cfg["world"].get("baseline", 12.0))
    clip = float(cfg["privacy"]["clip"])
    changed = 0
    corrupted_users: set[int] = set()
    for nullifier, clean_item in clean.items():
        other = attack[nullifier]
        if _observation_identity(clean_item) != _observation_identity(other):
            raise AssertionError("attack changed non-payload observation identity")
        if other.corrupted:
            corrupted_users.add(other.user_id)
            if other.epoch < start:
                raise AssertionError("attack corruption precedes registered start")
            if kind == "adversarial_drift":
                direction = (-1.0 if clean_world.truth[other.epoch, other.cell] >= baseline
                             else 1.0)
                delta = direction * amplitude * (other.epoch - start + 1) / max(
                    1, int(cfg["world"]["steps"]) - start)
            elif kind == "hotspot_suppression":
                delta = -amplitude
            else:
                raise AssertionError("unregistered validation attack kind")
            changed += 1
        else:
            delta = 0.0
        if not _clipped_perturbation_consistent(
                clean_item.value, other.value, delta, clip):
            raise AssertionError("attack observation differs from registered perturbation")
    if changed < 1:
        raise AssertionError("registered attack has no perturbed observations")
    expected_users = round(int(cfg["world"]["agents"]) * float(attack_cfg["fraction"]))
    if len(corrupted_users) != expected_users:
        raise AssertionError("attack used a different registered attacker fraction")
    return {"candidate_records": len(clean), "perturbed_records": changed,
            "attacker_users": len(corrupted_users), "latent_world_equal": True,
            "exact_perturbation": True}


def _save_base(root: Path, stage: Path, seed: int, cell: str) -> tuple[dict, Any, tuple, dict]:
    cfg = validation_campaign_config(root, cell)
    world = generate_world(cfg, seed)
    selected, arrivals, trace, transport, _ = _run_common_execution(world, cfg)
    burn = int(cfg["world"]["burn_in_steps"])
    steps = int(cfg["world"]["steps"])
    lag = int(cfg["twin"]["fixed_lag"])
    selected = tuple(item for item in selected if burn <= item.epoch < steps)
    public, public_diagnostics = _public_context(world, cfg)
    mechanism = world.physical_mechanism
    expected = expected_physical_operator(int(cfg["world"]["grid_side"]))
    if (mechanism is None or len(mechanism.exogenous_forcing) < steps + lag
            or ((mechanism.transition - expected).nnz
                and np.max(np.abs((mechanism.transition - expected).data)) > 1e-14)):
        raise ValueError("validation world lacks the registered physical mechanism")
    prediction_epochs = np.arange(burn, steps + lag, dtype=np.int16)
    center = public[burn:steps + lag]
    physical = mechanism.exogenous_forcing[burn:steps + lag]
    residual = public_residual_forcing(center, [expected] * len(center), physical,
                                       previous_public=public[burn - 1])
    prediction_dir = stage / "prediction" / str(seed)
    scoring_dir = stage / "scoring" / str(seed)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    scoring_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_dir / f"context_{cell}.npz",
        public_center=center, prediction_epochs=prediction_epochs,
        public_center_available_at=prediction_epochs,
        physical_forcing=physical, physical_forcing_available_at=prediction_epochs,
        residual_forcing=residual, previous_public_center=public[burn - 1])
    threshold = float(np.quantile(world.truth[:burn], .95))
    truth = world.truth[burn:steps]
    np.savez_compressed(scoring_dir / f"scoring_{cell}.npz",
        evaluation_truth=truth, event_mask=truth >= threshold,
        event_threshold=np.asarray(threshold), cell_groups=world.cell_groups)
    np.savez_compressed(prediction_dir / f"observations_{cell}.npz",
                        **_record_arrays(selected, arrivals, burn=burn))
    selected_ids = {item.nullifier for item in selected}
    selected_map = {item.nullifier: item for item in world.observations
                    if item.nullifier in selected_ids}
    summary = {
        "cell": cell, "config_hash": config_hash(cfg), "trace_hash": trace.trace_hash,
        "selected_records": len(selected), "context_rows": len(center),
        "scoring_rows": len(truth), "public_diagnostics": public_diagnostics,
        "transport_metrics_hash": hashlib.sha256(canonical_json({
            key: value for key, value in transport.metrics.items()
            if key != "local_decisions"})).hexdigest(),
        "resource_invariants_pass": True,
    }
    return summary, world, selected, arrivals | {"__selected_map__": selected_map}


def build_validation_seed(root: str | Path, stage: str | Path, seed: int) -> dict:
    root, stage, seed = Path(root), Path(stage), int(seed)
    registration = load_registration(root)
    if seed not in registration["worlds"]["seeds"]:
        raise ValueError("unregistered validation seed")
    summaries, severe = {}, None
    for cell in BASE_CELLS:
        summary, world, selected, packed = _save_base(root, stage, seed, cell)
        summaries[cell] = summary
        arrivals = {key: value for key, value in packed.items()
                    if key != "__selected_map__"}
        if cell == "severe_clean":
            severe = (world, selected, arrivals, packed["__selected_map__"])
    if severe is None:
        raise AssertionError("severe clean base not built")
    clean_world, selected, arrivals, clean_map = severe
    selected_ids = set(clean_map)
    prediction_dir = stage / "prediction" / str(seed)
    for cell in ATTACK_CELLS:
        scenario_cfg = validation_campaign_config(root, cell)
        registered_attack = registration["attack_contract"][cell]
        observed_attack = {key: scenario_cfg["attack"][key]
                           for key in ("kind", "fraction", "amplitude", "start_epoch")}
        if observed_attack != registered_attack:
            raise ValueError("validation executable attack differs from registration")
        scenario = generate_world(scenario_cfg, seed)
        pairing = _assert_attack_world_pair(clean_world, scenario, scenario_cfg)
        scenario_map = {item.nullifier: item for item in scenario.observations
                        if item.nullifier in selected_ids}
        _assert_scenario_identity(clean_map, scenario_map)
        ordered = tuple(scenario_map[item.nullifier] for item in selected)
        np.savez_compressed(prediction_dir / f"observations_{cell}.npz",
                            **_record_arrays(ordered, arrivals,
                                             burn=int(scenario_cfg["world"]["burn_in_steps"])))
        summaries[cell] = {
            "cell": cell, "base_cell": "severe_clean",
            "config_hash": config_hash(scenario_cfg),
            "selected_records": len(ordered), "resource_invariants_pass": True,
            "pairing_audit": pairing,
        }
    result = {
        "schema_version": 1, "role": "validation inputs only; scoring capability separate",
        "seed": seed, "cells": registration["worlds"]["cells"],
        "cell_summary": summaries, "scientific_outcome_computed": False,
        "truth_in_prediction_files": False, "protected_namespaces_opened": False,
    }
    write_json(stage / "prediction" / str(seed) / "summary.json", result)
    return result


def finalize_bundle(root: str | Path, stage: str | Path, source_lock_sha256: str) -> dict:
    root, stage = Path(root), Path(stage)
    registration = load_registration(root)
    calibration_source = root / registration["fixed_estimator"]["innovation_calibration"]
    shutil.copy2(calibration_source, stage / "global_calibration.json")
    mechanism = stage / "mechanism"
    mechanism.mkdir(exist_ok=True)
    save_csr(mechanism / "physical_operator.npz",
             expected_physical_operator(int(registration["worlds"]["grid_side"])))
    expected_payloads = _expected_payloads(registration)
    files = sorted(path for path in stage.rglob("*")
                   if path.is_file() and path.name != "manifest.json")
    payloads = {str(path.relative_to(stage)).replace("\\", "/"): sha256_file(path)
                for path in files}
    if set(payloads) != expected_payloads:
        missing = sorted(expected_payloads - set(payloads))
        extra = sorted(set(payloads) - expected_payloads)
        raise ValueError(f"validation payload inventory mismatch: missing={missing}, extra={extra}")
    manifest = {
        "schema_version": 1,
        "role": "hash-bound independent synthetic validation inputs; zero outcomes",
        "registration": REGISTRATION,
        "registration_sha256": sha256_file(root / REGISTRATION),
        "source_lock_sha256": source_lock_sha256,
        "seeds": registration["worlds"]["seeds"],
        "cells": registration["worlds"]["cells"],
        "context_base": {cell: _cell_base(cell) for cell in registration["worlds"]["cells"]},
        "calibration_source": str(calibration_source.relative_to(root)).replace("\\", "/"),
        "calibration_source_sha256": sha256_file(calibration_source),
        "source_lineage": _source_lineage(root, registration),
        "expected_payload_count": len(expected_payloads),
        "payload_sha256": payloads,
        "payload_root_sha256": hashlib.sha256(canonical_json(payloads)).hexdigest(),
        "scientific_outcomes": 0,
        "protected_namespaces_opened": False,
    }
    write_json(stage / "manifest.json", manifest)
    return manifest


def _validate_calibration(root: Path, bundle: Path, manifest: dict,
                          registration: dict[str, Any]) -> None:
    relative = registration["fixed_estimator"]["innovation_calibration"]
    source = root / relative
    copied = bundle / "global_calibration.json"
    if (manifest.get("calibration_source") != relative
            or manifest.get("calibration_source_sha256") != sha256_file(source)
            or sha256_file(copied) != sha256_file(source)):
        raise ValueError("validation calibration lineage mismatch")
    calibration = json.loads(copied.read_text(encoding="utf-8"))
    scale = calibration.get("innovation_scales", {}).get(
        registration["fixed_estimator"]["innovation_covariance"])
    if (calibration.get("schema_version") != 1
            or calibration.get("role") != "frozen_global_calibration"
            or calibration.get("fold_unit") != "world"
            or calibration.get("fold_count") != 8
            or calibration.get("development_worlds_read") is not False
            or calibration.get("candidate_selection_performed") is not False
            or calibration.get("scientific_outcome_computed") is not False
            or not isinstance(scale, (int, float)) or not math.isfinite(scale)
            or scale <= 0):
        raise ValueError("unsafe validation covariance calibration")


def _validate_attack_payloads(root: Path, bundle: Path, seed: int,
                              registration: dict[str, Any]) -> None:
    prediction = bundle / "prediction" / str(seed)
    clean_arrays = _load_exact_npz(
        prediction / "observations_severe_clean.npz", OBSERVATION_FIELDS)
    score = _load_exact_npz(
        bundle / "scoring" / str(seed) / "scoring_severe_clean.npz", SCORING_FIELDS)
    burn = int(registration["worlds"]["burn_in_epochs"])
    steps = int(registration["worlds"]["acquisition_epochs"])
    baseline = float(validation_campaign_config(root, "severe_clean")["world"].get(
        "baseline", 12.0))
    clip = float(validation_campaign_config(root, "severe_clean")["privacy"]["clip"])
    identity_fields = OBSERVATION_FIELDS - {"value"}
    for cell in ATTACK_CELLS:
        attacked = _load_exact_npz(prediction / f"observations_{cell}.npz",
                                   OBSERVATION_FIELDS)
        if any(not np.array_equal(clean_arrays[name], attacked[name])
               for name in identity_fields):
            raise ValueError("validation attack observation identity mismatch")
        contract = registration["attack_contract"][cell]
        epochs = clean_arrays["epoch"].astype(np.int64) + burn
        truth = score["evaluation_truth"][clean_arrays["epoch"], clean_arrays["cell"]]
        if contract["kind"] == "adversarial_drift":
            direction = np.where(truth >= baseline, -1.0, 1.0)
            delta = direction * float(contract["amplitude"]) * (
                epochs - int(contract["start_epoch"]) + 1) / max(
                    1, steps - int(contract["start_epoch"]))
        elif contract["kind"] == "hotspot_suppression":
            delta = np.full(len(epochs), -float(contract["amplitude"]))
        else:
            raise ValueError("unregistered validation attack kind")
        eligible = epochs >= int(contract["start_epoch"])
        unchanged = np.isclose(attacked["value"], clean_arrays["value"], rtol=0, atol=1e-12)
        perturbed = np.asarray([
            _clipped_perturbation_consistent(float(clean), float(other), float(change), clip)
            for clean, other, change in zip(
                clean_arrays["value"], attacked["value"], delta, strict=True)
        ])
        if np.any(~unchanged & (~eligible | ~perturbed)) or not np.any(~unchanged):
            raise ValueError("validation attack payload violates registered perturbation")


def _validate_bundle_semantics(root: Path, bundle: Path, manifest: dict,
                               registration: dict[str, Any]) -> None:
    operator = load_csr(bundle / "mechanism" / "physical_operator.npz")
    expected_operator = expected_physical_operator(int(registration["worlds"]["grid_side"]))
    difference = operator - expected_operator
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-14:
        raise ValueError("validation physical operator mismatch")
    _validate_calibration(root, bundle, manifest, registration)
    for seed in registration["worlds"]["seeds"]:
        summary_path = bundle / "prediction" / str(seed) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (set(summary) != {"schema_version", "role", "seed", "cells",
                             "cell_summary", "scientific_outcome_computed",
                             "truth_in_prediction_files", "protected_namespaces_opened"}
                or summary.get("schema_version") != 1 or summary.get("seed") != seed
                or summary.get("cells") != registration["worlds"]["cells"]
                or summary.get("scientific_outcome_computed") is not False
                or summary.get("truth_in_prediction_files") is not False
                or summary.get("protected_namespaces_opened") is not False):
            raise ValueError("unsafe validation seed summary")
        cell_summary = summary.get("cell_summary")
        if not isinstance(cell_summary, dict) or set(cell_summary) != set(
                registration["worlds"]["cells"]):
            raise ValueError("validation seed summary cell inventory mismatch")
        for cell, row in cell_summary.items():
            required = ({"cell", "config_hash", "trace_hash", "selected_records",
                         "context_rows", "scoring_rows", "public_diagnostics",
                         "transport_metrics_hash", "resource_invariants_pass"}
                        if cell in BASE_CELLS else
                        {"cell", "base_cell", "config_hash", "selected_records",
                         "resource_invariants_pass", "pairing_audit"})
            if (not isinstance(row, dict) or set(row) != required
                    or row.get("cell") != cell
                    or row.get("config_hash")
                    != registration["executable_sources"]["expected_config_hash"][cell]
                    or row.get("resource_invariants_pass") is not True
                    or int(row.get("selected_records", 0)) < 1):
                raise ValueError("validation seed summary semantics mismatch")
            if cell in BASE_CELLS:
                if (row["context_rows"] != registration["worlds"][
                        "acquisition_epochs"] + registration["worlds"][
                        "fixed_lag_epochs"] - registration["worlds"]["burn_in_epochs"]
                        or row["scoring_rows"] != registration["worlds"][
                        "acquisition_epochs"] - registration["worlds"]["burn_in_epochs"]):
                    raise ValueError("validation seed summary clock mismatch")
            else:
                pairing = row["pairing_audit"]
                if (row["base_cell"] != "severe_clean"
                        or not isinstance(pairing, dict)
                        or pairing.get("latent_world_equal") is not True
                        or pairing.get("exact_perturbation") is not True
                        or pairing.get("perturbed_records", 0) < 1
                        or pairing.get("attacker_users") != round(
                            registration["worlds"]["agents"]
                            * registration["attack_contract"][cell]["fraction"])):
                    raise ValueError("validation attack pairing audit mismatch")
        for cell in registration["worlds"]["clean_cells"]:
            context = _load_exact_npz(
                bundle / "prediction" / str(seed) / f"context_{cell}.npz",
                CONTEXT_FIELDS)
            if PREDICTION_FORBIDDEN & set(context):
                raise ValueError("forbidden validation prediction field")
            _validate_context(context, registration, operator)
            scoring = _load_exact_npz(
                bundle / "scoring" / str(seed) / f"scoring_{cell}.npz",
                SCORING_FIELDS)
            _validate_scoring(scoring, registration)
        expected_count = None
        for cell in registration["worlds"]["cells"]:
            records, _ = _validate_observations(
                bundle / "prediction" / str(seed) / f"observations_{cell}.npz",
                registration,
                expected_count=(expected_count if cell in ATTACK_CELLS else None),
            )
            if cell == "severe_clean":
                expected_count = len(records)
        _validate_attack_payloads(root, bundle, seed, registration)


def verify_bundle(root: str | Path, bundle: str | Path,
                  *, expected_manifest_sha256: str) -> dict:
    root, bundle = Path(root), Path(bundle)
    manifest_path = bundle / "manifest.json"
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or sha256_file(manifest_path) != expected_manifest_sha256):
        raise ValueError("validation input manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registration = load_registration(root)
    source_lock = root / registration["input_producer"]["source_lock"]
    expected_context = {cell: _cell_base(cell) for cell in registration["worlds"]["cells"]}
    expected_payloads = _expected_payloads(registration)
    payloads = manifest.get("payload_sha256")
    if (set(manifest) != MANIFEST_FIELDS or manifest.get("schema_version") != 1
            or manifest.get("role")
            != "hash-bound independent synthetic validation inputs; zero outcomes"
            or manifest.get("registration") != REGISTRATION
            or manifest.get("registration_sha256") != sha256_file(root / REGISTRATION)
            or manifest.get("source_lock_sha256") != sha256_file(source_lock)
            or manifest.get("scientific_outcomes") != 0
            or manifest.get("protected_namespaces_opened") is not False
            or manifest.get("seeds") != registration["worlds"]["seeds"]
            or manifest.get("cells") != registration["worlds"]["cells"]
            or manifest.get("context_base") != expected_context
            or manifest.get("source_lineage") != _source_lineage(root, registration)
            or manifest.get("expected_payload_count") != len(expected_payloads)
            or not isinstance(payloads, dict) or set(payloads) != expected_payloads):
        raise ValueError("unsafe or stale validation input manifest")
    actual_files = {
        str(path.relative_to(bundle)).replace("\\", "/")
        for path in bundle.rglob("*") if path.is_file()
    }
    if actual_files != expected_payloads | {"manifest.json"}:
        raise ValueError("validation bundle contains missing or unregistered files")
    actual = {relative: sha256_file(bundle / relative) for relative in sorted(payloads)}
    if (actual != payloads
            or hashlib.sha256(canonical_json(actual)).hexdigest()
            != manifest["payload_root_sha256"]):
        raise ValueError("validation input payload hash mismatch")
    _validate_bundle_semantics(root, bundle, manifest, registration)
    return manifest


def load_prediction_view(root: str | Path, bundle: str | Path, seed: int, cell: str,
                         *, expected_manifest_sha256: str) -> dict[str, Any]:
    root, bundle = Path(root), Path(bundle)
    manifest = verify_bundle(root, bundle,
                             expected_manifest_sha256=expected_manifest_sha256)
    if seed not in manifest["seeds"] or cell not in manifest["cells"]:
        raise ValueError("unregistered validation seed/cell")
    base = manifest["context_base"][cell]
    context_path = bundle / "prediction" / str(seed) / f"context_{base}.npz"
    observation_path = bundle / "prediction" / str(seed) / f"observations_{cell}.npz"
    with np.load(context_path, allow_pickle=False) as arrays:
        if PREDICTION_FORBIDDEN & set(arrays.files):
            raise ValueError("scoring field crossed validation prediction boundary")
    context = _load_exact_npz(context_path, CONTEXT_FIELDS)
    records, arrivals = restore_records(observation_path)
    return {
        "public_center": context["public_center"],
        "prediction_epochs": context["prediction_epochs"],
        "operator": load_csr(bundle / "mechanism" / "physical_operator.npz"),
        "residual_forcing": context["residual_forcing"],
        "observations": records,
        "arrival_map": arrivals,
    }


def load_scoring_view(bundle: str | Path, seed: int, cell: str) -> dict[str, np.ndarray]:
    base = _cell_base(cell)
    path = Path(bundle) / "scoring" / str(seed) / f"scoring_{base}.npz"
    return _load_exact_npz(path, SCORING_FIELDS)

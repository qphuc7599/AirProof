from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import config_hash
from .experiment import source_tree_digest
from .metrics import prediction_metrics
from .records import Observation
from .simulator import SyntheticWorld, generate_world
from .twin import FixedLagTwin

PREDICTORS = (
    "persistence",
    "idw",
    "histgb_spatiotemporal",
    "graph_squared",
    "airproof_predictive_residual",
)


def _cell_coordinates(side: int) -> np.ndarray:
    rows, columns = np.indices((side, side))
    return np.column_stack([rows.ravel(), columns.ravel()]).astype(float)


def _idw_field(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    observed = np.flatnonzero(np.isfinite(values))
    if not len(observed):
        return np.full(len(values), np.nan)
    distances = np.linalg.norm(
        coordinates[:, np.newaxis, :] - coordinates[observed][np.newaxis, :, :], axis=2
    )
    weights = 1.0 / np.maximum(distances, 1e-6) ** 2
    result = weights @ values[observed] / weights.sum(axis=1)
    result[observed] = values[observed]
    return result


def _epoch_cell_means(
    observations: tuple[Observation, ...],
    steps: int,
    cells: int,
) -> np.ndarray:
    totals = np.zeros((steps, cells), dtype=float)
    counts = np.zeros((steps, cells), dtype=int)
    for item in observations:
        totals[item.epoch, item.cell] += float(item.value)
        counts[item.epoch, item.cell] += 1
    means = np.full((steps, cells), np.nan)
    present = counts > 0
    means[present] = totals[present] / counts[present]
    return means


def _evidence_idw(world: SyntheticWorld, side: int) -> np.ndarray:
    steps, cells = world.truth.shape
    means = _epoch_cell_means(
        (*world.observations, *world.reference_observations), steps, cells
    )
    coordinates = _cell_coordinates(side)
    result = np.empty_like(world.truth)
    initial_mean = float(np.nanmean(means[0])) if np.any(np.isfinite(means[0])) else 12.0
    fallback = np.full(cells, initial_mean)
    for epoch in range(steps):
        field = _idw_field(means[epoch], coordinates)
        if np.any(np.isfinite(field)):
            fallback = np.where(np.isfinite(field), field, fallback)
        result[epoch] = fallback
    return result


def _reference_map(world: SyntheticWorld) -> dict[tuple[int, int], float]:
    return {(item.epoch, item.cell): float(item.value) for item in world.reference_observations}


def _features(
    current: np.ndarray,
    previous: np.ndarray,
    epoch: int,
    coordinates: np.ndarray,
    cells: np.ndarray,
    side: int,
) -> np.ndarray:
    hour = epoch % 24
    return np.column_stack(
        [
            current[cells],
            previous[cells],
            coordinates[cells, 0] / max(side - 1, 1),
            coordinates[cells, 1] / max(side - 1, 1),
            np.sin(2.0 * np.pi * hour / 24.0) * np.ones(len(cells)),
            np.cos(2.0 * np.pi * hour / 24.0) * np.ones(len(cells)),
        ]
    )


def _histgb_predictions(
    world: SyntheticWorld,
    citizen_idw: np.ndarray,
    side: int,
    burn_in: int,
    seed: int,
) -> np.ndarray:
    coordinates = _cell_coordinates(side)
    references = _reference_map(world)
    train_x: list[np.ndarray] = []
    train_y: list[float] = []
    for (epoch, cell), target in sorted(references.items()):
        if epoch >= burn_in:
            continue
        previous = citizen_idw[max(0, epoch - 1)]
        train_x.append(
            _features(
                citizen_idw[epoch],
                previous,
                epoch,
                coordinates,
                np.asarray([cell]),
                side,
            )[0]
        )
        train_y.append(target)
    if len(train_y) < 32:
        raise ValueError("insufficient burn-in reference observations for HistGB")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(np.asarray(train_x), np.asarray(train_y))
    predictions = citizen_idw.copy()
    cells = np.arange(side * side)
    for epoch in range(burn_in, world.truth.shape[0]):
        predictions[epoch] = model.predict(
            _features(
                citizen_idw[epoch],
                citizen_idw[epoch - 1],
                epoch,
                coordinates,
                cells,
                side,
            )
        )
    return np.maximum(predictions, 0.0)


def _twin_predictions(
    world: SyntheticWorld,
    config: dict[str, Any],
    *,
    predictive_residual: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    twin_cfg = config["twin"]
    side = int(config["world"]["grid_side"])
    steps = int(config["world"]["steps"])
    initial = np.full(side * side, float(config["world"].get("baseline", 12.0)))
    twin = FixedLagTwin(
        side=side,
        steps=steps,
        fixed_lag=int(twin_cfg["fixed_lag"]),
        huber=predictive_residual,
        delta=float(twin_cfg["huber_delta"]),
        lambda_prior=float(twin_cfg["lambda_prior"]),
        lambda_spatial=float(twin_cfg["lambda_spatial"]),
        max_irls=int(twin_cfg["max_irls"]),
        tolerance=float(twin_cfg["tolerance"]),
        initial_state=initial,
        predictive_residual=predictive_residual,
        predictive_adaptive_delta=bool(twin_cfg.get("predictive_adaptive_delta", False)),
        predictive_clean_delta=float(twin_cfg.get("predictive_clean_delta", 4.0)),
        predictive_robust_delta=float(twin_cfg.get("predictive_robust_delta", 2.9)),
        predictive_activation_gain=float(
            twin_cfg.get("predictive_activation_gain", 1.0)
        ),
        predictive_calibrated_gate=bool(
            twin_cfg.get("predictive_calibrated_gate", False)
        ),
        predictive_calibration_epochs=int(
            twin_cfg.get("predictive_calibration_epochs", 48)
        ),
        predictive_excess_gate_start=float(
            twin_cfg.get("predictive_excess_gate_start", 0.01)
        ),
        predictive_excess_gate_full=float(
            twin_cfg.get("predictive_excess_gate_full", 0.04)
        ),
        innovation_gate_start=float(twin_cfg.get("innovation_gate_start", 0.02)),
        innovation_gate_full=float(twin_cfg.get("innovation_gate_full", 0.12)),
        innovation_gate_tail_z=float(twin_cfg.get("innovation_gate_tail_z", 3.5)),
        correction_delta=float(twin_cfg.get("correction_delta", 2.9)),
        lambda_correction=float(twin_cfg.get("lambda_correction", 0.5)),
        correction_clip=float(twin_cfg.get("correction_clip", 8.0)),
        correction_clean_gain=float(twin_cfg.get("correction_clean_gain", 0.25)),
        correction_attack_gain=float(twin_cfg.get("correction_attack_gain", 3.0)),
        correction_gate_start=float(twin_cfg.get("correction_gate_start", 0.15)),
        correction_gate_full=float(twin_cfg.get("correction_gate_full", 0.22)),
        correction_gate_ewma=float(twin_cfg.get("correction_gate_ewma", 0.25)),
    )
    arrivals: dict[int, list[Observation]] = {epoch: [] for epoch in range(steps)}
    for item in (*world.observations, *world.reference_observations):
        arrivals[item.epoch].append(item)
    for epoch in range(steps):
        twin.ingest_at(epoch, arrivals[epoch])
    return twin.states, {
        "predictive_tail_fraction_mean": float(
            np.mean(twin.predictive_tail_fraction_history)
        ),
        "predictive_tail_fraction_p95": float(
            np.percentile(twin.predictive_tail_fraction_history, 95)
        ),
        "predictive_activation_mean": float(
            np.mean(twin.predictive_activation_by_epoch)
        ),
        "predictive_activation_p95": float(
            np.percentile(twin.predictive_activation_by_epoch, 95)
        ),
        "predictive_delta_mean": float(np.mean(twin.predictive_delta_history)),
        "predictive_tail_baseline_final": float(
            twin.predictive_tail_baseline_history[-1]
        ),
    }


def _score_predictions(
    world: SyntheticWorld,
    predictions: np.ndarray,
    burn_in: int,
    reference_cells: np.ndarray,
) -> dict[str, float]:
    mask = np.ones(world.truth.shape[1], dtype=bool)
    mask[reference_cells] = False
    metrics = prediction_metrics(
        world.truth[burn_in:, mask],
        predictions[burn_in:, mask],
        world.cell_groups[mask],
    )
    return {key: float(value) for key, value in metrics.items()}


def run_prediction_seed(
    config: dict[str, Any],
    seed: int,
    predictors: tuple[str, ...] = PREDICTORS,
) -> list[dict[str, Any]]:
    unknown = sorted(set(predictors) - set(PREDICTORS))
    if unknown:
        raise ValueError(f"unknown endpoint predictors {unknown}")
    world = generate_world(config, seed)
    side = int(config["world"]["grid_side"])
    burn_in = int(config["world"].get("burn_in_steps", 0))
    evidence_idw = _evidence_idw(world, side)
    reference_cells = np.unique([item.cell for item in world.reference_observations])
    prediction_map: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    if "persistence" in predictors:
        persistence = np.vstack([evidence_idw[0], evidence_idw[:-1]])
        prediction_map["persistence"] = persistence
    if "idw" in predictors:
        prediction_map["idw"] = evidence_idw
    if "histgb_spatiotemporal" in predictors:
        prediction_map["histgb_spatiotemporal"] = _histgb_predictions(
            world, evidence_idw, side, burn_in, seed
        )
    if "graph_squared" in predictors:
        prediction_map["graph_squared"], diagnostics["graph_squared"] = _twin_predictions(
            world, config, predictive_residual=False
        )
    if "airproof_predictive_residual" in predictors:
        (
            prediction_map["airproof_predictive_residual"],
            diagnostics["airproof_predictive_residual"],
        ) = _twin_predictions(
            world,
            config,
            predictive_residual=True,
        )
    return [
        {
            "seed": seed,
            "predictor": name,
            "metrics": _score_predictions(world, prediction_map[name], burn_in, reference_cells),
            "diagnostics": diagnostics.get(name, {}),
            "world": world.metadata,
        }
        for name in predictors
    ]


def _prediction_job(
    job: tuple[dict[str, Any], int, tuple[str, ...]],
) -> list[dict[str, Any]]:
    return run_prediction_seed(*job)


def run_prediction_benchmark(
    config: dict[str, Any],
    seeds: list[int],
    predictors: tuple[str, ...] = PREDICTORS,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    jobs = [(config, seed, predictors) for seed in seeds]
    if workers == 1:
        batches = [_prediction_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            batches = list(pool.map(_prediction_job, jobs))
    rows = [row for batch in batches for row in batch]
    report: dict[str, Any] = {
        "status": "endpoint-specific-prediction-benchmark",
        "protocol": {
            "common_world_per_seed": True,
            "evaluation_cells": "all non-reference grid cells",
            "histgb_training": "burn-in reference labels only",
            "common_test_evidence": "citizen observations and current trusted references",
            "test_labels_opened_during_fit": False,
            "network_and_privacy_layers": "excluded for endpoint isolation",
        },
        "config_hash": config_hash(config),
        "source_tree_sha256": source_tree_digest(),
        "rows": rows,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report


def write_prediction_benchmark(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path

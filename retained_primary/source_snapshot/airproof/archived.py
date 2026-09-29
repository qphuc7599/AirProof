from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .data import sha256_file
from .experiment import source_tree_digest
from .records import Observation
from .twin import robust_state_update

# Dataset station metadata used only for distance geometry, never as a target feature.
BEIJING_COORDINATES = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.217, 116.230),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}


def _distance_matrix(stations: list[str]) -> np.ndarray:
    coordinates = np.array([BEIJING_COORDINATES[station] for station in stations], dtype=float)
    return _coordinate_distance_matrix(coordinates)


def _coordinate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    lat_scale = 111.32
    lon_scale = 111.32 * np.cos(np.deg2rad(np.mean(coordinates[:, 0])))
    scaled = coordinates * np.array([lat_scale, lon_scale])
    return np.sqrt(np.sum((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2))


def _graph_laplacian(distances: np.ndarray, held_out: int, neighbors: int = 4) -> sparse.csr_matrix:
    count = distances.shape[0]
    adjacency = np.zeros((count, count), dtype=float)
    for index in range(count):
        if index == held_out:
            continue
        available = [item for item in np.argsort(distances[index]) if item not in (index, held_out)]
        for other in available[:neighbors]:
            weight = np.exp(-distances[index, other] / 15.0)
            adjacency[index, other] = adjacency[other, index] = max(adjacency[index, other], weight)
    # Held-out station is connected geometrically but never contributes an observation.
    nearest = [item for item in np.argsort(distances[held_out]) if item != held_out][:neighbors]
    for other in nearest:
        weight = np.exp(-distances[held_out, other] / 15.0)
        adjacency[held_out, other] = adjacency[other, held_out] = weight
    return sparse.diags(adjacency.sum(axis=1)) - sparse.csr_matrix(adjacency)


def _idw(values: np.ndarray, distances: np.ndarray, held_out: int) -> float | None:
    valid = np.flatnonzero(np.isfinite(values) & (np.arange(len(values)) != held_out))
    if len(valid) == 0:
        return None
    weights = 1.0 / np.maximum(distances[held_out, valid], 1e-6) ** 2
    return float(np.sum(weights * values[valid]) / np.sum(weights))


def _adaptive_kernel_weights(
    distances: np.ndarray,
    target: int,
    sources: list[int],
    *,
    kernel: str,
    scale: float,
    neighbors: int,
) -> np.ndarray:
    """Return a fixed, geometry-only edge vector for one target node."""
    if target in sources:
        raise ValueError("target must not be present in source nodes")
    if not sources:
        raise ValueError("at least one source node is required")
    source_array = np.asarray(sources, dtype=int)
    target_distances = distances[target, source_array]
    keep = np.argsort(target_distances)[: min(int(neighbors), len(sources))]
    selected = target_distances[keep]
    if kernel == "idw":
        selected_weights = 1.0 / np.maximum(selected, 0.25) ** float(scale)
    elif kernel == "exponential":
        selected_weights = np.exp(-selected / float(scale))
    elif kernel == "gaussian":
        selected_weights = np.exp(-0.5 * (selected / float(scale)) ** 2)
    else:
        raise ValueError(f"Unknown adaptive transfer kernel: {kernel}")
    selected_weights = np.maximum(selected_weights, 1e-12)
    weights = np.zeros(len(sources), dtype=float)
    weights[keep] = selected_weights / selected_weights.sum()
    return weights


def _weighted_huber_location(
    values: np.ndarray,
    weights: np.ndarray,
    robust_delta: float | None,
) -> np.ndarray:
    """Contemporaneous weighted location for rows with arbitrary missingness."""
    if values.ndim != 2 or values.shape[1] != len(weights):
        raise ValueError("values/weights shape mismatch")
    valid = np.isfinite(values)
    effective = valid * weights[None, :]
    denominator = effective.sum(axis=1)
    safe_values = np.where(valid, values, 0.0)
    location = np.divide(
        (safe_values * effective).sum(axis=1),
        denominator,
        out=np.full(len(values), np.nan, dtype=float),
        where=denominator > 0,
    )
    if robust_delta is None:
        return location
    for _ in range(3):
        residual = values - location[:, None]
        absolute = np.abs(residual)
        median_absolute = np.nanmedian(np.where(valid, absolute, np.nan), axis=1)
        scale = np.maximum(1.4826 * median_absolute, 1.0)
        normalized = absolute / scale[:, None]
        influence = np.minimum(1.0, float(robust_delta) / np.maximum(normalized, 1e-12))
        robust_weights = effective * np.where(valid, influence, 0.0)
        robust_denominator = robust_weights.sum(axis=1)
        location = np.divide(
            (safe_values * robust_weights).sum(axis=1),
            robust_denominator,
            out=location.copy(),
            where=robust_denominator > 0,
        )
    return location


def _adaptive_predict(
    values: np.ndarray,
    distances: np.ndarray,
    target: int,
    sources: list[int],
    candidate: dict[str, Any],
) -> np.ndarray:
    weights = _adaptive_kernel_weights(
        distances,
        target,
        sources,
        kernel=str(candidate["kernel"]),
        scale=float(candidate["scale"]),
        neighbors=int(candidate["neighbors"]),
    )
    return _weighted_huber_location(
        values[:, sources],
        weights,
        candidate.get("robust_delta"),
    )


def _learn_spatial_bias(
    fit_values: np.ndarray,
    distances: np.ndarray,
    target: int,
    sources: list[int],
    candidate: dict[str, Any],
) -> float:
    """Learn a city-level bias surface without ever using target labels."""
    source_biases = np.full(len(sources), np.nan, dtype=float)
    for position, pseudo_target in enumerate(sources):
        peers = [item for item in sources if item != pseudo_target]
        if not peers:
            continue
        prediction = _adaptive_predict(
            fit_values,
            distances,
            pseudo_target,
            peers,
            candidate,
        )
        truth = fit_values[:, pseudo_target]
        valid = np.isfinite(prediction) & np.isfinite(truth)
        if valid.any():
            source_biases[position] = float(np.mean(truth[valid] - prediction[valid]))
    valid_biases = np.isfinite(source_biases)
    if not valid_biases.any():
        return 0.0
    valid_sources = [source for source, valid in zip(sources, valid_biases) if valid]
    weights = _adaptive_kernel_weights(
        distances,
        target,
        valid_sources,
        kernel=str(candidate["kernel"]),
        scale=float(candidate["scale"]),
        neighbors=int(candidate["neighbors"]),
    )
    return float(np.dot(weights, source_biases[valid_biases]))


def _candidate_prediction(
    fit_values: np.ndarray,
    evaluation_values: np.ndarray,
    distances: np.ndarray,
    target: int,
    sources: list[int],
    candidate: dict[str, Any],
) -> np.ndarray:
    prediction = _adaptive_predict(
        evaluation_values,
        distances,
        target,
        sources,
        candidate,
    )
    shrinkage = float(candidate.get("bias_shrinkage", 0.0))
    if shrinkage:
        prediction = prediction + shrinkage * _learn_spatial_bias(
            fit_values,
            distances,
            target,
            sources,
            candidate,
        )
    return prediction


def _transfer_candidates(station_count: int) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    """Frozen two-stage search space for city-adaptive transfer v2."""
    neighbor_choices = sorted({3, 4, 6, station_count - 1})
    base: list[dict[str, Any]] = []
    for scale, neighbors in itertools.product((1.0, 1.5, 2.0, 2.5, 3.0), neighbor_choices):
        base.append({"kernel": "idw", "scale": scale, "neighbors": neighbors})
    for kernel in ("exponential", "gaussian"):
        for scale, neighbors in itertools.product((5.0, 10.0, 20.0, 40.0, 80.0), neighbor_choices):
            base.append({"kernel": kernel, "scale": scale, "neighbors": neighbors})
    return base, [1.345, 2.0, 3.0], [0.0, 0.5, 1.0]


def _score_transfer_candidate(
    fit_values: np.ndarray,
    validation_values: np.ndarray,
    distances: np.ndarray,
    available: list[int],
    candidate: dict[str, Any],
    *,
    worst_station_weight: float,
) -> dict[str, Any]:
    station_rmse: list[float] = []
    support = 0
    for pseudo_target in available:
        sources = [item for item in available if item != pseudo_target]
        prediction = _candidate_prediction(
            fit_values,
            validation_values,
            distances,
            pseudo_target,
            sources,
            candidate,
        )
        truth = validation_values[:, pseudo_target]
        valid = np.isfinite(prediction) & np.isfinite(truth)
        if not valid.any():
            continue
        residual = prediction[valid] - truth[valid]
        station_rmse.append(float(np.sqrt(np.mean(residual**2))))
        support += int(valid.sum())
    if not station_rmse:
        return {
            "score": float("inf"),
            "mean_rmse": float("inf"),
            "worst_rmse": float("inf"),
            "support": 0,
        }
    mean_rmse = float(np.mean(station_rmse))
    worst_rmse = float(np.max(station_rmse))
    return {
        "score": mean_rmse + float(worst_station_weight) * worst_rmse,
        "mean_rmse": mean_rmse,
        "worst_rmse": worst_rmse,
        "support": support,
    }


def _select_city_adaptive_candidate(
    fit_values: np.ndarray,
    validation_values: np.ndarray,
    distances: np.ndarray,
    available: list[int],
    *,
    station_count: int,
    shortlist: int,
    worst_station_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_candidates, robust_grid, bias_grid = _transfer_candidates(station_count)
    stage_one: list[dict[str, Any]] = []
    for candidate in base_candidates:
        complete = {**candidate, "robust_delta": None, "bias_shrinkage": 0.0}
        stage_one.append(
            {
                **complete,
                **_score_transfer_candidate(
                    fit_values,
                    validation_values,
                    distances,
                    available,
                    complete,
                    worst_station_weight=worst_station_weight,
                ),
            }
        )
    stage_one.sort(key=lambda item: (item["score"], item["worst_rmse"], item["mean_rmse"]))
    finalists: list[dict[str, Any]] = []
    for base in stage_one[: int(shortlist)]:
        stripped = {key: base[key] for key in ("kernel", "scale", "neighbors")}
        refinements = [(None, 0.0), *itertools.product(robust_grid, bias_grid)]
        for robust_delta, bias_shrinkage in refinements:
            candidate = {
                **stripped,
                "robust_delta": robust_delta,
                "bias_shrinkage": bias_shrinkage,
            }
            finalists.append(
                {
                    **candidate,
                    **_score_transfer_candidate(
                        fit_values,
                        validation_values,
                        distances,
                        available,
                        candidate,
                        worst_station_weight=worst_station_weight,
                    ),
                }
            )
    finalists.sort(key=lambda item: (item["score"], item["worst_rmse"], item["mean_rmse"]))
    selected = {
        key: finalists[0][key]
        for key in (
            "kernel",
            "scale",
            "neighbors",
            "robust_delta",
            "bias_shrinkage",
            "score",
            "mean_rmse",
            "worst_rmse",
            "support",
        )
    }
    return selected, finalists


def run_beijing_station_transfer(
    parquet_path: str | Path,
    output: str | Path,
    *,
    pollutant: str = "PM2.5",
    purge_hours: int = 6,
    max_test_steps: int = 336,
) -> dict:
    parquet_path = Path(parquet_path)
    frame = pd.read_parquet(parquet_path, columns=["timestamp_utc", "station", pollutant])
    pivot = frame.pivot(index="timestamp_utc", columns="station", values=pollutant).sort_index()
    stations = sorted(pivot.columns)
    if set(stations) != set(BEIJING_COORDINATES):
        raise ValueError("Station coordinate ledger does not match the processed dataset")
    pivot = pivot[stations]
    split = int(len(pivot) * 0.8)
    train = pivot.iloc[:split]
    test = pivot.iloc[split + purge_hours : split + purge_hours + max_test_steps]
    distances = _distance_matrix(stations)
    rows: list[dict] = []
    started = perf_counter()

    for held_out, station in enumerate(stations):
        # The held-out station is removed before initialization and graph observation creation.
        other_columns = [name for name in stations if name != station]
        global_mean = float(train[other_columns].stack().mean())
        initial = np.array(
            [float(train[name].mean()) if name != station else global_mean for name in stations], dtype=float
        )
        laplacian = _graph_laplacian(distances, held_out)
        robust_state = initial.copy()
        squared_state = initial.copy()
        errors: dict[str, list[float]] = {"idw": [], "graph_squared": [], "graph_huber": []}
        observed_targets = 0
        for epoch, (_, series) in enumerate(test.iterrows()):
            values = series.to_numpy(dtype=float)
            observations = [
                Observation(
                    user_id=index,
                    epoch=epoch,
                    cell=index,
                    group=0,
                    value=float(values[index]),
                    sigma=8.0,
                    quality=1.0,
                    size_bytes=0,
                    nullifier=f"archive-{station}-{epoch}-{index}",
                    direct_arrival=epoch,
                    relay_arrival=epoch,
                    source_class="regulatory",
                )
                for index in range(len(stations))
                if index != held_out and np.isfinite(values[index])
            ]
            robust_state, _ = robust_state_update(
                robust_state,
                observations,
                laplacian,
                huber=True,
                delta=1.5,
                lambda_prior=0.5,
                lambda_spatial=0.25,
                max_irls=8,
                tolerance=1e-5,
            )
            squared_state, _ = robust_state_update(
                squared_state,
                observations,
                laplacian,
                huber=False,
                delta=1.5,
                lambda_prior=0.5,
                lambda_spatial=0.25,
                max_irls=2,
                tolerance=1e-5,
            )
            target = values[held_out]
            if not np.isfinite(target):
                continue
            observed_targets += 1
            idw = _idw(values, distances, held_out)
            if idw is not None:
                errors["idw"].append(idw - target)
            errors["graph_squared"].append(float(squared_state[held_out] - target))
            errors["graph_huber"].append(float(robust_state[held_out] - target))
        for method, residuals in errors.items():
            values = np.asarray(residuals)
            rows.append(
                {
                    "held_out_station": station,
                    "method": method,
                    "n": len(values),
                    "rmse": float(np.sqrt(np.mean(values**2))),
                    "mae": float(np.mean(np.abs(values))),
                    "bias": float(np.mean(values)),
                    "target_support": observed_targets,
                }
            )
    result_frame = pd.DataFrame(rows)
    aggregate = (
        result_frame.groupby("method")
        .agg(mean_station_rmse=("rmse", "mean"), worst_station_rmse=("rmse", "max"), mean_station_mae=("mae", "mean"))
        .reset_index()
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output.with_suffix(".folds.csv"), index=False)
    report = {
        "status": "measured-archived-supplementary",
        "dataset_sha256": sha256_file(parquet_path),
        "pollutant": pollutant,
        "train_rows": len(train),
        "purge_hours": purge_hours,
        "test_steps": len(test),
        "held_out_station_excluded_from_all_model_inputs": True,
        "runtime_seconds": perf_counter() - started,
        "aggregate": aggregate.to_dict(orient="records"),
        "folds": rows,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_beijing_city_adaptive_transfer(
    parquet_path: str | Path,
    output: str | Path,
    *,
    pollutant: str = "PM2.5",
    outer_train_fraction: float = 0.8,
    outer_purge_hours: int = 6,
    max_test_steps: int = 336,
    inner_fit_steps: int = 2016,
    inner_validation_steps: int = 336,
    inner_purge_hours: int = 6,
    shortlist: int = 6,
    worst_station_weight: float = 0.25,
    selection_only: bool = False,
) -> dict[str, Any]:
    """Nested, leakage-safe Beijing transfer with city-specific graph adaptation."""
    parquet_path = Path(parquet_path)
    frame = pd.read_parquet(parquet_path, columns=["timestamp_utc", "station", pollutant])
    pivot = frame.pivot(index="timestamp_utc", columns="station", values=pollutant).sort_index()
    stations = sorted(pivot.columns)
    if set(stations) != set(BEIJING_COORDINATES):
        raise ValueError("Station coordinate ledger does not match the processed dataset")
    pivot = pivot[stations]
    split = int(len(pivot) * float(outer_train_fraction))
    outer_train = pivot.iloc[:split]
    required_inner = int(inner_fit_steps) + int(inner_purge_hours) + int(inner_validation_steps)
    if len(outer_train) < required_inner:
        raise ValueError("Outer training partition is too short for the frozen inner protocol")
    inner_fit = outer_train.iloc[-required_inner : -(inner_purge_hours + inner_validation_steps)]
    inner_validation = outer_train.iloc[-inner_validation_steps:]
    # V1 used the first 336 post-split hours. V2 uses the archive tail, which was
    # untouched during the diagnosis and architecture correction.
    outer_test = pivot.iloc[-max_test_steps:]
    if outer_test.index.min() <= outer_train.index.max() + pd.Timedelta(hours=outer_purge_hours):
        raise ValueError("Outer train/test purge invariant is not satisfied")
    distances = _distance_matrix(stations)
    train_values = outer_train.to_numpy(dtype=float)
    fit_values = inner_fit.to_numpy(dtype=float)
    validation_values = inner_validation.to_numpy(dtype=float)
    test_values = outer_test.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    candidate_ledgers: dict[str, list[dict[str, Any]]] = {}
    started = perf_counter()

    for held_out, station in enumerate(stations):
        available = [index for index in range(len(stations)) if index != held_out]
        selected, finalists = _select_city_adaptive_candidate(
            fit_values,
            validation_values,
            distances,
            available,
            station_count=len(stations),
            shortlist=shortlist,
            worst_station_weight=worst_station_weight,
        )
        selections.append({"held_out_station": station, **selected})
        candidate_ledgers[station] = finalists
        if selection_only:
            continue
        candidate = {
            key: selected[key]
            for key in ("kernel", "scale", "neighbors", "robust_delta", "bias_shrinkage")
        }
        prediction = _candidate_prediction(
            train_values,
            test_values,
            distances,
            held_out,
            available,
            candidate,
        )
        target = test_values[:, held_out]
        valid = np.isfinite(prediction) & np.isfinite(target)
        residual = prediction[valid] - target[valid]
        rows.append(
            {
                "held_out_station": station,
                "method": "city_adaptive_graph_v2",
                "n": int(valid.sum()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
                "target_support": int(np.isfinite(target).sum()),
                "selected_kernel": selected["kernel"],
                "selected_scale": selected["scale"],
                "selected_neighbors": selected["neighbors"],
                "selected_robust_delta": selected["robust_delta"],
                "selected_bias_shrinkage": selected["bias_shrinkage"],
            }
        )
        idw_prediction = _adaptive_predict(
            test_values,
            distances,
            held_out,
            available,
            {
                "kernel": "idw",
                "scale": 2.0,
                "neighbors": len(available),
                "robust_delta": None,
            },
        )
        idw_valid = np.isfinite(idw_prediction) & np.isfinite(target)
        idw_residual = idw_prediction[idw_valid] - target[idw_valid]
        rows.append(
            {
                "held_out_station": station,
                "method": "idw_frozen",
                "n": int(idw_valid.sum()),
                "rmse": float(np.sqrt(np.mean(idw_residual**2))),
                "mae": float(np.mean(np.abs(idw_residual))),
                "bias": float(np.mean(idw_residual)),
                "target_support": int(np.isfinite(target).sum()),
            }
        )

    result_frame = pd.DataFrame(rows)
    if len(result_frame):
        aggregate_rows = (
            result_frame.groupby("method")
            .agg(
                mean_station_rmse=("rmse", "mean"),
                worst_station_rmse=("rmse", "max"),
                mean_station_mae=("mae", "mean"),
                mean_station_bias=("bias", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        aggregate_rows = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(result_frame):
        result_frame.to_csv(output.with_suffix(".folds.csv"), index=False)
    protocol = {
        "version": "beijing-city-adaptive-v2",
        "outer_train_fraction": outer_train_fraction,
        "outer_purge_hours": outer_purge_hours,
        "outer_test_window": "archive-tail-untouched-by-v1",
        "max_test_steps": max_test_steps,
        "inner_fit_steps": inner_fit_steps,
        "inner_validation_steps": inner_validation_steps,
        "inner_purge_hours": inner_purge_hours,
        "shortlist": shortlist,
        "worst_station_weight": worst_station_weight,
        "candidate_family": {
            "kernels": ["idw", "exponential", "gaussian"],
            "idw_exponents": [1.0, 1.5, 2.0, 2.5, 3.0],
            "length_scales_km": [5.0, 10.0, 20.0, 40.0, 80.0],
            "neighbors": sorted({3, 4, 6, len(stations) - 1}),
            "robust_delta": [None, 1.345, 2.0, 3.0],
            "bias_shrinkage": [0.0, 0.5, 1.0],
        },
        "selection_objective": "mean_station_rmse + 0.25 * worst_station_rmse",
    }
    protocol_canonical = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    report: dict[str, Any] = {
        "status": (
            "measured-city-adaptive-v2-selection-only"
            if selection_only
            else "measured-city-adaptive-v2-locked-test"
        ),
        "dataset_sha256": sha256_file(parquet_path),
        "source_tree_sha256": source_tree_digest(),
        "protocol_sha256": hashlib.sha256(protocol_canonical.encode()).hexdigest(),
        "protocol": protocol,
        "pollutant": pollutant,
        "outer_train_rows": len(outer_train),
        "inner_fit_rows": len(inner_fit),
        "inner_validation_rows": len(inner_validation),
        "outer_test_rows": len(outer_test),
        "outer_train_end": outer_train.index.max().isoformat(),
        "outer_test_start": outer_test.index.min().isoformat(),
        "outer_test_end": outer_test.index.max().isoformat(),
        "outer_target_excluded_from_training_validation_and_inputs": True,
        "test_labels_used_for_model_selection": False,
        "v1_preserved_at": "reports/analysis/beijing_transfer.json",
        "runtime_seconds": perf_counter() - started,
        "aggregate": aggregate_rows,
        "selections": selections,
        "folds": rows,
        "candidate_ledgers": candidate_ledgers,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def run_epa_station_transfer(
    parquet_path: str | Path,
    output: str | Path,
    *,
    purge_hours: int = 6,
    max_test_steps: int = 336,
) -> dict:
    """Leakage-safe held-out-monitor evaluation on normalized EPA AQS data."""
    parquet_path = Path(parquet_path)
    frame = pd.read_parquet(parquet_path)
    required = {"timestamp_utc", "station", "value", "latitude", "longitude"}
    if not required.issubset(frame.columns):
        raise ValueError(f"EPA frame missing columns: {sorted(required - set(frame.columns))}")
    coordinates_frame = frame.groupby("station", as_index=True)[["latitude", "longitude"]].median()
    stations = sorted(coordinates_frame.index.astype(str))
    if len(stations) < 4:
        raise ValueError("EPA transfer requires at least four monitoring stations")
    pivot = (
        frame.pivot(index="timestamp_utc", columns="station", values="value")
        .sort_index()[stations]
    )
    split = int(len(pivot) * 0.8)
    train = pivot.iloc[:split]
    test = pivot.iloc[split + purge_hours : split + purge_hours + max_test_steps]
    coordinate_values = coordinates_frame.loc[stations, ["latitude", "longitude"]].to_numpy()
    distances = _coordinate_distance_matrix(coordinate_values)
    rows: list[dict] = []
    started = perf_counter()
    for held_out, station in enumerate(stations):
        other_columns = [name for name in stations if name != station]
        global_mean = float(train[other_columns].stack().mean())
        initial = np.array(
            [float(train[name].mean()) if name != station else global_mean for name in stations],
            dtype=float,
        )
        laplacian = _graph_laplacian(distances, held_out)
        robust_state = initial.copy()
        squared_state = initial.copy()
        errors: dict[str, list[float]] = {"idw": [], "graph_squared": [], "graph_huber": []}
        observed_targets = 0
        for epoch, (_, series) in enumerate(test.iterrows()):
            values = series.to_numpy(dtype=float)
            observations = [
                Observation(
                    user_id=index,
                    epoch=epoch,
                    cell=index,
                    group=0,
                    value=float(values[index]),
                    sigma=5.0,
                    quality=1.0,
                    size_bytes=0,
                    nullifier=f"epa-{station}-{epoch}-{index}",
                    direct_arrival=epoch,
                    relay_arrival=epoch,
                    source_class="regulatory",
                )
                for index in range(len(stations))
                if index != held_out and np.isfinite(values[index])
            ]
            robust_state, _ = robust_state_update(
                robust_state,
                observations,
                laplacian,
                huber=True,
                delta=1.5,
                lambda_prior=0.5,
                lambda_spatial=0.25,
                max_irls=20,
                tolerance=1e-5,
            )
            squared_state, _ = robust_state_update(
                squared_state,
                observations,
                laplacian,
                huber=False,
                delta=1.5,
                lambda_prior=0.5,
                lambda_spatial=0.25,
                max_irls=2,
                tolerance=1e-5,
            )
            target = values[held_out]
            if not np.isfinite(target):
                continue
            observed_targets += 1
            idw = _idw(values, distances, held_out)
            if idw is not None:
                errors["idw"].append(idw - target)
            errors["graph_squared"].append(float(squared_state[held_out] - target))
            errors["graph_huber"].append(float(robust_state[held_out] - target))
        for method, residuals in errors.items():
            values = np.asarray(residuals)
            if len(values) == 0:
                continue
            rows.append(
                {
                    "held_out_station": station,
                    "method": method,
                    "n": len(values),
                    "rmse": float(np.sqrt(np.mean(values**2))),
                    "mae": float(np.mean(np.abs(values))),
                    "bias": float(np.mean(values)),
                    "target_support": observed_targets,
                }
            )
    result_frame = pd.DataFrame(rows)
    aggregate = (
        result_frame.groupby("method")
        .agg(
            mean_station_rmse=("rmse", "mean"),
            worst_station_rmse=("rmse", "max"),
            mean_station_mae=("mae", "mean"),
        )
        .reset_index()
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output.with_suffix(".folds.csv"), index=False)
    report = {
        "status": "measured-archived-primary-us",
        "dataset_sha256": sha256_file(parquet_path),
        "pollutant": "PM2.5",
        "train_rows": len(train),
        "purge_hours": purge_hours,
        "test_steps": len(test),
        "held_out_station_excluded_from_all_model_inputs": True,
        "runtime_seconds": perf_counter() - started,
        "aggregate": aggregate.to_dict(orient="records"),
        "folds": rows,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["artifact_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

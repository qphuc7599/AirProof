"""Nested public-station transfer using the production, weather-aware v4 twin.

This is contemporaneous spatial reconstruction, not future forecasting or a
citizen deployment. Outer target concentrations and weather never enter the
model. Public source stations remain explicitly regulatory observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist

from .archived import BEIJING_COORDINATES, _adaptive_kernel_weights
from .meteorology import PublicMeteorology, city_graph, transition_series
from .records import Observation
from .twin import FixedLagTwin


COMPASS = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()


def meteorological_components(direction, speed):
    """Meteorological FROM direction: u=-speed*sin(phi), v=-speed*cos(phi)."""
    labels = np.asarray(direction)
    speed = np.asarray(speed, dtype=float)
    angle = np.full(labels.shape, np.nan)
    for index, label in enumerate(COMPASS):
        angle[labels == label] = index * np.pi / 8
    if angle.shape != speed.shape:
        raise ValueError("wind direction/speed shape mismatch")
    speed = np.where(speed >= 0, speed, np.nan)
    u, v = -speed * np.sin(angle), -speed * np.cos(angle)
    # A measured calm wind has zero components even if its direction is missing.
    return np.where(speed == 0, 0., u), np.where(speed == 0, 0., v)


def relative_humidity(temperature, dewpoint):
    """Magnus approximation over water; derived RH, not separately measured RH."""
    temperature, dewpoint = np.asarray(temperature, float), np.asarray(dewpoint, float)
    if temperature.shape != dewpoint.shape:
        raise ValueError("temperature/dew point shape mismatch")
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        humidity = 100 * np.exp(17.625 * dewpoint / (243.04 + dewpoint)
                               - 17.625 * temperature / (243.04 + temperature))
    return np.clip(humidity, 0., 100.)


@dataclass(frozen=True)
class BeijingArchive:
    concentrations: np.ndarray
    weather: dict[str, np.ndarray]
    times: pd.DatetimeIndex
    stations: tuple[str, ...]
    coordinates: np.ndarray


def load_beijing_archive(path: str | Path) -> BeijingArchive:
    # Reconstruct from calendar columns. The legacy parquet incorrectly tagged
    # these local-clock values UTC; do not mutate that historical artifact.
    columns = ["year", "month", "day", "hour", "station", "PM2.5", "TEMP", "DEWP", "wd", "WSPM"]
    frame = pd.read_parquet(path, columns=columns)
    frame["local_time"] = pd.to_datetime(frame[["year", "month", "day", "hour"]]).dt.tz_localize("Asia/Shanghai")
    stations = tuple(sorted(frame.station.unique()))
    if set(stations) != set(BEIJING_COORDINATES):
        raise ValueError("Beijing station/geometry ledger mismatch")
    def pivot(name):
        return frame.pivot(index="local_time", columns="station", values=name).sort_index().loc[:, list(stations)]
    pollution = pivot("PM2.5")
    times = pd.DatetimeIndex(pollution.index)
    if not np.all(np.diff(times.asi8) == 3600 * 10**9):
        raise ValueError("archive must retain every hourly slot")
    temperature, dewpoint = pivot("TEMP").to_numpy(float), pivot("DEWP").to_numpy(float)
    u, v = meteorological_components(pivot("wd").to_numpy(), pivot("WSPM").to_numpy(float))
    latlon = np.array([BEIJING_COORDINATES[name] for name in stations])
    xy = (latlon - latlon.mean(axis=0)) * [111.32, 111.32 * np.cos(np.deg2rad(latlon[:, 0].mean()))]
    values = pollution.to_numpy(float)
    if np.any(values[np.isfinite(values)] < 0):
        raise ValueError("negative measured concentration")
    return BeijingArchive(values, {"wind_u": u, "wind_v": v, "temperature": temperature,
                                   "humidity": relative_humidity(temperature, dewpoint)}, times, stations, xy)


def archive_slices(length: int, protocol: dict) -> dict[str, slice]:
    split = int(length * protocol["outer_train_fraction"])
    validation_steps = protocol["inner_validation_steps"]
    fit_end = split - validation_steps - protocol["purge_hours"]
    fit_start = fit_end - protocol["inner_fit_steps"]
    # Exclude the previously inspected v2 last-336-hour endpoint and a six-hour
    # boundary. This fresh late-archive block is fixed before v4 outcomes.
    test_end = length - protocol["historical_v2_tail_steps"] - protocol["purge_hours"]
    test_start = test_end - protocol["test_steps"]
    if fit_start < 0 or test_start <= split + protocol["historical_v1_steps"] + protocol["purge_hours"]:
        raise ValueError("insufficient non-overlapping fitting/validation/test archive")
    return {"fit": slice(fit_start, fit_end), "validation": slice(split - validation_steps, split),
            "test": slice(test_start, test_end)}


def public_weather(fit: dict[str, np.ndarray], current: dict[str, np.ndarray], sources: list[int]) -> PublicMeteorology:
    """Source-only cross-sectional means with past-only carry and fit fallback."""
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("distinct source stations required")
    output = {}
    for name in ("wind_u", "wind_v", "temperature", "humidity"):
        training = fit[name][:, sources]
        finite = training[np.isfinite(training)]
        if not len(finite):
            raise ValueError("no fitting-side public weather support")
        fallback = float(np.median(finite))
        values = current[name][:, sources]
        valid = np.isfinite(values)
        counts = valid.sum(axis=1)
        averages = np.divide(np.where(valid, values, 0).sum(axis=1), counts,
                             out=np.full(len(values), np.nan), where=counts > 0)
        for epoch in range(len(averages)):
            if not np.isfinite(averages[epoch]):
                averages[epoch] = averages[epoch - 1] if epoch else fallback
        output[name] = averages
    weather = PublicMeteorology(**output, source="UCI-Beijing-CMA-matched-observed-weather; RH-derived-from-TEMP-DEWP")
    weather.validate(len(output["wind_u"]))
    return weather


def public_field(values: np.ndarray, fit_values: np.ndarray, coordinates: np.ndarray,
                 sources: list[int], candidate: dict) -> np.ndarray:
    """Causal public interpolation, never reading non-source concentration columns."""
    values, fit_values = np.asarray(values, float), np.asarray(fit_values, float)
    if not sources or len(set(sources)) != len(sources) or values.shape[1] != len(coordinates):
        raise ValueError("invalid public interpolation sources")
    training = fit_values[:, sources]
    initial = float(np.median(training[np.isfinite(training)]))
    if not np.isfinite(initial) or initial < 0:
        raise ValueError("no valid fitting-side public initial value")
    distance = cdist(coordinates, coordinates)
    source_values = values[:, sources]
    valid = np.isfinite(source_values)
    safe_values = np.where(valid, source_values, 0.)
    result = np.empty_like(values)
    for target in range(values.shape[1]):
        # At known public nodes, interpolation excluding self supplies a fallback
        # only; a valid contemporaneous regulatory observation takes precedence.
        peers = [source for source in sources if source != target]
        if not peers:
            result[:, target] = initial
            continue
        weights = _adaptive_kernel_weights(distance, target, peers, kernel=candidate["kernel"],
                                           scale=candidate["scale"], neighbors=candidate["neighbors"])
        positions = [sources.index(source) for source in peers]
        denominator = valid[:, positions] @ weights
        prediction = np.divide(safe_values[:, positions] @ weights, denominator,
                               out=np.full(len(values), np.nan), where=denominator > 0)
        if target in sources:
            source_position = sources.index(target)
            prediction = np.where(valid[:, source_position], source_values[:, source_position], prediction)
        for epoch in range(len(prediction)):
            if not np.isfinite(prediction[epoch]):
                prediction[epoch] = prediction[epoch - 1] if epoch else initial
        result[:, target] = np.maximum(prediction, 0.)
    return result


def public_candidates(station_count: int) -> list[dict]:
    shapes = [("idw", 1.), ("idw", 2.), ("idw", 3.), ("gaussian", 10.),
              ("gaussian", 20.), ("gaussian", 40.), ("exponential", 20.), ("exponential", 40.)]
    return [{"kernel": kernel, "scale": scale, "neighbors": neighbors}
            for (kernel, scale), neighbors in itertools.product(shapes, (4, station_count - 1))]


def graph_candidates(protocol: dict) -> list[dict]:
    return [{"neighbors": neighbors, "length_scale_km": length, "correction_clip": clip}
            for neighbors, length, clip in itertools.product(protocol["graph_neighbors"],
                protocol["graph_length_scales_km"], protocol["correction_clips"])]


def run_weather_twin(values: np.ndarray, public_predictions: np.ndarray, coordinates: np.ndarray,
                     sources: list[int], weather: PublicMeteorology, graph_config: dict,
                     protocol: dict, *, identity_transition: bool = False) -> tuple[np.ndarray, dict]:
    """Same FixedLagTwin as the synthetic study; irregular public-only transfer.

    A zero private-channel sample does not test citizen robustness or DP utility.
    The radius still bounds the final graph correction around the public field.
    """
    values, baseline = np.asarray(values, float), np.asarray(public_predictions, float)
    if values.shape != baseline.shape or not np.isfinite(baseline).all() or np.any(baseline < 0):
        raise ValueError("invalid public transfer field")
    steps, nodes = baseline.shape
    weather.validate(steps)
    side = int(np.ceil(np.sqrt(nodes)))
    padding = side * side - nodes
    graph = city_graph(coordinates, neighbors=graph_config["neighbors"], length_scale=graph_config["length_scale_km"])
    laplacian = sparse.diags(np.asarray(graph.sum(axis=1)).ravel()) - graph
    padded_laplacian = sparse.block_diag((laplacian, sparse.csr_matrix((padding, padding))), format="csr")
    operators = (tuple(sparse.eye(nodes, format="csr") for _ in range(steps)) if identity_transition
                 else transition_series(graph, coordinates, weather, transport=protocol["transport"],
                                        directionality=protocol["directionality"]))
    transitions = tuple(sparse.block_diag((operator, sparse.eye(padding)), format="csr") for operator in operators)
    padded = np.pad(baseline, ((0, 0), (0, padding)))
    twin = FixedLagTwin(side=side, steps=steps, fixed_lag=protocol["fixed_lag"], huber=True,
        delta=protocol["citizen_delta"], lambda_prior=protocol["lambda_temporal"],
        lambda_spatial=protocol["lambda_spatial"], max_irls=2, tolerance=1e-6,
        initial_state=padded[0].copy(), predictive_residual=True,
        correction_delta=protocol["citizen_delta"], correction_clip=graph_config["correction_clip"],
        transitions=transitions, spatial_laplacian=padded_laplacian)
    live = np.empty_like(baseline)
    for epoch in range(steps):
        records = [Observation(source, epoch, source, 0, float(values[epoch, source]),
            protocol["regulatory_sigma_model"], 1., 0, f"public-{source}-{epoch}", epoch, epoch,
            source_class="regulatory") for source in sources if np.isfinite(values[epoch, source])]
        start = max(0, epoch - protocol["fixed_lag"])
        twin.ingest_at(epoch, records, external_predictor=padded[start : epoch + 1])
        live[epoch] = twin.states[epoch, :nodes]
    identity = sparse.eye(nodes, format="csr")
    return live, {"solver_failures": sum(not row.converged for row in twin.diagnostics),
                  "maximum_correction": float(np.max(np.abs(live - baseline))),
                  "mean_operator_distance_from_identity": float(np.mean([
                      sparse.linalg.norm(operator - identity) for operator in operators])),
                  "estimate_clock": "frozen-live-current-public-stations-and-current-observed-weather",
                  "protected_citizen_observations": 0, "padded_isolated_vertices": padding,
                  "reference_observations_reused_in_baseline_and_correction": True,
                  "reference_reuse_interpretation": "deterministic estimator; no independence or calibrated uncertainty claim"}


def paired_rmse(prediction, truth):
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not valid.any():
        raise ValueError("no target evaluation support")
    return float(np.sqrt(np.mean((prediction[valid] - truth[valid]) ** 2)))


def select_outer_fold(archive: BeijingArchive, outer_target: int, protocol: dict,
                      progress=None) -> dict:
    """Two-stage selection reads only outer training-side rows and stations."""
    spans = archive_slices(len(archive.times), protocol)
    fit, validation = archive.concentrations[spans["fit"]], archive.concentrations[spans["validation"]]
    available = [index for index in range(len(archive.stations)) if index != outer_target]
    fit_weather = {key: values[spans["fit"]] for key, values in archive.weather.items()}
    validation_weather = {key: values[spans["validation"]] for key, values in archive.weather.items()}
    baseline_candidate = {"kernel": "idw", "scale": 2., "neighbors": len(archive.stations) - 1}
    baseline_rmse = []
    weather_by_target = {}
    for target in available:
        sources = [index for index in available if index != target]
        prediction = public_field(validation, fit, archive.coordinates, sources, baseline_candidate)
        baseline_rmse.append(paired_rmse(prediction[:, target], validation[:, target]))
        weather_by_target[target] = public_weather(fit_weather, validation_weather, sources)
    baseline_rmse = np.asarray(baseline_rmse)
    public_ledger = []
    for candidate in public_candidates(len(archive.stations)):
        errors = []
        for target in available:
            sources = [index for index in available if index != target]
            prediction = public_field(validation, fit, archive.coordinates, sources, candidate)
            errors.append(paired_rmse(prediction[:, target], validation[:, target]))
        ratios = np.asarray(errors) / baseline_rmse
        public_ledger.append({"candidate": candidate, "station_rmse": errors,
            "eligible": bool(ratios.max() <= protocol["public_max_station_ratio"]),
            "score": float(np.mean(errors) + protocol["worst_station_weight"] * np.max(errors))})
    selected_public = min((row for row in public_ledger if row["eligible"]), key=lambda row: row["score"])
    if progress:
        progress({"stage": "public_selection", "outer_target": archive.stations[outer_target],
                  "selected": selected_public["candidate"]})
    fields = {}
    for target in available:
        fields[target] = public_field(validation, fit, archive.coordinates,
            [index for index in available if index != target], selected_public["candidate"])
    graph_ledger = []
    for candidate in graph_candidates(protocol):
        errors, failures, distances = [], 0, []
        for target in available:
            sources = [index for index in available if index != target]
            prediction, diagnostics = run_weather_twin(validation, fields[target], archive.coordinates,
                sources, weather_by_target[target], candidate, protocol)
            errors.append(paired_rmse(prediction[:, target], validation[:, target]))
            failures += diagnostics["solver_failures"]
            distances.append(diagnostics["mean_operator_distance_from_identity"])
        graph_ledger.append({"candidate": candidate, "station_rmse": errors,
            "solver_failures": failures, "mean_operator_distance_from_identity": float(np.mean(distances)),
            "mean_station_ratio_vs_idw": float(np.mean(np.asarray(errors) / baseline_rmse)),
            "score": float(np.mean(errors) + protocol["worst_station_weight"] * np.max(errors))})
        if progress:
            progress({"stage": "graph_candidate", "outer_target": archive.stations[outer_target],
                      "completed": len(graph_ledger), "total": len(graph_candidates(protocol))})
    successful = [row for row in graph_ledger if row["solver_failures"] == 0]
    if not successful:
        raise RuntimeError("all city graph candidates had numerical failures")
    selected_graph = min(successful, key=lambda row: (row["score"], -row["candidate"]["correction_clip"]))
    return {"outer_target": archive.stations[outer_target], "outer_target_index": outer_target,
            "inner_target_stations": [archive.stations[index] for index in available],
            "selected_public": selected_public["candidate"], "selected_graph": selected_graph["candidate"],
            "validation_rmse_ratio": selected_graph["mean_station_ratio_vs_idw"],
            "idw_station_rmse": baseline_rmse.tolist(), "public_ledger": public_ledger,
            "graph_ledger": graph_ledger, "outer_target_excluded_from_weather_and_concentrations": True,
            "test_labels_used_for_selection": False}


def evaluate_outer_fold(archive: BeijingArchive, selection: dict, protocol: dict) -> tuple[dict, dict]:
    target = selection["outer_target_index"]
    if archive.stations[target] != selection["outer_target"]:
        raise ValueError("selection/station order mismatch")
    spans = archive_slices(len(archive.times), protocol)
    fit, test = archive.concentrations[spans["fit"]], archive.concentrations[spans["test"]]
    sources = [index for index in range(len(archive.stations)) if index != target]
    weather = public_weather({key: value[spans["fit"]] for key, value in archive.weather.items()},
                             {key: value[spans["test"]] for key, value in archive.weather.items()}, sources)
    baseline = public_field(test, fit, archive.coordinates, sources, selection["selected_public"])
    idw = public_field(test, fit, archive.coordinates, sources,
                       {"kernel": "idw", "scale": 2., "neighbors": len(sources)})
    prediction, diagnostics = run_weather_twin(test, baseline, archive.coordinates, sources, weather,
                                                selection["selected_graph"], protocol)
    identity, identity_diagnostics = run_weather_twin(test, baseline, archive.coordinates, sources, weather,
        selection["selected_graph"], protocol, identity_transition=True)
    truth = test[:, target]
    support = np.isfinite(truth)
    errors = {}
    predictions = {"airproof_v4_city": prediction[:, target], "public_backbone": baseline[:, target],
                   "idw_frozen": idw[:, target], "airproof_identity_ablation": identity[:, target]}
    for method, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError("nonfinite predictions must not silently change target support")
        residual = values[support] - truth[support]
        errors[method] = {"rmse": float(np.sqrt(np.mean(residual**2))),
                          "mae": float(np.mean(np.abs(residual))), "bias": float(residual.mean()),
                          "support": int(support.sum())}
    return {"station": selection["outer_target"], "methods": errors, "diagnostics": diagnostics,
            "identity_diagnostics": identity_diagnostics}, {"truth": truth,
            **predictions, "valid": support, "utc_nanoseconds": archive.times[spans["test"]].asi8}

"""Frozen calibration from provenance-labelled public LOSO residuals only."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .records import Observation


@dataclass(frozen=True)
class PublicLOSOResidual:
    residual: float
    reference_age: float
    geometry_distance: float
    epoch: float
    prediction_available_at: float
    label_available_at: float
    latest_predictor_input_at: float
    station_id: str
    predictor_station_ids: tuple[str, ...]
    sample_id: str
    split_id: str
    public_model_id: str
    role: str = "training"
    source_class: str = "public_regulatory"


@dataclass(frozen=True)
class ContextSummary:
    count: int
    scale: float
    radii: tuple[float, float]


@dataclass(frozen=True)
class PublicContextPrediction:
    innovation_scales: np.ndarray
    radii: np.ndarray
    pooled_fallback: np.ndarray
    calibration_counts: np.ndarray


@dataclass(frozen=True)
class PublicContextCalibrator:
    training_end: float
    latest_training_epoch: float
    training_split_id: str
    public_model_id: str
    age_edges: tuple[float, ...]
    geometry_edges: tuple[float, ...]
    pooled: ContextSummary
    strata: tuple[tuple[tuple[int, int], ContextSummary], ...]
    minimum_stratum_size: int
    sample_ids: tuple[str, ...]

    def predict(self, reference_age, geometry_distance, *, prediction_epoch,
                available_at, context_available_at) -> PublicContextPrediction:
        """Broadcast public context; no target/residual/observation argument exists."""
        age, geometry, epoch, available, context_time = np.broadcast_arrays(
            *[np.asarray(x, float) for x in (reference_age, geometry_distance,
              prediction_epoch, available_at, context_available_at)])
        if (not all(np.isfinite(x).all() for x in (age, geometry, epoch, available, context_time))
                or (age < 0).any() or (geometry < 0).any()
                or (available < self.training_end).any()
                or (epoch <= self.latest_training_epoch).any()
                or (context_time > available).any() or (available > epoch).any()):
            raise ValueError("prediction must follow training and use available public context")
        scales = np.full(age.shape, self.pooled.scale)
        radii = np.broadcast_to(self.pooled.radii, (*age.shape, 2)).copy()
        fallback = np.ones(age.shape, dtype=bool)
        counts = np.full(age.shape, self.pooled.count, dtype=int)
        age_bin = np.searchsorted(self.age_edges, age, side="right")
        geometry_bin = np.searchsorted(self.geometry_edges, geometry, side="right")
        for (a, g), summary in self.strata:
            mask = (age_bin == a) & (geometry_bin == g)
            scales[mask] = summary.scale
            radii[mask] = summary.radii
            fallback[mask] = False
            counts[mask] = summary.count
        for output in (scales, radii, fallback, counts):
            output.setflags(write=False)
        return PublicContextPrediction(scales, radii, fallback, counts)


def fit_public_context_calibrator(records, *, training_end: float, training_split_id: str,
                                  public_model_id: str, age_edges=(1., 6., 24.),
                                  geometry_edges=(.1, .25, .5), minimum_stratum_size=100,
                                  scale_floor=1e-3) -> PublicContextCalibrator:
    """Fit locked bins; public_model_id identifies a previously frozen predictor.

    LOSO predictions must exclude the held station from all predictor inputs. Metadata
    guards are enforced here; authentication and truthful upstream provenance are the
    caller's responsibility. No candidate model or bin selection is performed.
    """
    rows = tuple(records)
    age_edges, geometry_edges = tuple(age_edges), tuple(geometry_edges)
    if (not np.isfinite((training_end, scale_floor)).all() or scale_floor <= 0
            or not isinstance(minimum_stratum_size, int) or minimum_stratum_size < 100
            or not training_split_id or not public_model_id or len(rows) < 20):
        raise ValueError("valid frozen calibration metadata and at least 20 public rows required")
    for edges in (age_edges, geometry_edges):
        if not np.isfinite(edges).all() or any(e < 0 for e in edges) or any(
                a >= b for a, b in pairwise(edges)):
            raise ValueError("context edges must be finite, nonnegative and increasing")
    ids, station_epochs, groups = set(), set(), {}
    for row in rows:
        if not isinstance(row, PublicLOSOResidual):
            raise TypeError("only explicit public LOSO records accepted")
        numeric = (row.residual, row.reference_age, row.geometry_distance, row.epoch,
                   row.prediction_available_at, row.label_available_at, row.latest_predictor_input_at)
        if (not np.isfinite(numeric).all() or min(row.reference_age, row.geometry_distance) < 0
                or row.role != "training" or row.source_class != "public_regulatory"
                or row.split_id != training_split_id or row.public_model_id != public_model_id
                or not row.station_id or not row.sample_id or not row.predictor_station_ids
                or row.station_id in row.predictor_station_ids
                or row.latest_predictor_input_at > row.prediction_available_at
                or row.prediction_available_at > row.epoch
                or row.epoch > row.label_available_at or row.label_available_at > training_end
                or row.sample_id in ids or (row.station_id, row.epoch) in station_epochs):
            raise ValueError("invalid/duplicate public training LOSO provenance or future leakage")
        ids.add(row.sample_id)
        station_epochs.add((row.station_id, row.epoch))
        key = (int(np.searchsorted(age_edges, row.reference_age, side="right")),
               int(np.searchsorted(geometry_edges, row.geometry_distance, side="right")))
        groups.setdefault(key, []).append(row.residual)

    def summarize(errors):
        absolute = np.sort(np.abs(np.asarray(errors, float)))
        # Stable RMS includes public bias; no bias correction/model change is applied.
        maximum = float(absolute[-1])
        rms = maximum*np.sqrt(np.mean((absolute/maximum)**2)) if maximum else 0.
        radii = tuple(float(absolute[min(int(np.ceil((len(absolute)+1)*level))-1,
                                        len(absolute)-1)]) for level in (.9, .95))
        return ContextSummary(len(absolute), max(scale_floor, float(rms)), radii)

    return PublicContextCalibrator(float(training_end), max(row.epoch for row in rows),
        training_split_id, public_model_id, age_edges, geometry_edges,
        summarize([row.residual for row in rows]),
        tuple((key, summarize(values)) for key, values in sorted(groups.items())
              if len(values) >= minimum_stratum_size), minimum_stratum_size, tuple(sorted(ids)))


def produce_public_loso_residuals(records, coordinates, *, training_end, training_split_id,
                                  public_model_id="fixed-idw-power2", distance_floor=1e-6):
    """Causal fixed-IDW LOSO producer; no candidate/model selection.

    Each regulatory cell is one station. All history at the held station is excluded.
    Latest available measurements at other stations are interpolated by inverse
    squared distance. This explicit adapter is not a reproduction of the tuned v5
    backbone; callers must give every comparator its matching IDW public field.
    Missing support is skipped and reported, never silently imputed from the future.
    """
    coords = np.asarray(coordinates, float)
    if (coords.ndim != 2 or coords.shape[0] < 2 or not np.isfinite(coords).all()
            or not np.isfinite((training_end, distance_floor)).all() or distance_floor <= 0
            or not training_split_id or public_model_id != "fixed-idw-power2"):
        raise ValueError("valid coordinates and fixed-IDW public model required")
    items = tuple(records)
    identities = set()
    for item in items:
        if not isinstance(item, Observation):
            raise TypeError("regulatory Observation inputs required")
        key = (item.cell, item.epoch)
        if (item.source_class != "regulatory" or not 0 <= item.cell < len(coords)
                or not np.isfinite(item.value) or item.direct_arrival is None
                or item.direct_arrival < item.epoch or item.epoch < 0 or key in identities):
            raise ValueError("unique authenticated regulatory station/epoch inputs required")
        identities.add(key)
    training = sorted((item for item in items if item.direct_arrival <= training_end),
                      key=lambda item: (item.epoch, item.cell))
    rows, missing = [], 0
    for held in training:
        # latest *acquisition* among available values; delayed old records never
        # replace a newer reference at the same station.
        available = {}
        for other in training:
            if other.cell != held.cell and other.direct_arrival <= held.epoch:
                previous = available.get(other.cell)
                if previous is None or previous.epoch < other.epoch:
                    available[other.cell] = other
        if not available:
            missing += 1
            continue
        support = [available[cell] for cell in sorted(available)]
        distances = np.linalg.norm(coords[[item.cell for item in support]]-coords[held.cell], axis=1)
        weights = 1/np.maximum(distances, distance_floor)**2
        weights /= weights.sum()
        prediction = float(weights @ np.array([item.value for item in support]))
        age = float(weights @ np.array([held.epoch-item.epoch for item in support]))
        rows.append(PublicLOSOResidual(
            held.value-prediction, age, float(distances.min()), float(held.epoch),
            float(held.epoch), float(held.direct_arrival),
            float(max(item.direct_arrival for item in support)), str(held.cell),
            tuple(str(item.cell) for item in support), f"station{held.cell}:epoch{held.epoch}",
            training_split_id, public_model_id))
    return tuple(rows), {"input_rows": len(items), "available_training_rows": len(training),
                         "produced_rows": len(rows), "missing_loso_support": missing,
                         "excluded_after_training_end": len(items)-len(training),
                         "model": public_model_id, "distance_floor": distance_floor}


def produce_frozen_backbone_loso(records, *, side, steps, training_end, training_split_id,
                                 public_model_id, selected, transitions=None, initial=12.,
                                 smoothing=.001):
    """LOSO residuals of the exact frozen v4 kernel/temporal recurrence.

    Selection is external and never repeated. If selected used these same rows,
    residuals are post-selection development proxies, not independent calibration.
    Requires the complete on-time regulatory prefix used by the existing backbone.
    """
    from .meteorology import grid_coordinates
    from .reference import _reference_weights

    if (not isinstance(steps, int) or steps < 2 or not isinstance(side, int) or side < 2
            or training_end < steps-1 or not training_split_id or not public_model_id
            or not np.isfinite((training_end, initial, smoothing)).all() or smoothing < 0
            or initial < 0):
        raise ValueError("valid complete training prefix and frozen model metadata required")
    kernel, epsilon = str(selected["kernel"]), float(selected["epsilon"])
    temporal = float(selected["temporal_weight"])
    if (kernel not in ("thin_plate_spline", "gaussian") or not np.isfinite(epsilon)
            or epsilon <= 0 or not 0 <= temporal <= 1
            or (transitions is None and temporal != 0)):
        raise ValueError("invalid frozen backbone selection or missing transitions")
    if transitions is not None and (len(transitions) < steps or any(
            op.shape != (side*side, side*side) for op in transitions[:steps])):
        raise ValueError("training-prefix transition dimensions mismatch")
    items = tuple(records)
    grouped = {}
    for item in items:
        if (item.source_class != "regulatory" or item.direct_arrival != item.epoch
                or not 0 <= item.cell < side*side or item.epoch < 0 or not np.isfinite(item.value)):
            raise ValueError("on-time public regulatory records required")
        if item.epoch < steps:
            grouped.setdefault((item.epoch, item.cell), []).append(item.value)
    stations = tuple(sorted({cell for _, cell in grouped}))
    if len(stations) < 4 or any((epoch, cell) not in grouped
                               for epoch in range(steps) for cell in stations):
        raise ValueError("complete prefix with at least four public stations required")
    values = np.array([[np.mean(grouped[epoch, cell]) for cell in stations]
                       for epoch in range(steps)])
    coords, rows = grid_coordinates(side), []
    for held_index, held_cell in enumerate(stations):
        keep = [i for i in range(len(stations)) if i != held_index]
        held_cells = tuple(stations[i] for i in keep)
        interpolated = np.maximum(values[:, keep] @ _reference_weights(
            side, held_cells, smoothing, kernel, epsilon).T, 0)
        previous = np.full(side*side, initial)
        nearest = float(np.linalg.norm(coords[list(held_cells)]-coords[held_cell], axis=1).min())
        for epoch in range(steps):
            forecast = previous if transitions is None else transitions[epoch] @ previous
            previous = (1-temporal)*interpolated[epoch] + temporal*forecast
            if epoch:  # Match the v4 LOSO exclusion of initialization.
                rows.append(PublicLOSOResidual(
                    float(values[epoch, held_index]-previous[held_cell]), 0., nearest,
                    float(epoch), float(epoch), float(epoch), float(epoch), str(held_cell),
                    tuple(map(str, held_cells)), f"station{held_cell}:epoch{epoch}",
                    training_split_id, public_model_id))
    return tuple(rows), {
        "model": public_model_id, "selected": dict(selected), "produced_rows": len(rows),
        "public_stations": len(stations), "prefix_steps": steps,
        "selection_refitted": False, "whole_held_station_excluded": True,
        "scale_scope": "public post-selection LOSO proxy; not citizen-total innovation covariance",
        "independent_interval_calibration": False,
    }


def frozen_backbone_scale_context(records, *, side, calibration_steps, horizon, selected,
                                   transitions=None, training_split_id, public_model_id,
                                   initial=12., smoothing=.001):
    """Matched-backbone producer + fit + per-cell frozen scales for main adapters.

    Returned arrays include a prefix solely for shape compatibility: never score or
    assimilate using fitted scales before calibration_steps. Deployment entries are
    produced through the calibrator's causal boundary guards.
    """
    from .meteorology import grid_coordinates

    if horizon <= calibration_steps:
        raise ValueError("horizon must extend past public calibration prefix")
    items = tuple(records)
    rows, provenance = produce_frozen_backbone_loso(items, side=side, steps=calibration_steps,
        training_end=calibration_steps-1, training_split_id=training_split_id,
        public_model_id=public_model_id, selected=selected, transitions=transitions,
        initial=initial, smoothing=smoothing)
    calibrator = fit_public_context_calibrator(rows, training_end=calibration_steps-1,
        training_split_id=training_split_id, public_model_id=public_model_id)
    # A target station's LOSO error is calibrated against nearest *other* station.
    # Grid deployment uses nearest available station; this transfer is a proxy and
    # needs held-out validation, especially at station locations/extrapolation cells.
    station_cells = sorted({o.cell for o in items if o.epoch < calibration_steps})
    present = {(o.epoch, o.cell) for o in items}
    if any((epoch, cell) not in present for epoch in range(horizon) for cell in station_cells):
        raise ValueError("age-zero scale adapter requires complete on-time reference horizon")
    coords = grid_coordinates(side)
    distances = np.linalg.norm(coords[:, None, :]-coords[station_cells][None, :, :], axis=2).min(axis=1)
    deployment = calibrator.predict(np.zeros((horizon-calibration_steps, 1)), distances[None, :],
        prediction_epoch=np.arange(calibration_steps, horizon)[:, None],
        available_at=np.arange(calibration_steps, horizon)[:, None],
        context_available_at=np.arange(calibration_steps, horizon)[:, None])
    scales = np.full((horizon, side*side), calibrator.pooled.scale)
    scales[calibration_steps:] = deployment.innovation_scales
    return scales, {**provenance, "first_valid_epoch": calibration_steps,
                    "training_split_id": training_split_id,
                    "calibration_count": calibrator.pooled.count,
                    "pooled_scale": calibrator.pooled.scale,
                    "fitted_strata": len(calibrator.strata),
                    "deployment_context": "on-time age0 and nearest public station distance",
                    "prefix_values": "shape placeholders; prohibited for scoring/fitted inference"}

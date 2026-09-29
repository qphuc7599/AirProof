"""Citizen-independent, causal public reference fields and public-only calibration."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator

from .records import Observation


@lru_cache(maxsize=32)
def _reference_weights(side: int, cells: tuple[int, ...], smoothing: float,
                       kernel: str = "thin_plate_spline", epsilon: float = 1.0) -> np.ndarray:
    coordinates = np.indices((side, side)).reshape(2, -1).T / (side - 1)
    observed = coordinates[list(cells)]
    # An affine thin-plate interpolant requires three non-collinear reference cells.
    design = np.column_stack([np.ones(len(cells)), observed])
    if kernel == "gaussian" or (len(cells) >= 3 and np.linalg.matrix_rank(design) == 3):
        return RBFInterpolator(
            observed, np.eye(len(cells)), kernel=kernel, smoothing=smoothing,
            epsilon=epsilon, degree=0 if kernel == "gaussian" else 1,
        )(coordinates)
    distances = np.linalg.norm(coordinates[:, None, :] - observed[None, :, :], axis=2)
    weights = 1.0 / np.maximum(distances, 1e-8) ** 2
    return weights / weights.sum(axis=1, keepdims=True)


def public_reference_fields(
    records: Iterable[Observation],
    *,
    side: int,
    steps: int,
    initial: float = 12.0,
    smoothing: float = 0.001,
    kernel: str = "thin_plate_spline",
    epsilon: float = 1.0,
) -> np.ndarray:
    """Interpolate each public-reference epoch independently; never use citizen data.

    This diagnostic interface requires on-time regulatory inputs. A missing epoch
    carries the preceding *public* field forward. It neither smooths backwards nor
    fills an earlier gap with a later observation. The caller must authenticate the
    regulatory source label before invoking this numerical routine.
    """
    if side < 2 or steps < 1 or not np.isfinite(initial) or initial < 0:
        raise ValueError("invalid public-reference field dimensions or initial value")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("reference smoothing must be finite and non-negative")
    if kernel not in {"thin_plate_spline", "gaussian"} or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid reference interpolation kernel or scale")
    epochs: dict[int, dict[int, list[float]]] = {}
    for item in records:
        if item.source_class != "regulatory":
            raise ValueError("public reference fields accept only authenticated regulatory data")
        if item.direct_arrival != item.epoch:
            raise ValueError("public reference diagnostic requires on-time regulatory data")
        if not 0 <= item.epoch < steps or not 0 <= item.cell < side * side:
            raise ValueError("reference observation is outside the field domain")
        if not np.isfinite(item.value):
            raise ValueError("reference observation must be finite")
        epochs.setdefault(item.epoch, {}).setdefault(item.cell, []).append(float(item.value))
    result = np.full((steps, side * side), initial, dtype=float)
    for epoch in range(steps):
        grouped = epochs.get(epoch)
        if grouped:
            cells = tuple(sorted(grouped))
            values = np.asarray([np.mean(grouped[cell]) for cell in cells])
            result[epoch] = np.maximum(
                _reference_weights(side, cells, smoothing, kernel, epsilon) @ values, 0.0
            )
        elif epoch:
            result[epoch] = result[epoch - 1]
    if not np.all(np.isfinite(result)):
        raise ValueError("public reference interpolation produced a non-finite field")
    return result


@dataclass(frozen=True)
class PublicReferenceBackbone:
    fields: np.ndarray
    diagnostics: dict


def calibrated_public_reference(records: Iterable[Observation], *, side: int, steps: int,
                                initial: float = 12, calibration_epochs: int = 48,
                                smoothing: float = .001, transitions=None,
                                kernels: tuple[tuple[str, float], ...] = (
                                    ("thin_plate_spline", 1.), ("gaussian", 2.),
                                    ("gaussian", 3.), ("gaussian", 4.),
                                ), temporal_weights: tuple[float, ...] = (0., .1, .25),
                                ) -> PublicReferenceBackbone:
    """LOSO on a completed public training prefix; deployment strictly afterwards.

    Before prefix closure use the untuned causal thin-plate predictor. Calibration
    never revises those fields. At each post-prefix epoch only current/past public
    references and current observed weather enter the prediction. The caller must
    exclude the calibration prefix from confirmatory scoring.
    """
    items = tuple(records)
    default = public_reference_fields(items, side=side, steps=steps, initial=initial,
                                      smoothing=smoothing)
    if not 2 <= calibration_epochs < steps:
        raise ValueError("public calibration prefix must be >=2 and shorter than horizon")
    if not kernels or not temporal_weights or any(not 0 <= v <= 1 for v in temporal_weights):
        raise ValueError("invalid public calibration candidates")
    if transitions is not None and len(transitions) != steps:
        raise ValueError("transition horizon differs from reference horizon")
    if transitions is None:
        temporal_weights = (0.,)
    train = [item for item in items if item.epoch < calibration_epochs]
    cells = tuple(sorted({item.cell for item in train}))
    if len(cells) < 4:
        return PublicReferenceBackbone(default, {
            "calibration": "insufficient-public-stations-default-thin-plate",
            "calibration_epochs": calibration_epochs, "citizen_independent": True,
        })
    values = np.full((calibration_epochs, len(cells)), np.nan)
    grouped = {}
    for item in train:
        grouped.setdefault((item.epoch, item.cell), []).append(item.value)
    for index, cell in enumerate(cells):
        for epoch in range(calibration_epochs):
            if (epoch, cell) in grouped:
                values[epoch, index] = np.mean(grouped[epoch, cell])
    # A varying station set still gets causal predictions, but cannot be quietly
    # imputed using future held-out measurements for hyperparameter selection.
    if not np.all(np.isfinite(values)):
        return PublicReferenceBackbone(default, {
            "calibration": "incomplete-public-prefix-default-thin-plate",
            "calibration_epochs": calibration_epochs, "citizen_independent": True,
        })
    candidates = []
    for kernel, epsilon in kernels:
        squared_errors = {weight: [] for weight in temporal_weights}
        for held in range(len(cells)):
            keep = [index for index in range(len(cells)) if index != held]
            held_cells = tuple(cells[index] for index in keep)
            interpolated = np.maximum(
                values[:, keep] @ _reference_weights(
                    side, held_cells, smoothing, kernel, epsilon
                ).T, 0,
            )
            for weight in temporal_weights:
                previous = np.full(side * side, initial)
                errors = []
                for epoch in range(calibration_epochs):
                    forecast = previous if transitions is None else transitions[epoch] @ previous
                    previous = (1 - weight) * interpolated[epoch] + weight * forecast
                    if epoch:
                        errors.append(float(previous[cells[held]] - values[epoch, held]) ** 2)
                squared_errors[weight].extend(errors)
        for weight, errors in squared_errors.items():
            candidates.append({"kernel": kernel, "epsilon": epsilon, "temporal_weight": weight,
                               "public_loso_mse": float(np.mean(errors))})
    selected = min(candidates, key=lambda candidate: candidate["public_loso_mse"])
    calibrated = public_reference_fields(items, side=side, steps=steps, initial=initial,
                                          smoothing=smoothing, kernel=selected["kernel"],
                                          epsilon=selected["epsilon"])
    fields = default.copy()
    present_epochs = {item.epoch for item in items}
    for epoch in range(calibration_epochs, steps):
        forecast = fields[epoch - 1] if transitions is None else transitions[epoch] @ fields[epoch - 1]
        weight = selected["temporal_weight"]
        fields[epoch] = (np.maximum((1 - weight) * calibrated[epoch] + weight * forecast, 0)
                         if epoch in present_epochs else np.maximum(forecast, 0))
    return PublicReferenceBackbone(fields, {
        "calibration": "completed-public-prefix-leave-one-station-out",
        "calibration_epochs": calibration_epochs, "station_count": len(cells),
        "selected": selected, "candidates": candidates, "citizen_independent": True,
        "first_selected_model_epoch": calibration_epochs,
    })

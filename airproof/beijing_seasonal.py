"""Season-balanced public calibration; the v4 solver and earlier evidence stay fixed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .beijing_v4 import (BeijingArchive, graph_candidates, paired_rmse,
    public_candidates, public_field, public_weather, run_weather_twin)


def seasonal_spans(archive: BeijingArchive, protocol: dict) -> dict:
    times = archive.times
    cutoff = int(len(times) * protocol["outer_train_fraction"])

    def index(value):
        timestamp = pd.Timestamp(value, tz=times.tz)
        result = int(times.searchsorted(timestamp))
        if result >= len(times) or times[result] != timestamp:
            raise ValueError("registered calendar boundary absent from hourly archive")
        return result

    seasons = []
    for date in protocol["validation_starts"]:
        start = index(date)
        stop = start + protocol["inner_validation_steps"]
        fit_end = start - protocol["purge_hours"]
        fit_start = fit_end - protocol["inner_fit_steps"]
        if fit_start < 0 or stop > cutoff:
            raise ValueError("seasonal calibration exceeds the outer training side")
        seasons.append({"fit": slice(fit_start, fit_end), "validation": slice(start, stop)})
    if len({row["validation"].start for row in seasons}) != len(seasons):
        raise ValueError("duplicate validation season")
    ordered = sorted((row["validation"] for row in seasons), key=lambda span: span.start)
    if any(left.stop > right.start for left, right in zip(ordered, ordered[1:])):
        raise ValueError("overlapping validation windows")
    test = slice(index(protocol["test_start_inclusive"]), index(protocol["test_stop_exclusive"]))
    if test.start <= cutoff + protocol["purge_hours"] or test.stop - test.start != protocol["test_steps"]:
        raise ValueError("invalid independent calendar test partition")
    for start, stop in protocol["previous_test_windows"]:
        if max(test.start, start) < min(test.stop, stop):
            raise ValueError("new test overlaps previously inspected outcomes")
    return {"seasons": seasons, "test": test,
            "deployment_fit": slice(cutoff - protocol["inner_fit_steps"], cutoff),
            "outer_training_stop_exclusive": cutoff}


def anchor_candidate(stations: int) -> dict:
    return {"kernel": "idw", "scale": 2., "neighbors": stations - 1}


def anchored_public_field(values, fit, coordinates, sources, candidate):
    anchor = public_field(values, fit, coordinates, sources, anchor_candidate(len(coordinates)))
    weight = candidate["mix_weight"]
    if not 0 <= weight <= 1:
        raise ValueError("public mixture must be convex")
    if weight == 0:
        return anchor
    alternative = public_field(values, fit, coordinates, sources, candidate["kernel_candidate"])
    return (1 - weight) * anchor + weight * alternative


def normalized_score(errors, reference, weight):
    errors, reference = np.asarray(errors, float), np.asarray(reference, float)
    if (errors.shape != reference.shape or not np.isfinite(errors).all()
            or not np.isfinite(reference).all() or np.any(reference <= 0)):
        raise ValueError("positive finite calibration reference required")
    ratios = errors / reference
    return float(ratios.mean() + weight * ratios.max()), float(ratios.max())


def select_seasonal_fold(archive: BeijingArchive, outer_target: int, protocol: dict) -> dict:
    spans = seasonal_spans(archive, protocol)
    available = [index for index in range(len(archive.stations)) if index != outer_target]
    calibration = []
    for season, span in enumerate(spans["seasons"]):
        fit, values = archive.concentrations[span["fit"]], archive.concentrations[span["validation"]]
        fit_weather = {name: values[span["fit"]] for name, values in archive.weather.items()}
        now_weather = {name: values[span["validation"]] for name, values in archive.weather.items()}
        for target in available:
            sources = [index for index in available if index != target]
            anchor = public_field(values, fit, archive.coordinates, sources, anchor_candidate(len(archive.stations)))
            calibration.append({"season": season, "target": target, "sources": sources,
                "values": values, "fit": fit, "anchor": anchor,
                "weather": public_weather(fit_weather, now_weather, sources),
                "reference_rmse": paired_rmse(anchor[:, target], values[:, target])})
    references = [row["reference_rmse"] for row in calibration]
    public_ledger = [{"candidate": {"kernel_candidate": anchor_candidate(len(archive.stations)), "mix_weight": 0.},
        "season_station_rmse": references, "eligible": True, "maximum_ratio": 1.,
        "score": 1. + protocol["worst_station_weight"]}]
    for kernel in public_candidates(len(archive.stations)):
        if kernel == anchor_candidate(len(archive.stations)):
            continue
        alternatives = [public_field(row["values"], row["fit"], archive.coordinates, row["sources"], kernel)
                        for row in calibration]
        for weight in protocol["public_mix_weights"]:
            errors = [paired_rmse(((1 - weight) * row["anchor"] + weight * alternative)[:, row["target"]],
                                   row["values"][:, row["target"]])
                      for row, alternative in zip(calibration, alternatives, strict=True)]
            score, maximum = normalized_score(errors, references, protocol["worst_station_weight"])
            public_ledger.append({"candidate": {"kernel_candidate": kernel, "mix_weight": weight},
                "season_station_rmse": errors, "score": score, "maximum_ratio": maximum,
                "eligible": maximum <= protocol["public_max_station_ratio"]})
    selected_public = min((row for row in public_ledger if row["eligible"]),
                          key=lambda row: (row["score"], row["candidate"]["mix_weight"]))
    fields = [anchored_public_field(row["values"], row["fit"], archive.coordinates,
                                    row["sources"], selected_public["candidate"]) for row in calibration]
    graph_ledger = []
    for candidate in graph_candidates(protocol):
        errors, failures, distances = [], 0, []
        for row, field in zip(calibration, fields, strict=True):
            prediction, diagnostics = run_weather_twin(row["values"], field, archive.coordinates,
                row["sources"], row["weather"], candidate, protocol)
            errors.append(paired_rmse(prediction[:, row["target"]], row["values"][:, row["target"]]))
            failures += diagnostics["solver_failures"]
            distances.append(diagnostics["mean_operator_distance_from_identity"])
        score, maximum = normalized_score(errors, references, protocol["worst_station_weight"])
        graph_ledger.append({"candidate": candidate, "season_station_rmse": errors,
            "solver_failures": failures, "score": score, "maximum_ratio": maximum,
            "mean_operator_distance_from_identity": float(np.mean(distances))})
    valid = [row for row in graph_ledger if row["solver_failures"] == 0]
    if not valid:
        raise RuntimeError("no numerically valid seasonal city graph")
    selected_graph = min(valid, key=lambda row: (row["score"], -row["candidate"]["correction_clip"]))
    return {"outer_target": archive.stations[outer_target], "outer_target_index": outer_target,
        "selected_public": selected_public["candidate"], "selected_graph": selected_graph["candidate"],
        "calibration_order": [{"season": row["season"], "station": archive.stations[row["target"]]}
                              for row in calibration],
        "idw_season_station_rmse": references, "public_ledger": public_ledger, "graph_ledger": graph_ledger,
        "outer_target_excluded_from_weather_and_concentrations": True,
        "test_labels_used_for_selection": False}


def evaluate_seasonal_fold(archive: BeijingArchive, selection: dict, protocol: dict):
    target = selection["outer_target_index"]
    if archive.stations[target] != selection["outer_target"]:
        raise ValueError("locked target order mismatch")
    spans = seasonal_spans(archive, protocol)
    fit, test = archive.concentrations[spans["deployment_fit"]], archive.concentrations[spans["test"]]
    sources = [index for index in range(len(archive.stations)) if index != target]
    weather = public_weather({key: value[spans["deployment_fit"]] for key, value in archive.weather.items()},
                             {key: value[spans["test"]] for key, value in archive.weather.items()}, sources)
    baseline = anchored_public_field(test, fit, archive.coordinates, sources, selection["selected_public"])
    idw = public_field(test, fit, archive.coordinates, sources, anchor_candidate(len(archive.stations)))
    prediction, diagnostics = run_weather_twin(test, baseline, archive.coordinates, sources, weather,
                                              selection["selected_graph"], protocol)
    identity, identity_diagnostics = run_weather_twin(test, baseline, archive.coordinates, sources, weather,
        selection["selected_graph"], protocol, identity_transition=True)
    truth = test[:, target]
    valid = np.isfinite(truth)
    predictions = {"airproof_v4_city": prediction[:, target], "public_backbone": baseline[:, target],
                   "idw_frozen": idw[:, target], "airproof_identity_ablation": identity[:, target]}
    if not valid.any() or any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("invalid common outer-test support")
    errors = {}
    for name, values in predictions.items():
        residual = values[valid] - truth[valid]
        errors[name] = {"rmse": float(np.sqrt(np.mean(residual**2))),
                       "mae": float(np.mean(np.abs(residual))), "bias": float(residual.mean()),
                       "support": int(valid.sum())}
    return {"station": selection["outer_target"], "methods": errors, "diagnostics": diagnostics,
            "identity_diagnostics": identity_diagnostics}, {
        "truth": truth, **predictions, "valid": valid, "utc_nanoseconds": archive.times[spans["test"]].asi8}

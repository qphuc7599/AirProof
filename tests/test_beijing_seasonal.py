from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from airproof.beijing_seasonal import (anchored_public_field, evaluate_seasonal_fold,
    normalized_score, seasonal_spans, select_seasonal_fold)
from airproof.beijing_v4 import BeijingArchive


def fixture():
    rng = np.random.default_rng(1916)
    values = np.maximum(0, 20 + np.arange(4)[None, :] + rng.normal(0, 2, (40, 4)))
    weather = {name: np.full_like(values, scalar) for name, scalar in
               (("wind_u", 5.), ("wind_v", 0.), ("temperature", 22.), ("humidity", 65.))}
    archive = BeijingArchive(values, weather,
        pd.date_range("2020-01-01", periods=40, freq="h", tz="Asia/Shanghai"),
        tuple("ABCD"), np.array([[0., 0.], [0., 10.], [10., 0.], [10., 10.]]))
    protocol = {"outer_train_fraction": .5, "inner_fit_steps": 6, "inner_validation_steps": 4,
        "purge_hours": 1, "validation_starts": ["2020-01-01 08:00", "2020-01-01 14:00"],
        "test_start_inclusive": "2020-01-02 03:00", "test_stop_exclusive": "2020-01-02 09:00",
        "test_steps": 6, "previous_test_windows": [[21, 23], [37, 40]],
        "public_mix_weights": [.25, 1.], "public_max_station_ratio": 1.05,
        "graph_neighbors": [2], "graph_length_scales_km": [15.], "correction_clips": [2.],
        "worst_station_weight": .25, "fixed_lag": 2, "citizen_delta": 4.,
        "lambda_temporal": .02, "lambda_spatial": .02, "regulatory_sigma_model": 3.,
        "transport": .12, "directionality": 1.}
    return archive, protocol


def test_calendar_partitions_exclude_all_previous_test_windows():
    archive, protocol = fixture()
    spans = seasonal_spans(archive, protocol)
    assert spans["test"] == slice(27, 33)
    assert all(row["fit"].stop + 1 == row["validation"].start for row in spans["seasons"])
    with pytest.raises(ValueError, match="overlaps previously"):
        seasonal_spans(archive, {**protocol, "previous_test_windows": [[30, 35]]})
    with pytest.raises(ValueError, match="outer training side"):
        seasonal_spans(archive, {**protocol, "validation_starts": ["2020-01-01 18:00"]})


def test_selection_ignores_outer_station_and_every_post_training_value():
    archive, protocol = fixture()
    expected = select_seasonal_fold(archive, 3, protocol)
    values = archive.concentrations.copy()
    values[:, 3] = 999999
    values[20:] = 222222
    weather = {name: array.copy() for name, array in archive.weather.items()}
    for array in weather.values():
        array[:, 3] = 999999
        array[20:] = 888888
    actual = select_seasonal_fold(replace(archive, concentrations=values, weather=weather), 3, protocol)
    assert actual == expected
    assert len(expected["calibration_order"]) == 6
    assert expected["test_labels_used_for_selection"] is False


def test_mixture_is_convex_source_only_and_causal():
    archive, _ = fixture()
    candidate = {"kernel_candidate": {"kernel": "gaussian", "scale": 20., "neighbors": 3},
                 "mix_weight": .5}
    values, fit = archive.concentrations[:8].copy(), archive.concentrations[8:14].copy()
    expected = anchored_public_field(values, fit, archive.coordinates, [0, 1, 2], candidate)
    values[:, 3] = 999999
    fit[:, 3] = 999999
    values[4:, :3] = 333333
    actual = anchored_public_field(values, fit, archive.coordinates, [0, 1, 2], candidate)
    np.testing.assert_array_equal(actual[:4], expected[:4])
    with pytest.raises(ValueError, match="convex"):
        anchored_public_field(values, fit, archive.coordinates, [0, 1, 2], {**candidate, "mix_weight": 2})


def test_long_test_uses_same_active_production_solver_and_excludes_target_inputs():
    archive, protocol = fixture()
    selected = {"outer_target": "D", "outer_target_index": 3,
        "selected_public": {"kernel_candidate": {"kernel": "gaussian", "scale": 20., "neighbors": 3},
                            "mix_weight": .25},
        "selected_graph": {"neighbors": 2, "length_scale_km": 15., "correction_clip": 2.}}
    row, arrays = evaluate_seasonal_fold(archive, selected, protocol)
    assert row["diagnostics"]["solver_failures"] == 0
    assert row["diagnostics"]["mean_operator_distance_from_identity"] > 0
    assert row["diagnostics"]["maximum_correction"] <= 2 + 1e-10
    values = archive.concentrations.copy()
    values[:, 3] = 999999
    _, changed = evaluate_seasonal_fold(replace(archive, concentrations=values), selected, protocol)
    for name in ("airproof_v4_city", "public_backbone", "idw_frozen", "airproof_identity_ablation"):
        np.testing.assert_array_equal(arrays[name], changed[name])


def test_normalized_selection_scores_each_season_station_pair():
    score, maximum = normalized_score([9., 110.], [10., 100.], .25)
    assert np.isclose(score, 1.275) and np.isclose(maximum, 1.1)
    with pytest.raises(ValueError):
        normalized_score([1.], [0.], .25)


def test_registered_real_calendar_has_four_training_seasons_and_5136_new_hours():
    archive, _ = fixture()
    clock_only = replace(archive, times=pd.date_range("2013-03-01", periods=35064,
                                                   freq="h", tz="Asia/Shanghai"))
    protocol = yaml.safe_load((Path(__file__).resolve().parents[1]
                              / "configs/transfer/beijing_seasonal_v4.yaml").read_text(encoding="utf-8"))
    spans = seasonal_spans(clock_only, protocol)
    assert len(spans["seasons"]) == 4
    assert spans["test"].stop - spans["test"].start == 5136
    assert all(row["validation"].stop < spans["outer_training_stop_exclusive"]
               for row in spans["seasons"])

from dataclasses import replace

import numpy as np
import pandas as pd

from airproof.beijing_v4 import (BeijingArchive, archive_slices, meteorological_components,
    public_field, public_weather, relative_humidity, run_weather_twin, select_outer_fold)


def fixture():
    rng = np.random.default_rng(916)
    steps, stations = 40, 4
    xy = np.array([[0., 0.], [0., 10.], [10., 0.], [10., 10.]])
    values = np.maximum(0, 20 + np.arange(stations)[None, :] + rng.normal(0, 2, (steps, stations)))
    weather = {"wind_u": np.full_like(values, 5.), "wind_v": np.full_like(values, 0.),
               "temperature": np.full_like(values, 22.), "humidity": np.full_like(values, 65.)}
    archive = BeijingArchive(values, weather, pd.date_range("2020-01-01", periods=steps, freq="h", tz="Asia/Shanghai"),
                              tuple("ABCD"), xy)
    protocol = {"outer_train_fraction": .5, "inner_fit_steps": 6, "inner_validation_steps": 4,
        "purge_hours": 1, "test_steps": 6, "historical_v1_steps": 2, "historical_v2_tail_steps": 2,
        "graph_neighbors": [2], "graph_length_scales_km": [15.], "correction_clips": [2.],
        "public_max_station_ratio": 1.10, "worst_station_weight": .25, "fixed_lag": 2,
        "citizen_delta": 4., "lambda_temporal": .02, "lambda_spatial": .02,
        "regulatory_sigma_model": 3., "transport": .12, "directionality": 1.}
    return archive, protocol


def test_weather_direction_and_derived_humidity():
    u, v = meteorological_components(["N", "E", "S", "W", "unknown", "unknown"], [2, 2, 2, 2, 0, 3])
    np.testing.assert_allclose(u[:5], [0, -2, 0, 2, 0], atol=1e-10)
    np.testing.assert_allclose(v[:5], [-2, 0, 2, 0, 0], atol=1e-10)
    assert np.isnan(u[-1]) and np.isnan(v[-1])
    humidity = relative_humidity([20., 20., 20., np.nan], [20., 10., 25., 10.])
    assert humidity[0] == 100 and 50 < humidity[1] < 55 and humidity[2] == 100 and np.isnan(humidity[-1])


def test_public_weather_excludes_target_and_never_backfills():
    archive, _ = fixture()
    fit = {key: value[:6] for key, value in archive.weather.items()}
    current = {key: value[6:12].copy() for key, value in archive.weather.items()}
    for values in current.values():
        values[1, :3] = np.nan
    expected = public_weather(fit, current, [0, 1, 2])
    changed = {key: value.copy() for key, value in current.items()}
    for name, values in changed.items():
        values[:, 3] = 100000
        values[3:, :3] = 10 if name != "humidity" else 20
    result = public_weather(fit, changed, [0, 1, 2])
    for name in current:
        np.testing.assert_array_equal(getattr(expected, name)[:3], getattr(result, name)[:3])
        assert getattr(expected, name)[1] == getattr(expected, name)[0]


def test_public_interpolation_masks_target_and_is_causal():
    archive, _ = fixture()
    values, fit = archive.concentrations[:8].copy(), archive.concentrations[8:14].copy()
    values[2, :3] = np.nan
    candidate = {"kernel": "idw", "scale": 2., "neighbors": 3}
    expected = public_field(values, fit, archive.coordinates, [0, 1, 2], candidate)
    fit[:, 3] = 999999
    values[:, 3] = 999999
    values[4:, :3] = 333333
    actual = public_field(values, fit, archive.coordinates, [0, 1, 2], candidate)
    np.testing.assert_array_equal(actual[:4], expected[:4])
    np.testing.assert_array_equal(expected[2], expected[1])


def test_production_twin_weather_is_active_bounded_and_target_excluded():
    archive, protocol = fixture()
    values = archive.concentrations[:8].copy()
    sources = [0, 1, 2]
    candidate = {"kernel": "idw", "scale": 2., "neighbors": 3}
    baseline = public_field(values, archive.concentrations[8:14], archive.coordinates, sources, candidate)
    weather = public_weather({key: value[8:14] for key, value in archive.weather.items()},
                             {key: value[:8] for key, value in archive.weather.items()}, sources)
    graph = {"neighbors": 2, "length_scale_km": 15., "correction_clip": 2.}
    predictions, diagnostics = run_weather_twin(values, baseline, archive.coordinates, sources, weather, graph, protocol)
    assert diagnostics["solver_failures"] == 0
    assert diagnostics["maximum_correction"] <= 2. + 1e-10
    assert diagnostics["mean_operator_distance_from_identity"] > 0
    values[:, 3] = 999999
    unchanged, _ = run_weather_twin(values, baseline, archive.coordinates, sources, weather, graph, protocol)
    np.testing.assert_array_equal(predictions, unchanged)
    identity, _ = run_weather_twin(values, baseline, archive.coordinates, sources, weather, graph, protocol,
                                   identity_transition=True)
    assert np.max(np.abs(predictions - identity)) > 1e-5
    values[4:, :3] = 999999
    future_baseline = baseline.copy()
    future_baseline[4:] = 999999
    future_result, _ = run_weather_twin(values, future_baseline, archive.coordinates, sources, weather, graph, protocol)
    np.testing.assert_array_equal(predictions[:4], future_result[:4])


def test_nested_selection_ignores_outer_station_and_all_test_labels():
    archive, protocol = fixture()
    expected = select_outer_fold(archive, 3, protocol)
    concentration = archive.concentrations.copy()
    concentration[:, 3] = 999999
    concentration[20:] = 222222
    weather = {key: values.copy() for key, values in archive.weather.items()}
    for values in weather.values():
        values[:, 3] = 999999
        values[20:] = 888888
    changed = replace(archive, concentrations=concentration, weather=weather)
    assert select_outer_fold(changed, 3, protocol) == expected
    spans = archive_slices(40, protocol)
    assert spans["fit"].stop + protocol["purge_hours"] == spans["validation"].start
    assert spans["test"].stop == 40 - protocol["historical_v2_tail_steps"] - protocol["purge_hours"]

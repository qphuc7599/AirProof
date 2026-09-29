from __future__ import annotations

import numpy as np

from airproof.config import load_config, with_overrides
from airproof.prediction import PREDICTORS, _cell_coordinates, _idw_field, run_prediction_seed


def test_idw_preserves_observed_cells() -> None:
    values = np.asarray([1.0, np.nan, np.nan, 4.0])
    estimate = _idw_field(values, _cell_coordinates(2))
    assert estimate[0] == 1.0
    assert estimate[3] == 4.0
    assert np.all(np.isfinite(estimate))


def test_prediction_endpoint_methods_share_one_world() -> None:
    config = with_overrides(
        load_config("configs/v3/prediction_endpoints.yaml"),
        {
            "world.agents": 30,
            "world.grid_side": 4,
            "world.steps": 24,
            "world.burn_in_steps": 12,
            "world.reference_station_count": 4,
            "attack.start_epoch": 12,
        },
    )
    rows = run_prediction_seed(config, 17, PREDICTORS)
    assert [row["predictor"] for row in rows] == list(PREDICTORS)
    assert len({row["world"]["candidate_count"] for row in rows}) == 1
    assert all(np.isfinite(row["metrics"]["rmse"]) for row in rows)
    assert all(row["metrics"]["rmse"] >= 0 for row in rows)

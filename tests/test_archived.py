import numpy as np

from airproof.archived import (
    BEIJING_COORDINATES,
    _adaptive_kernel_weights,
    _adaptive_predict,
    _distance_matrix,
    _graph_laplacian,
)


def test_station_coordinate_ledger_and_heldout_graph():
    stations = sorted(BEIJING_COORDINATES)
    assert len(stations) == 12
    distances = _distance_matrix(stations)
    assert (distances.diagonal() == 0).all()
    laplacian = _graph_laplacian(distances, held_out=0)
    assert laplacian.shape == (12, 12)
    assert abs(laplacian.sum()) < 1e-9


def test_adaptive_graph_never_uses_target_and_contains_frozen_idw():
    stations = sorted(BEIJING_COORDINATES)
    distances = _distance_matrix(stations)
    target = 0
    sources = list(range(1, len(stations)))
    weights = _adaptive_kernel_weights(
        distances,
        target,
        sources,
        kernel="idw",
        scale=2.0,
        neighbors=len(sources),
    )
    assert np.isclose(weights.sum(), 1.0)
    values = np.arange(3 * len(stations), dtype=float).reshape(3, len(stations))
    candidate = {
        "kernel": "idw",
        "scale": 2.0,
        "neighbors": len(sources),
        "robust_delta": None,
    }
    first = _adaptive_predict(values, distances, target, sources, candidate)
    values[:, target] = 1e9
    second = _adaptive_predict(values, distances, target, sources, candidate)
    assert np.allclose(first, second)

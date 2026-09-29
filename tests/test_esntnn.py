from dataclasses import replace

import numpy as np
import pytest

from airproof.esntnn import (DESNConfig, DeepEchoStateEnsemble, convex_weight,
                            make_transformer, station_windows)


def fixture():
    rng = np.random.default_rng(44)
    return np.maximum(0, 20 + np.cumsum(rng.normal(size=(100, 4)), axis=0))


def small_config():
    return DESNConfig(units=16, reduced_units=3, ensemble=2, washout=3, sparsity=0.5)


def test_desn_future_targets_do_not_change_earlier_forecasts():
    data = fixture()
    model = DeepEchoStateEnsemble(small_config(), seed=12).fit(data[:50])
    original = model.predict_series(data, include_next=True)
    altered = data.copy()
    altered[80:] += 1000
    changed = model.predict_series(altered, include_next=True)
    np.testing.assert_allclose(original[2:81], changed[2:81], rtol=0, atol=1e-10)
    prefix = model.predict_series(data[:80], include_next=True)
    np.testing.assert_allclose(original[2:81], prefix[2:], rtol=0, atol=1e-10)
    assert not np.allclose(original[81:], changed[81:])
    assert np.isfinite(original[2:]).all()


def test_desn_reproducible_frozen_prefix_and_training_errors():
    data = fixture()
    a = DeepEchoStateEnsemble(small_config(), seed=12).fit(data[:50])
    b = DeepEchoStateEnsemble(small_config(), seed=12).fit(data[:50])
    np.testing.assert_allclose(a.predict_series(data)[2:], b.predict_series(data)[2:])
    assert a.metadata()["fit_length"] == 50
    with pytest.raises(ValueError):
        DeepEchoStateEnsemble(replace(small_config(), spectral_radii=(0.5,)))
    with pytest.raises(ValueError):
        DeepEchoStateEnsemble(small_config()).fit(data[:4])
    with pytest.raises(ValueError):
        a.predict_series(data[:, :2])


def test_window_has_no_current_or_future_truth():
    data = fixture()
    predictions = data + 1
    a, b = station_windows(data, predictions, np.array([70]), history=12, mean=10, scale=2)
    altered = data.copy()
    altered[70:] = 100000
    c, d = station_windows(altered, predictions, np.array([70]), history=12, mean=10, scale=2)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    np.testing.assert_allclose(a[0, :, 1], (data[58:70, 0] - 10) / 2)
    np.testing.assert_allclose(b[:, 0, 1], (predictions[70] - 10) / 2)
    assert a.shape == (4, 12, 2) and b.shape == (4, 1, 2)


def test_validation_convex_combination_is_optimal_in_interval():
    a = np.array([1., 2., 3.])
    b = np.array([4., 5., 6.])
    assert convex_weight(a, b, 0.3 * a + 0.7 * b) == pytest.approx(0.7)
    assert convex_weight(a, b, a - 1) == 0
    assert convex_weight(a, b, b + 1) == 1
    assert convex_weight(a, a, b) == 0


def test_transformer_decoder_mask_and_gradients():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = make_transformer(dropout=0)
    src, dec = torch.randn(3, 12, 2), torch.randn(3, 4, 2)
    model.eval()
    a = model(src, dec)
    changed = dec.clone()
    changed[:, 2:] += 1000
    b = model(src, changed)
    torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=1e-6)
    assert a.shape == (3, 4, 1)
    a.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

import numpy as np
import pytest

from airproof.v6_public_dynamics import fit_public_increment
from airproof.v6_public_event_center import make_high_state_hold


def _model(history):
    neighbors = np.array([[1, 2], [0, 2], [0, 1]])
    base = fit_public_increment(history[:24], neighbors)
    return make_high_state_hold(base, history[:24], threshold_quantile=.9, blend=1.)


def test_high_state_hold_is_causal_and_never_lowers_base():
    history = 10 + np.random.default_rng(311).normal(size=(40, 3))
    model = _model(history)
    prediction = model.predict_next(history[:30])
    base = model.base.predict_next(history[:30])
    changed = history.copy()
    changed[30:] = 1e8
    assert np.array_equal(prediction, model.predict_next(changed[:30]))
    assert np.all(prediction >= base)


def test_high_state_hold_uses_only_closed_prefix_threshold():
    history = np.tile(np.arange(40, dtype=float)[:, None], (1, 3))
    model = _model(history)
    changed = history.copy()
    changed[24:] = 1e9
    same_model = _model(changed)
    assert model.threshold == same_model.threshold
    with pytest.raises(ValueError):
        make_high_state_hold(model.base, history[:24], threshold_quantile=1., blend=.5)

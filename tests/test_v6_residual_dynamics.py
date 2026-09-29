import numpy as np
import pytest
from airproof.v6_residual_dynamics import (fit_frozen_residual_ar,
    public_residual_forcing)


def test_coordinate_conversion_preserves_physical_innovations_and_prefix():
    rng = np.random.default_rng(131)
    public, state, forcing = (rng.normal(size=(5, 3)) for _ in range(3))
    previous_public, previous_state = rng.normal(size=(2, 3))
    operators = [rng.normal(size=(3, 3)) for _ in range(5)]
    converted = public_residual_forcing(public, operators, forcing, previous_public=previous_public)
    z = state-public
    for t, operator in enumerate(operators):
        physical = state[t] - operator @ (state[t-1] if t else previous_state) - forcing[t]
        residual = z[t] - operator @ (z[t-1] if t else previous_state-previous_public) - converted[t]
        assert np.allclose(physical, residual, atol=1e-12)
    prefix = public_residual_forcing(public[:3], operators[:3], forcing[:3], previous_public=previous_public)
    assert np.array_equal(prefix, converted[:3])


def test_zero_residual_forcing_requires_public_to_follow_declared_physics():
    public = np.array([[2.], [7.]])
    operators = [np.eye(1), np.eye(1)]
    result = public_residual_forcing(public, operators, np.zeros((2, 1)), previous_public=np.array([2.]))
    assert np.array_equal(result, [[0.], [-5.]])
    with pytest.raises(ValueError):
        public_residual_forcing(public, operators, np.zeros((2, 1)), previous_public=np.array([np.nan]))


def test_frozen_residual_ar_recovers_causal_centered_process():
    rng = np.random.default_rng(71)
    residual = np.zeros((500, 2)); residual[0] = [2., -1.]
    for epoch in range(1, len(residual)):
        residual[epoch] = ([2., -1.]+.8*(residual[epoch-1]-[2., -1.])
                           +rng.normal(0, .05, 2))
    public = np.full_like(residual, 10.); truth = public+residual
    model = fit_frozen_residual_ar(public, truth, training_start=0,
                                   training_end=400, minimum_pairs=100)
    assert np.allclose(model.mean, [2., -1.], atol=.1)
    assert np.allclose(model.coefficient, .8, atol=.15)
    transitions, forcing = model.system(12)
    assert len(transitions) == 12 and forcing.shape == (12, 2)
    assert np.allclose(transitions[0].diagonal(), model.coefficient)
    changed = truth.copy(); changed[400:] = 1e9
    same = fit_frozen_residual_ar(public, changed, training_start=0,
                                  training_end=400, minimum_pairs=100)
    assert np.array_equal(model.coefficient, same.coefficient)

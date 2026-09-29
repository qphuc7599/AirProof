import numpy as np
import pytest
from airproof.v6_public_dynamics import fit_public_increment,increment_features


def test_causal_prediction_does_not_change_with_future_labels():
    h=10+np.random.default_rng(112).normal(size=(40,3))
    neighbors=np.array([[1,2],[0,2],[0,1]])
    model=fit_public_increment(h[:24],neighbors)
    predictions=np.array([model.predict_next(h[:t]) for t in range(24,30)])
    changed=h.copy();changed[30:]=1e6
    assert np.array_equal(predictions,[model.predict_next(changed[:t]) for t in range(24,30)])
    assert model.training_epochs==24
    with pytest.raises(ValueError):increment_features(h[:5],neighbors)


def test_constant_public_process_has_zero_increment():
    h=np.full((24,3),7.)
    model=fit_public_increment(h,np.array([[1],[2],[0]]),event_weight=2)
    assert np.allclose(model.predict_next(h),7)

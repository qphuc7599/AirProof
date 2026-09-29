import numpy as np
import pytest
from airproof.v6_release_value import release_age_features,ridge_correction


def test_release_features_are_causal_and_suppression_is_missing():
    values=np.full((60,2),np.nan);mask=np.zeros((60,2),bool);scheduled=np.zeros(60,bool)
    scheduled[[0,24]]=True;values[0]=[2,3];mask[0]=True;values[24,0]=7;mask[24,0]=True
    a,x,c=release_age_features(values,mask,scheduled,centers=np.zeros(2))
    assert not a[:24*2].any() and x[24*2,0]==2 and x[24*2+1,3]==3
    # Suppressed group1 at acquisition24 retains its older value/age state.
    assert x[48*2,0]==7 and x[48*2+1,5]==3
    changed=values.copy();changed[24,0]=100
    _,future,_=release_age_features(changed,mask,scheduled,centers=np.zeros(2))
    assert np.array_equal(x[:48*2],future[:48*2])
    with pytest.raises(ValueError):release_age_features(np.nan_to_num(values),mask,scheduled)


def test_ridge_correction_has_declared_zero_without_features():
    x=np.zeros((20,3));coef=ridge_correction(np.arange(20),x)
    assert np.array_equal(coef,np.zeros(3))

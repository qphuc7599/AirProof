import numpy as np

from airproof.v6_public_archive_model import causal_archive_design
from airproof.v6_public_rolling import rolling_stationwise_ridge


def test_rolling_ridge_does_not_use_future_targets_or_refit_within_block():
    rng=np.random.default_rng(808)
    values=10+rng.normal(size=(360,3));coordinates=np.array([[0.,0.],[1.,0.],[0.,1.]])
    medians=np.median(values[:200],axis=0)
    features=causal_archive_design(values,medians,coordinates)
    base=np.full_like(values,10.)
    first,fits=rolling_stationwise_ridge(features,values,base,start_epoch=240,
        lookback=200,update_every=24,minimum_history=168)
    changed=values.copy();changed[300:]=1e8
    changed_features=causal_archive_design(changed,medians,coordinates)
    second,_=rolling_stationwise_ridge(changed_features,changed,base,start_epoch=240,
        lookback=200,update_every=24,minimum_history=168)
    assert np.array_equal(first[:300],second[:300])
    assert fits[0]["fit_end"]==240 and fits[0]["prediction_end"]==264
    assert np.isfinite(first).all() and (first>=0).all()


def test_rolling_ridge_rejects_too_short_lookback():
    values=np.ones((200,1));features=np.ones((200,1,2))
    try:
        rolling_stationwise_ridge(features,values,values,start_epoch=180,lookback=100)
    except ValueError:
        return
    raise AssertionError("short lookback accepted")

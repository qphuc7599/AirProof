from dataclasses import replace

import numpy as np
import pytest

from airproof.records import Observation
from airproof.v6_estimator import (
    EstimatorConfig, estimate_public_field,
    history_normalized_sensitivity_certificate, history_normalized_weights)


def obs(user=0,epoch=0,value=14.,arrival=0,key='a',quality=1.):
    return Observation(user,epoch,0,0,value,1.,quality,512,key,arrival,arrival)


def run(records,config=None,steps=1):
    public=np.full((steps,1),10.)
    return estimate_public_field(public,np.zeros((1,2)),records,config,
        innovation_scales=2.)


def normalized(budget=1.,**kwargs):
    return EstimatorConfig(history_normalized_huber=True,
        per_user_weight_budget=budget,tolerance=1e-9,**kwargs)


def test_weight_normalization_and_prolific_identical_duplication_invariance():
    weights=history_normalized_weights(np.array([1,1,2]),np.array([1.,3.,.5]),2.)
    np.testing.assert_allclose(weights,[.5,1.5,.5])
    single=run([obs() ],normalized())
    prolific=run([obs(key=f'copy-{i}') for i in range(20)],normalized())
    np.testing.assert_allclose(single.reconstructed,prolific.reconstructed,atol=1e-8)
    assert prolific.diagnostics['epoch_terms'][0]['maximum_user_window_weight']==pytest.approx(1.)


def test_singleton_parity_when_raw_weight_is_within_prospective_budget():
    record=obs(quality=.85)
    ordinary=run([record],EstimatorConfig(tolerance=1e-9))
    protected=run([record],normalized(1.))
    np.testing.assert_allclose(ordinary.live,protected.live,atol=1e-9)


def test_full_history_replacement_is_below_fixed_window_certificate():
    cfg=normalized(1.,lag=2,lambda_zero=.5)
    first=[obs(4,e,30.,e,f'a-{e}') for e in range(3)]
    replacement=[obs(4,e,-30.,e,f'b-{e}') for e in range(3)]
    left=run(first,cfg,steps=3);right=run(replacement,cfg,steps=3)
    certificate=left.diagnostics['history_normalized_sensitivity_certificate']
    distance=float(np.linalg.norm(left.reconstructed-right.reconstructed))
    assert distance <= certificate['fixed_window_minimizer_l2_bound']+1e-7
    expected=history_normalized_sensitivity_certificate(strong_convexity_mu=.5,
        huber_delta=1.345,scale_floor=2.,per_user_weight_budget=1.)
    assert certificate==expected


def test_causal_arrival_and_default_remain_unchanged():
    cfg=normalized(lag=1)
    record=obs(arrival=1)
    protected=run([record],cfg,steps=2)
    empty=run([],cfg,steps=2)
    np.testing.assert_array_equal(protected.live[0],empty.live[0])
    assert protected.reconstructed[0,0]>empty.reconstructed[0,0]
    default=EstimatorConfig()
    assert default.history_normalized_huber is False
    assert (default.cap,default.huber_delta,default.lag)==(8.,1.345,6)
    with pytest.raises(ValueError,match='ordinary Huber'):
        replace(cfg,loss='quadratic')
    with pytest.raises(ValueError,match='integer user partition'):
        run([replace(obs(),user_id='ambiguous')],cfg)


def test_absolute_cap8_and_kkt_are_retained():
    records=[obs(0,0,1e9,0,f'x-{i}') for i in range(30)]
    result=run(records,normalized(lambda_zero=.001,lambda_temporal=0.))
    assert result.live[0,0]==pytest.approx(18.)
    assert result.diagnostics['maximum_correction']<=8.
    assert result.diagnostics['epoch_terms'][0]['projected_gradient_inf']<1e-7

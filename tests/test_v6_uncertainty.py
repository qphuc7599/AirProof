import numpy as np
import pytest

from airproof.v5_estimator import EstimatorResult
from airproof.v6_uncertainty import (
    evaluate_estimator_intervals,
    evaluate_intervals,
    fit_frozen_intervals,
    declared_uncertainty_strata,
)


def calibrate(clock="live", **kwargs):
    return fit_frozen_intervals(np.full((100, 2), 10.), np.full((100, 2), 12.),
                               clock=clock, split_id="independent-cal", training_end=100.,
                               target_available_at=kwargs.pop("target_available_at", 100.),
                               selection_id='frozen-model',selection_end=99.,
                               evaluation_split_id='heldout-eval',evaluation_start=101.,**kwargs)


def test_independent_frozen_clock_calibration_and_support_fallback():
    model = calibrate(public_strata=np.array(["a", "b"]))
    assert len(model.strata) == 2
    lower, upper = model.predict([[10., 10.]], epochs=101.,evaluation_split_id='heldout-eval', public_strata=["unseen", "a"])
    np.testing.assert_array_equal(lower, [[ [8., 8.], [8., 8.] ]])
    np.testing.assert_array_equal(upper, [[ [12., 12.], [12., 12.] ]])
    with pytest.raises(ValueError):
        model.predict([[10.]], epochs=100.,evaluation_split_id='heldout-eval')
    with pytest.raises(ValueError):
        calibrate(role="test")
    with pytest.raises(ValueError):
        calibrate(target_available_at=101.)
    with pytest.raises(ValueError):
        calibrate(target_available_at=99.)


def test_metrics_and_deterministic_whole_time_block_bootstrap():
    truth = np.full((8, 2), 10.)
    low, high = np.full((8, 2, 2), 8.), np.full((8, 2, 2), 12.)
    result = evaluate_intervals(truth, low, high, public_strata=["a", "b"],
                               event_mask=False, block_length=2, bootstrap_replicates=20, seed=5)
    assert result["groups"]["event"] == {"n": 0, "status": "unavailable"}
    for group in ("all", "stratum:a", "stratum:b"):
        assert result["groups"][group]["coverage"]["estimate"] == [1., 1.]
        assert result["groups"][group]["width"]["estimate"] == [4., 4.]
        assert result["groups"][group]["interval_score"]["estimate"] == [4., 4.]
        assert result["groups"][group]["coverage"]["ci95"] == [[1., 1.], [1., 1.]]
    repeat = evaluate_intervals(truth, low, high, public_strata=["a", "b"],
                               event_mask=False, block_length=2, bootstrap_replicates=20, seed=5)
    assert repeat == result


def test_outside_interval_penalty_and_clock_binding():
    truth = np.full((4, 1), 15.)
    pred = np.full((4, 1), 10.)
    result = EstimatorResult(pred, pred.copy(), {})
    calibrators = {clock: calibrate(clock) for clock in ("live", "reconstructed")}
    outcome = evaluate_estimator_intervals(result, calibrators, truth, epochs=101.,
                                           evaluation_split_id='heldout-eval',
                                           block_length=2, bootstrap_replicates=10)
    score = outcome["live"]["groups"]["all"]["interval_score"]["estimate"]
    np.testing.assert_allclose(score, [64., 124.])
    calibrators["live"] = calibrators["reconstructed"]
    with pytest.raises(ValueError):
        evaluate_estimator_intervals(result, calibrators, truth, epochs=101.,
                                     evaluation_split_id='heldout-eval',block_length=2)


def test_evaluation_never_refits_frozen_calibration():
    model = calibrate()
    pred = np.full((4, 1), 10.)
    before = model.predict(pred, epochs=101.,evaluation_split_id='heldout-eval')
    evaluate_intervals(np.full_like(pred, 1e3), *before, block_length=2, bootstrap_replicates=10)
    after = model.predict(pred, epochs=101.,evaluation_split_id='heldout-eval')
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)


def test_sparse_event_bootstrap_can_be_unavailable_without_crashing():
    truth = np.ones((100, 1))
    events = np.zeros_like(truth, dtype=bool)
    events[0] = True
    report = evaluate_intervals(truth, np.zeros((100, 1, 2)), np.ones((100, 1, 2)),
                                event_mask=events, block_length=50, bootstrap_replicates=1, seed=0)
    event = report["groups"]["event"]
    assert event["n"] == 1
    assert event["coverage"]["ci95"] is None
    assert event["coverage"]["bootstrap_valid"] == 0


def test_declared_group_age_context_labels_and_small_stratum_fallback():
    labels=declared_uncertainty_strata([[0,1]],[[0,6]],[['near','far']])
    assert labels.tolist()==[['group=0|age=0|context=near','group=1|age=6|context=far']]
    sparse=np.array(['large']*199+['small'])[:,None]
    model=fit_frozen_intervals(np.full((200,1),10.),np.full((200,1),12.),clock='live',
        split_id='cal',training_end=2,target_available_at=2,public_strata=sparse,
        selection_id='selected',selection_end=1,evaluation_split_id='eval',evaluation_start=3)
    assert model.fallback_strata==('small',)
    low,high=model.predict([[10.]],epochs=3,evaluation_split_id='eval',public_strata=['small'])
    np.testing.assert_array_equal(low,[[[8.,8.]]]);np.testing.assert_array_equal(high,[[[12.,12.]]])
    with pytest.raises(ValueError):model.predict([[10.]],epochs=3,evaluation_split_id='wrong')
    with pytest.raises(ValueError):model.predict([[10.]],epochs=2.5,evaluation_split_id='eval')

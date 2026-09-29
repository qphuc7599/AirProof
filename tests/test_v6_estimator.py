from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from airproof.records import Observation
from airproof.v6_estimator import (
    EstimatorConfig,
    estimate_matched_controls,
    estimate_public_field,
    huber_value_gradient,
)


def observation(epoch=0, cell=0, value=14., arrival=0, key="a"):
    return Observation(1, epoch, cell, 0, value, 1., 1., 512, key, arrival, arrival)


def run(public, observations=(), config=None, **kwargs):
    public = np.asarray(public, float)
    return estimate_public_field(public, np.zeros((public.shape[1], 2)), observations,
                                 config, innovation_scales=kwargs.pop("innovation_scales", 2.),
                                 **kwargs)


def test_dense_reference_and_box_kkt():
    public = np.full((2, 2), 10.)
    config = EstimatorConfig(cap=2., lag=1, tolerance=1e-8)
    lap = np.array([[1., -1.], [-1., 1.]])
    obs = [observation(0, 0, 30., 0, "a"), observation(1, 1, -30., 1, "b")]
    result = run(public, obs, config, laplacian=sparse.csr_matrix(lap))
    D = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [-1., 0, 1., 0], [0, -1., 0, 1.]])
    Q = .5*D.T@D + np.kron(np.eye(2), .2*lap+.5*np.eye(2))

    def dense(z):
        r = (z[[0, 3]] - np.array([20., -40.]))/2
        a = abs(r)
        return .5*z@Q@z + np.sum(np.where(a < 1.345, .5*r*r, 1.345*(a-.5*1.345)))

    reference = minimize(dense, np.zeros(4), method="SLSQP", bounds=[(-2, 2)]*4,
                         options={"ftol": 1e-13, "maxiter": 1000})
    assert reference.success
    np.testing.assert_allclose((result.reconstructed-public).ravel(), reference.x, atol=2e-6)
    assert result.diagnostics["epoch_terms"][-1]["projected_gradient_inf"] < 1e-7


def test_nonnegative_and_finite_cap_with_extreme_histories():
    p = np.full((3, 1), .1)
    config = EstimatorConfig(lambda_zero=.001, lambda_temporal=0)
    high = run(p, [observation(value=1e8)], config)
    low = run(p, [observation(value=-1e8)], config)
    assert np.all(low.reconstructed >= 0)
    assert np.max(abs(high.reconstructed-p)) <= 8.
    assert np.max(abs(high.live-low.live)) <= 16.
    assert high.live[0, 0] == pytest.approx(8.1)


def test_huber_bounded_score_and_finite_difference():
    r = np.array([-100., -1., 0., .7, 100.])
    _, score = huber_value_gradient(r, 1.345)
    numerical = (huber_value_gradient(r+1e-5, 1.345)[0]
                 - huber_value_gradient(r-1e-5, 1.345)[0])/2e-5
    np.testing.assert_allclose(score, numerical, atol=1e-8)
    assert np.max(abs(score)) <= 1.345


def test_causal_duplicates_and_future_private_public_invariance():
    p = np.full((5, 1), 10.)
    early = observation(value=16., arrival=0)
    delayed_duplicate = replace(early, value=-100., relay_arrival=4)
    a = run(p, [delayed_duplicate, early])
    b = run(p, [early])
    np.testing.assert_array_equal(a.live, b.live)
    assert a.diagnostics["duplicate_records"] == 1
    changed = p.copy()
    changed[3:] = 100
    scales = np.full_like(p, 2.)
    scales[3:] = 50
    c = run(changed, [early, observation(3, 0, -200., 3, "future")],
            innovation_scales=scales)
    np.testing.assert_array_equal(c.live[:3], b.live[:3])
    d = run(p, [replace(early, sigma=1e100)])
    np.testing.assert_array_equal(d.live, b.live)


def test_closed_prefix_inclusive_lag_and_immutable_live():
    p = np.full((5, 1), 10.)
    cfg = EstimatorConfig(lag=1)
    baseline = run(p, config=cfg)
    timely = run(p, [observation(arrival=1)], cfg)
    late = run(p, [observation(arrival=2)], cfg)
    assert timely.live[0] == baseline.live[0]
    assert timely.reconstructed[0] > baseline.reconstructed[0]
    np.testing.assert_array_equal(late.reconstructed, baseline.reconstructed)
    prefix = run(p[:2], [observation(arrival=1)], cfg)
    np.testing.assert_array_equal(prefix.reconstructed[0], timely.reconstructed[0])
    assert late.diagnostics["retrospective_records"] == 1


def test_forcing_and_closed_boundary_have_correct_normal_equations():
    p = np.full((3, 1), 10.)
    cfg = EstimatorConfig(lag=0, lambda_temporal=2., lambda_zero=.5)
    forcing = np.array([[1.], [2.], [-1.]])
    result = run(p, config=cfg, transitions=[sparse.csr_matrix([[.7]])]*3,
                 residual_forcing=forcing)
    expected, boundary = [], 0.
    for f in forcing[:, 0]:
        boundary = 2*(.7*boundary+f)/2.5
        expected.append(boundary+10)
    np.testing.assert_allclose(result.live[:, 0], expected, atol=1e-7)


def test_explicit_time_and_precision_normalization():
    cfg = EstimatorConfig(time_step=2., precision_normalizer=4., huber_delta=100.)
    result = run([[10.]], [observation(value=12.)], cfg)
    # Data precision 1/(4*2**2); temporal .5/2; zero .5*2.
    expected = 10 + (2/16)/(.25+1+1/16)
    assert result.live[0, 0] == pytest.approx(expected)


def test_context_and_laplacian_validation():
    with pytest.raises(TypeError):
        estimate_public_field(np.ones((1, 1)), np.zeros((1, 2)), [])
    with pytest.raises(ValueError):
        run([[1]], innovation_scales=0)
    with pytest.raises(ValueError):
        run([[1, 1]], laplacian=sparse.csr_matrix([[-1, 1], [1, -1]]))
    with pytest.raises(ValueError):
        EstimatorConfig(cap=np.inf)


def test_matched_controls_quadratic_limits_and_cap_kkt():
    p = np.array([[10.]])
    cfg = EstimatorConfig(lambda_temporal=0., lambda_zero=.01, clip_delta=4.)
    controls = estimate_matched_controls(p, np.zeros((1, 2)), [observation(value=110.)],
                                        cfg, innovation_scales=2.)
    assert set(controls) == {"public_only", "quadratic", "clipping_only", "cap_only", "bounded_huber"}
    assert controls["quadratic"].live[0, 0] == pytest.approx(10+25/.26)
    assert controls["clipping_only"].live[0, 0] == pytest.approx(10+2/.26)
    assert controls["cap_only"].live[0, 0] == pytest.approx(18.)
    assert controls["bounded_huber"].live[0, 0] == pytest.approx(18.)
    for name, arm in controls.items():
        if name != "public_only":
            assert arm.diagnostics["solver_failure_rate"] == 0
    np.testing.assert_array_equal(controls["public_only"].live, p)


def test_quadratic_and_huber_agree_in_quadratic_region_without_wrappers():
    p = np.full((2, 1), 10.)
    data = [observation(value=11.), observation(1, 0, 9., 1, "b")]
    cfg = EstimatorConfig(loss="quadratic", output_cap=False, input_clip=False)
    quadratic = run(p, data, cfg)
    huber = run(p, data, replace(cfg, loss="huber", huber_delta=100.))
    np.testing.assert_allclose(huber.reconstructed, quadratic.reconstructed, atol=1e-8)
    # Independent dense normal equation on the final window.
    Q = np.array([[1.5, -.5], [-.5, 1.]]) + np.eye(2)/4
    expected = np.linalg.solve(Q, np.array([1., -1.])/4)
    np.testing.assert_allclose(quadratic.reconstructed[:, 0]-10, expected, atol=1e-7)

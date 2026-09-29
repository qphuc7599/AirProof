from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from airproof.records import Observation
from airproof.v5_estimator import EstimatorConfig, enumerate_candidates, estimate_public_field
from airproof.v5_release_twin import PublicReleaseObservation, ReleaseOnlyTwin, fit_release_calibration
from airproof.v5_uncertainty import fit_interval_calibrator, predict_intervals, interval_metrics


def obs(t, cell, value, arrival=None, source="citizen"):
    return Observation(0, t, cell, 0, value, 1., 1., 512, f"{t}-{cell}",
                       t if arrival is None else arrival, t if arrival is None else arrival,
                       source_class=source)


@pytest.mark.parametrize("objective", ["residual", "full_field"])
def test_dense_oracle_nonidentity_fixed_boundary(objective):
    public = np.array([[10., 12.], [13., 9.], [8., 15.], [14., 7.]])
    operators = [np.array([[.8, .1], [.2, .7]]) for _ in public]
    lap = np.array([[1., -1.], [-1., 1.]])
    cfg = EstimatorConfig(objective=objective, lag=1, input_clip=False, output_cap=False,
                          tolerance=1e-11)
    rows = [obs(t, j, public[t, j]+1+j) for t in range(4) for j in range(2)]
    result = estimate_public_field(public, np.zeros((2, 2)), rows, cfg,
                                   laplacian=lap, transitions=operators)
    states = public.copy()
    expected_live = []
    for t in range(4):
        start = max(0, t-1)
        n = t-start+1
        d = np.eye(n*2)
        if n > 1:
            d[2:, :2] = -operators[t]
        boundary = (np.zeros(2) if objective == "residual" else public[0]) if start == 0 else states[start-1].copy()
        if start and objective == "residual":
            boundary -= public[start-1]
        boundary = operators[start]@boundary
        target = np.array([[r.value for r in rows if r.epoch == e] for e in range(start, t+1)])
        if objective == "residual":
            target -= public[start:t+1]
        matrix = .5*d.T@d+np.kron(np.eye(n), .2*lap+(.5 if objective == "residual" else 0)*np.eye(2))+np.eye(n*2)
        rhs = target.ravel()
        rhs[:2] += .5*boundary
        solution = np.linalg.solve(matrix, rhs).reshape(n, 2)
        states[start:t+1] = solution+(public[start:t+1] if objective == "residual" else 0)
        expected_live.append(states[t].copy())
    np.testing.assert_allclose(result.live, expected_live, atol=1e-8)
    np.testing.assert_allclose(result.reconstructed, states, atol=1e-8)
    assert result.diagnostics["solver_failure_rate"] == 0


def test_causality_replay_clipping_and_closed_prefix():
    public = np.full((8, 1), 10.)
    cfg = EstimatorConfig(lag=1)
    rows = [obs(0, 0, 100.), obs(3, 0, 20., arrival=4), obs(0, 0, -100., arrival=7)]
    run = lambda p, r: estimate_public_field(p, np.zeros((1, 2)), r, cfg)
    a = run(public, rows[:2])
    b = run(public, rows[:2]+rows[:2])
    np.testing.assert_array_equal(a.live, b.live)
    prefix = run(public[:3], rows[:1])
    np.testing.assert_array_equal(prefix.live, a.live[:3])
    np.testing.assert_array_equal(prefix.reconstructed[:1], a.reconstructed[:1])
    future = public.copy(); future[5:] = 100
    np.testing.assert_array_equal(run(future, rows[:2]).live[:5], a.live[:5])
    assert a.diagnostics["maximum_bounded_input_residual"] == 4
    regulatory = run(public, [obs(0, 0, 100., source="regulatory")])
    assert regulatory.diagnostics["input_clip_count"] == 0
    assert regulatory.diagnostics["maximum_correction"] <= 8
    late = run(public, [replace(rows[2], nullifier="late")])
    assert late.diagnostics["retrospective_records"] == 1
    assert len({x.candidate_id for x in enumerate_candidates()}) == 12


def test_intervals_frozen_strata_and_scores():
    pred = np.full(240, 20.)
    truth = pred+np.arange(240)/100
    labels = np.array(["large"]*150+["small"]*90)
    cal = fit_interval_calibrator(pred, truth, labels)
    assert set(cal.stratum_radii) == {"large"}
    lo, hi = predict_intervals(cal, pred, labels)
    np.testing.assert_allclose(hi[200]-pred[200], cal.pooled_radii)
    assert np.all(hi[:, 1] >= hi[:, 0])
    metrics = interval_metrics(truth, lo, hi)
    assert metrics["0.9"]["n"] == 240
    assert metrics["0.95"]["mean_interval_score"] >= 0


def test_release_calibration_world_boundaries_and_scalar_update():
    rng = np.random.default_rng(42)
    truth = 20+rng.normal(size=(3, 60, 1))
    public = np.full_like(truth, 20)
    query = truth+rng.normal(size=truth.shape)
    cal = fit_release_calibration(truth, public, query)
    latent = np.concatenate((truth-public, query-truth), axis=-1)
    centered = latent-latent.mean(axis=(0, 1))
    a, b = centered[:, :-1].reshape(-1, 2), centered[:, 1:].reshape(-1, 2)
    np.testing.assert_allclose(np.diag(cal.transition), np.clip((a*b).sum(0)/(a*a).sum(0), -.98, .98))
    twin = ReleaseOnlyTwin(cal)
    release = PublicReleaseObservation("one", 0, 0, 0, 24., 2.)
    h = np.ones(2)
    gain = cal.initial_covariance@h/(h@cal.initial_covariance@h+8.)
    expected = cal.mean+gain*(4-h@cal.mean)
    twin.update(0, np.array([20.]), [release, release])
    np.testing.assert_allclose(twin.mean, expected)
    assert twin.diagnostics()["duplicate_releases"] == 1
    old = twin.live[0].copy()
    for t in range(1, 55):
        incoming = [PublicReleaseObservation("delayed", 0, t, 0, 22., 1.)] if t == 24 else []
        twin.update(t, np.array([20.]), incoming)
    np.testing.assert_array_equal(twin.live[0], old)
    assert len(twin.epochs) == 49
    twin.update(55, np.array([20.]), [PublicReleaseObservation("late", 0, 55, 0, 22., 1.),
                                   PublicReleaseObservation("suppressed", 55, 55, 0, None, 1.)])
    assert twin.retrospective == twin.suppressed == 1
    with pytest.raises(ValueError):
        twin.update(56, np.array([20.]), [PublicReleaseObservation("future", 56, 57, 0, 20., 1.)])
    with pytest.raises(ValueError):
        twin.update(56, np.array([20.]), [PublicReleaseObservation("conflict", 56, 56, 0, 20., 1.),
                                       PublicReleaseObservation("conflict", 56, 56, 0, 21., 1.)])
    assert max(twin.public) == 55  # Invalid batches are rejected before state mutation.


def test_clipping_center_is_public_even_after_private_history():
    public = np.full((5, 1), 10.)
    rows = [obs(0, 0, 100., source="regulatory"), obs(3, 0, 100.)]
    cfg = EstimatorConfig(output_cap=False)
    result = estimate_public_field(public, np.zeros((1, 2)), rows, cfg)
    equivalent = estimate_public_field(public, np.zeros((1, 2)),
                                      [rows[0], replace(rows[1], value=14.)], cfg)
    np.testing.assert_allclose(result.live, equivalent.live)
    assert result.diagnostics["input_clip_count"] == 1

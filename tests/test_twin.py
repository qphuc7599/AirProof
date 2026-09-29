import numpy as np
from scipy import sparse

from airproof.records import Observation
from airproof.twin import (
    FixedLagTwin,
    adaptive_huber_state,
    innovation_contamination_fraction,
    robust_score_correction_trajectory,
    robust_state_update,
    robust_trajectory_update,
)


def record(value, epoch=0, arrival=0, sigma=1.0):
    return Observation(1, epoch, 0, 0, value, sigma, 1.0, 100, f"n-{epoch}-{arrival}", arrival, arrival)


def test_standardized_objective_counts_precision_once():
    prior = np.array([0.0])
    state, _ = robust_state_update(
        prior, [record(10, sigma=2)], sparse.csr_matrix((1, 1)), huber=False, delta=1.5,
        lambda_prior=1.0, lambda_spatial=0.0, max_irls=2, tolerance=1e-12,
    )
    # weight=1/sigma^2=0.25 exactly once -> (0.25*10)/(1+0.25)=2
    assert np.isclose(state[0], 2.0)


def test_huber_reduces_extreme_outlier_influence():
    prior = np.array([10.0])
    observations = [record(10), record(10), record(100)]
    robust, _ = robust_state_update(
        prior, observations, sparse.csr_matrix((1, 1)), huber=True, delta=1.5,
        lambda_prior=0.2, lambda_spatial=0, max_irls=20, tolerance=1e-8,
    )
    squared, _ = robust_state_update(
        prior, observations, sparse.csr_matrix((1, 1)), huber=False, delta=1.5,
        lambda_prior=0.2, lambda_spatial=0, max_irls=2, tolerance=1e-8,
    )
    assert abs(robust[0] - 10) < abs(squared[0] - 10)


def test_late_observation_updates_original_state_within_lag():
    twin = FixedLagTwin(2, 5, 3, False, 1.5, 1.0, 0.1, 5, 1e-8, np.full(4, 10.0))
    twin.ingest_at(0, [])
    before_original = twin.states[1, 0]
    twin.ingest_at(3, [record(20, epoch=1, arrival=3, sigma=0.5)])
    assert twin.states[1, 0] > before_original
    assert twin.states[3, 0] > 10.0


def test_on_time_observation_back_smooths_entire_mutable_window():
    twin = FixedLagTwin(2, 5, 3, False, 1.5, 1.0, 0.0, 5, 1e-8, np.full(4, 10.0))
    for epoch in range(3):
        twin.ingest_at(epoch, [])
    before = twin.states[2, 0]
    twin.ingest_at(3, [record(20, epoch=3, arrival=3, sigma=0.5)])
    assert twin.states[2, 0] > before


def test_too_late_observation_is_retrospective_only():
    twin = FixedLagTwin(2, 6, 2, False, 1.5, 1.0, 0.1, 5, 1e-8, np.full(4, 10.0))
    for epoch in range(5):
        twin.ingest_at(epoch, [])
    before = twin.states[0].copy()
    old = record(50, epoch=0, arrival=5)
    twin.ingest_at(5, [old])
    assert np.array_equal(twin.states[0], before)
    assert twin.retrospective_records == [old]


def test_joint_trajectory_matches_dense_linear_system():
    observation = record(4, epoch=1)
    trajectory, diagnostics = robust_trajectory_update(
        np.array([0.0]),
        np.zeros((3, 1)),
        [[], [observation], []],
        sparse.csr_matrix((1, 1)),
        huber=False,
        delta=1.5,
        lambda_temporal=1.0,
        lambda_spatial=0.0,
        max_irls=4,
        tolerance=1e-12,
    )
    system = np.array([[2.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 1.0]])
    expected = np.linalg.solve(system, np.array([0.0, 4.0, 0.0]))
    assert diagnostics.cg_failures == 0
    assert np.allclose(trajectory[:, 0], expected)


def test_adaptive_huber_is_clean_efficient_and_monotone():
    clean = adaptive_huber_state([0.1, -0.4, 0.7, 1.1] * 20)
    mixed = adaptive_huber_state([0.1] * 90 + [4.0] * 10)
    contaminated = adaptive_huber_state([4.0] * 40)
    assert clean.delta == 6.0
    assert np.isclose(clean.residual_location, 0.4)
    assert clean.delta > mixed.delta > contaminated.delta
    assert np.isclose(contaminated.delta, 1.345)


def test_adaptive_twin_uses_previous_epoch_residuals_only():
    twin = FixedLagTwin(
        2,
        2,
        1,
        True,
        1.345,
        1.0,
        0.0,
        5,
        1e-8,
        np.full(4, 10.0),
        adaptive_huber=True,
    )
    twin.ingest_at(0, [record(100, epoch=0, sigma=1.0)])
    assert twin.adaptive_history[0].delta == 6.0
    twin.ingest_at(1, [record(10, epoch=1, sigma=1.0)])
    assert twin.adaptive_history[1].delta <= twin.adaptive_history[0].delta


def test_gated_correction_starts_at_clean_squared_path():
    gated = FixedLagTwin(
        2,
        1,
        0,
        True,
        1.345,
        1.0,
        0.0,
        8,
        1e-8,
        np.full(4, 10.0),
        gated_correction=True,
    )
    squared = FixedLagTwin(
        2,
        1,
        0,
        False,
        1.345,
        1.0,
        0.0,
        8,
        1e-8,
        np.full(4, 10.0),
    )
    observations = [record(12, epoch=0, sigma=1.0), record(13, epoch=0, sigma=1.0)]
    gated.ingest_at(0, observations)
    squared.ingest_at(0, observations)
    assert gated.robust_gate_history[0] == 0.0
    assert np.allclose(gated.states, squared.states)


def test_robust_innovation_fraction_detects_minority_outliers():
    observations = [record(10 + value, sigma=1.0) for value in [0, 0.1, -0.1, 0.2] * 5]
    observations.extend(record(30 + value, sigma=1.0) for value in [0, 1, 2, 3, 4])
    fraction = innovation_contamination_fraction(observations, np.full(4, 10.0))
    assert 0.15 <= fraction <= 0.25


def test_residual_score_correction_has_exact_squared_limit():
    correction, diagnostics, tail_fraction = robust_score_correction_trajectory(
        np.array([[10.0]]),
        [[record(11.0, sigma=1.0)]],
        sparse.csr_matrix((1, 1)),
        delta=3.0,
        lambda_correction=1.0,
        lambda_temporal=1.0,
        lambda_spatial=0.0,
        correction_clip=8.0,
        tolerance=1e-12,
    )
    assert diagnostics.converged
    assert tail_fraction == 0.0
    assert np.array_equal(correction, np.zeros((1, 1)))


def test_residual_score_correction_matches_dense_system():
    correction, diagnostics, tail_fraction = robust_score_correction_trajectory(
        np.zeros((3, 1)),
        [[], [record(4.0, epoch=1, sigma=1.0)], []],
        sparse.csr_matrix((1, 1)),
        delta=1.0,
        lambda_correction=1.0,
        lambda_temporal=1.0,
        lambda_spatial=0.0,
        correction_clip=8.0,
        tolerance=1e-12,
    )
    system = np.array([[3.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 2.0]])
    expected = np.linalg.solve(system, np.array([0.0, -3.0, 0.0]))
    assert diagnostics.converged
    assert tail_fraction == 1.0
    assert np.allclose(correction[:, 0], expected)


def test_residual_correction_twin_preserves_clean_squared_path():
    residual = FixedLagTwin(
        2,
        1,
        0,
        True,
        1.345,
        1.0,
        0.0,
        8,
        1e-8,
        np.full(4, 10.0),
        residual_correction=True,
        correction_delta=100.0,
    )
    squared = FixedLagTwin(
        2,
        1,
        0,
        False,
        1.345,
        1.0,
        0.0,
        8,
        1e-8,
        np.full(4, 10.0),
    )
    observations = [record(12.0, sigma=1.0), record(13.0, sigma=1.0)]
    residual.ingest_at(0, observations)
    squared.ingest_at(0, observations)
    assert residual.correction_tail_fraction_history == [0.0]
    assert np.array_equal(residual.correction_states, np.zeros_like(residual.states))
    assert np.allclose(residual.states, squared.states)


def test_residual_correction_opposes_excess_positive_influence():
    correction, _, tail_fraction = robust_score_correction_trajectory(
        np.array([[10.0]]),
        [[record(25.0, sigma=1.0)]],
        sparse.csr_matrix((1, 1)),
        delta=3.0,
        lambda_correction=0.5,
        lambda_temporal=0.5,
        lambda_spatial=0.0,
        correction_clip=8.0,
        tolerance=1e-12,
    )
    assert tail_fraction == 1.0
    assert -8.0 <= correction[0, 0] < 0.0


def test_guarded_correction_uses_only_closed_residuals():
    twin = FixedLagTwin(
        2,
        2,
        0,
        True,
        1.345,
        1.0,
        0.0,
        8,
        1e-8,
        np.full(4, 10.0),
        residual_correction=True,
        guarded_correction=True,
        correction_clean_gain=0.0,
        correction_attack_gain=4.0,
        correction_gate_start=0.01,
        correction_gate_full=0.02,
        correction_gate_ewma=1.0,
    )
    twin.ingest_at(0, [record(100.0, epoch=0, sigma=1.0)])
    assert twin.robust_gate_history[0] == 0.0
    twin.ingest_at(1, [record(10.0, epoch=1, sigma=1.0)])
    assert twin.robust_gate_history[1] == 4.0


def test_predictive_residual_has_exact_clean_squared_path():
    predictive = FixedLagTwin(
        2,
        1,
        0,
        True,
        1.345,
        1.0,
        0.0,
        8,
        1e-8,
        np.full(4, 10.0),
        predictive_residual=True,
        correction_delta=3.0,
    )
    squared = FixedLagTwin(
        2,
        1,
        0,
        False,
        1.345,
        1.0,
        0.0,
        8,
        1e-8,
        np.full(4, 10.0),
    )
    observations = [record(11.0, sigma=1.0), record(12.0, sigma=1.0)]
    predictive.ingest_at(0, observations)
    squared.ingest_at(0, observations)
    assert predictive.correction_tail_fraction_history == [0.0]
    assert np.allclose(predictive.states, squared.states)


def test_predictive_baseline_is_closed_before_current_batch():
    kwargs = {
        "side": 2,
        "steps": 1,
        "fixed_lag": 0,
        "huber": True,
        "delta": 1.345,
        "lambda_prior": 1.0,
        "lambda_spatial": 0.0,
        "max_irls": 8,
        "tolerance": 1e-8,
        "initial_state": np.full(4, 10.0),
        "predictive_residual": True,
        "correction_delta": 2.0,
    }
    moderate = FixedLagTwin(**kwargs)
    extreme = FixedLagTwin(**kwargs)
    moderate.ingest_at(0, [record(20.0, sigma=1.0)])
    extreme.ingest_at(0, [record(100.0, sigma=1.0)])
    assert np.array_equal(moderate.predictive_states, extreme.predictive_states)
    assert np.array_equal(moderate.release_baselines, extreme.release_baselines)
    assert np.allclose(moderate.states, extreme.states)


def test_predictive_delta_contracts_only_when_current_innovations_show_tail_mass():
    kwargs = {
        "side": 2,
        "steps": 1,
        "fixed_lag": 0,
        "huber": True,
        "delta": 1.345,
        "lambda_prior": 1.0,
        "lambda_spatial": 0.0,
        "max_irls": 8,
        "tolerance": 1e-8,
        "initial_state": np.full(4, 10.0),
        "predictive_residual": True,
        "predictive_adaptive_delta": True,
        "predictive_clean_delta": 4.0,
        "predictive_robust_delta": 2.0,
        "innovation_gate_start": 0.02,
        "innovation_gate_full": 0.12,
    }
    clean = FixedLagTwin(**kwargs)
    attacked = FixedLagTwin(**kwargs)
    clean.ingest_at(0, [record(10.0, sigma=1.0) for _ in range(10)])
    attacked.ingest_at(
        0,
        [record(10.0, sigma=1.0) for _ in range(8)]
        + [record(100.0, sigma=1.0) for _ in range(2)],
    )
    assert clean.predictive_delta_history == [4.0]
    assert attacked.predictive_delta_history == [2.0]


def test_predictive_calibrated_gate_freezes_burn_in_tail_baseline():
    twin = FixedLagTwin(
        side=2,
        steps=2,
        fixed_lag=0,
        huber=True,
        delta=1.345,
        lambda_prior=1.0,
        lambda_spatial=0.0,
        max_irls=8,
        tolerance=1e-8,
        initial_state=np.full(4, 10.0),
        predictive_residual=True,
        predictive_adaptive_delta=True,
        predictive_clean_delta=4.0,
        predictive_robust_delta=2.0,
        predictive_calibrated_gate=True,
        predictive_calibration_epochs=1,
        predictive_excess_gate_start=0.10,
        predictive_excess_gate_full=0.20,
        innovation_gate_tail_z=3.5,
    )
    twin.ingest_at(0, [record(10.0, epoch=0, sigma=1.0) for _ in range(10)])
    twin.ingest_at(
        1,
        [record(10.0, epoch=1, sigma=1.0) for _ in range(8)]
        + [record(100.0, epoch=1, sigma=1.0) for _ in range(2)],
    )
    assert twin.predictive_tail_baseline_history == [0.0, 0.0]
    assert twin.predictive_activation_by_epoch.tolist() == [0.0, 1.0]
    assert twin.predictive_delta_history == [4.0, 2.0]

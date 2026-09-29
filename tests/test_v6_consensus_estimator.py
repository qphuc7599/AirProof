from dataclasses import replace

import numpy as np

from airproof.records import Observation
from airproof.v6_estimator import EstimatorConfig, estimate_public_field


def _observation(user, value):
    return Observation(user, 0, 0, 0, value, 3., 1., 512,
                       f"n-{user}", 0, 0, False)


def test_consensus_gate_uses_mean_for_tight_group_and_preserves_cap():
    public = np.array([[10.]])
    records = [_observation(0, 16.), _observation(1, 16.5), _observation(2, 15.5)]
    cfg = EstimatorConfig(lag=0, lambda_zero=.01, lambda_temporal=0,
                          lambda_spatial=0, loss="consensus_huber",
                          consensus_spread_threshold=1.)
    result = estimate_public_field(public, np.zeros((1, 2)), records, cfg,
                                   innovation_scales=3.)
    ordinary = estimate_public_field(public, np.zeros((1, 2)), records,
                                     replace(cfg, loss="huber"), innovation_scales=3.)
    assert abs(result.live[0, 0]-16.) < .2
    assert abs(ordinary.live[0, 0]-16.) < .2
    assert result.live[0, 0] <= 18.
    assert result.diagnostics["observation_rule"].startswith("same-cell")


def test_dispersion_gate_limits_one_large_disagreement():
    public = np.array([[10.]])
    clean = [_observation(0, 11.), _observation(1, 11.5), _observation(2, 12.)]
    attacked = clean[:2] + [_observation(2, 40.)]
    cfg = EstimatorConfig(lag=0, lambda_zero=.05, lambda_temporal=0,
                          lambda_spatial=0, loss="consensus_huber",
                          consensus_spread_threshold=1.)
    clean_result = estimate_public_field(public, np.zeros((1, 2)), clean, cfg,
                                         innovation_scales=3.)
    attacked_result = estimate_public_field(public, np.zeros((1, 2)), attacked, cfg,
                                            innovation_scales=3.)
    assert abs(attacked_result.live[0, 0]-clean_result.live[0, 0]) < 1.


def test_consensus_estimate_is_invariant_to_future_arrivals():
    public = np.full((3, 1), 10.)
    early = [_observation(0, 12.), _observation(1, 12.5)]
    future = Observation(2, 1, 0, 0, 40., 3., 1., 512, "future", 1, 1, False)
    cfg = EstimatorConfig(lag=1, lambda_zero=.1, lambda_spatial=0,
                          loss="consensus_huber")
    first = estimate_public_field(public, np.zeros((1, 2)), early, cfg,
                                  innovation_scales=3.)
    second = estimate_public_field(public, np.zeros((1, 2)), early+[future], cfg,
                                   innovation_scales=3.)
    assert first.live[0, 0] == second.live[0, 0]


def test_singleton_quadratic_uses_clipped_payload_and_output_envelope():
    public = np.array([[10.]])
    record = [_observation(0, 16.)]
    adaptive = EstimatorConfig(lag=0, lambda_zero=.5, lambda_temporal=0,
        lambda_spatial=0, loss="consensus_huber", input_clip=True,
        clip_delta=4., consensus_singleton_quadratic=True)
    ordinary = replace(adaptive, consensus_singleton_quadratic=False)
    adaptive_result = estimate_public_field(public, np.zeros((1, 2)), record,
                                             adaptive, innovation_scales=3.)
    ordinary_result = estimate_public_field(public, np.zeros((1, 2)), record,
                                             ordinary, innovation_scales=3.)
    assert adaptive_result.live[0, 0] > ordinary_result.live[0, 0]
    huge = estimate_public_field(public, np.zeros((1, 2)), [_observation(0, 1000.)],
                                 adaptive, innovation_scales=3.)
    clipped = estimate_public_field(public, np.zeros((1, 2)), [_observation(0, 22.)],
                                    adaptive, innovation_scales=3.)
    assert huge.live[0, 0] == clipped.live[0, 0] <= 18.

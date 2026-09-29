from dataclasses import replace

import numpy as np

from airproof.records import Observation
from airproof.v6_estimator import EstimatorConfig
from airproof.v8_external_wrapper import (
    PublicEnvelopeConfig,
    envelope_certificate,
    matched_corruption_channel,
    project_to_public_envelope,
    public_context,
    run_equal_information_controls,
)


def test_public_context_and_projection_have_global_history_uniform_envelope():
    primary = np.array([[2., 4.], [3., 5.]])
    auxiliary = np.array([[3., 2.], [9., 1.]])
    cfg = PublicEnvelopeConfig(.8, 1., .5, 3.)
    center, radius = public_context(primary, auxiliary, cfg)
    high = project_to_public_envelope(
        np.full_like(center, 1e6), center, radius, global_radius=3.)
    low = project_to_public_envelope(
        np.full_like(center, -1e6), center, radius, global_radius=3.)
    assert np.max(np.abs(high - center)) <= 3.
    assert np.max(np.abs(low - center)) <= 3.
    assert np.max(np.abs(high - low)) <= 6.
    assert envelope_certificate(center, radius, high, global_radius=3.)["cap_pass"]


def test_public_context_is_prefix_causal_and_does_not_accept_private_inputs():
    primary = np.arange(12, dtype=float).reshape(6, 2)
    auxiliary = primary + 1
    cfg = PublicEnvelopeConfig(.5, 1., 1., 4.)
    a = public_context(primary, auxiliary, cfg)
    changed_primary = primary.copy()
    changed_auxiliary = auxiliary.copy()
    changed_primary[4:] += 100
    changed_auxiliary[4:] += 50
    b = public_context(changed_primary, changed_auxiliary, cfg)
    np.testing.assert_array_equal(a[0][:4], b[0][:4])
    np.testing.assert_array_equal(a[1][:4], b[1][:4])


def test_matched_corruption_changes_only_malicious_post_onset_payloads():
    truth = np.full((24, 3), 10.)
    clean = matched_corruption_channel(truth, seed=70, kind="clean", onset_fraction=.75)
    attacked = matched_corruption_channel(truth, seed=70, kind="drift", onset_fraction=.75)
    metadata = lambda x: (x.user_id, x.epoch, x.cell, x.relay_arrival, x.nullifier)
    assert [metadata(x) for x in clean] == [metadata(x) for x in attacked]
    for before, after in zip(clean, attacked, strict=True):
        if after.epoch < 18 or not after.corrupted:
            assert before.value == after.value


def test_full_live_wrapper_has_no_future_information_and_controls_are_matched():
    public = np.full((5, 1), 10.)
    auxiliary = np.full((5, 1), 11.)
    env = PublicEnvelopeConfig(.8, 2., .5, 4.)
    center, radius = public_context(public, auxiliary, env)
    early = Observation(1, 0, 0, 0, 14., 2., 1., 512, "early", 0, 0)
    cfg = EstimatorConfig(lag=1, lambda_temporal=.1, lambda_spatial=0,
                          lambda_zero=.1, time_step=1., numerical_refinements=1)
    base = run_equal_information_controls(center, radius, np.zeros((1, 2)), [early],
        estimator_config=cfg, innovation_scale=2., global_radius=4.)
    changed_public = public.copy()
    changed_auxiliary = auxiliary.copy()
    changed_public[3:] = 100
    changed_auxiliary[3:] = 80
    changed_center, changed_radius = public_context(changed_public, changed_auxiliary, env)
    future = Observation(2, 3, 0, 0, 0., 2., 1., 512, "future", 3, 3)
    changed = run_equal_information_controls(
        changed_center, changed_radius, np.zeros((1, 2)), [early, future],
        estimator_config=cfg, innovation_scale=2., global_radius=4.)
    for method in base:
        np.testing.assert_array_equal(base[method][0][:3], changed[method][0][:3])
    assert base["adaptive_bounded"][1]["cap_pass"]
    assert base["robust_unbounded"][1]["assimilated_records"] == 1
    assert changed["robust_unbounded"][1]["assimilated_records"] == 2

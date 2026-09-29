import numpy as np
import pytest

from airproof.v8_release_assimilation import (
    ProtectedReleaseAssimilator,
    ReleaseAssimilationConfig,
    persistent_reference_stress,
    published_release_carry,
    run_protected_assimilation,
)


def _config(*, scale=7.0, gate=4.0):
    return ReleaseAssimilationConfig(
        groups=2,
        scheduled_acquisition_epochs=tuple(range(28)),
        laplace_scale=scale,
        public_uncertainty_gate=gate,
    )


def test_registered_privacy_and_clock_contract_is_fixed():
    with pytest.raises(ValueError):
        ReleaseAssimilationConfig(2, tuple(range(27)))
    with pytest.raises(ValueError):
        ReleaseAssimilationConfig(2, tuple(range(28)), deadline_epochs=23)
    with pytest.raises(ValueError):
        ReleaseAssimilationConfig(2, tuple(range(28)), epsilon_history=8.1)
    with pytest.raises(ValueError):
        ReleaseAssimilationConfig(2, tuple(range(28)), k_min=19)


def test_consumer_is_causal_gated_bounded_and_has_no_private_count_input():
    cfg = _config()
    consumer = ProtectedReleaseAssimilator(cfg)
    for epoch in range(24):
        np.testing.assert_array_equal(
            consumer.update(epoch, np.array([10.0, 20.0]), np.array([5.0, 3.0])),
            [10.0, 20.0],
        )
    prediction = consumer.update(
        24,
        np.array([10.0, 20.0]),
        np.array([5.0, 3.0]),
        acquisition_epoch=0,
        protected_values=np.array([1000.0, 1000.0]),
        released_mask=np.array([True, True]),
    )
    assert prediction[0] <= 18.0
    assert prediction[1] == 20.0
    diagnostics = consumer.diagnostics()
    assert diagnostics["privacy_spend_added_by_postprocessing"] == 0.0
    assert not diagnostics["exact_private_counts_used"]
    assert not diagnostics["raw_records_used"]
    assert not diagnostics["noiseless_query_used"]
    with pytest.raises(ValueError):
        consumer.update(
            25,
            np.array([10.0, 20.0]),
            np.array([5.0, 5.0]),
            acquisition_epoch=0,
            protected_values=np.array([10.0, 20.0]),
            released_mask=np.array([True, True]),
        )


def test_suppression_is_missing_and_fixed_support_is_enforced():
    public = np.full((52, 2), 10.0)
    uncertainty = np.full_like(public, 5.0)
    values = np.full_like(public, np.nan)
    mask = np.zeros_like(public, dtype=bool)
    values[0, 0] = 12.0
    mask[0, 0] = True
    prediction, diagnostics = run_protected_assimilation(
        public, uncertainty, values, mask, _config()
    )
    assert np.array_equal(prediction[:24], public[:24])
    assert diagnostics["assimilated_releases"] == 1
    assert diagnostics["suppressed_releases"] == 55
    bad = values.copy()
    bad[30, 0] = 1.0
    bad_mask = mask.copy()
    bad_mask[30, 0] = True
    with pytest.raises(ValueError):
        run_protected_assimilation(public, uncertainty, bad, bad_mask, _config())


def test_reference_stress_discloses_uncertainty_but_not_latent_bias_to_consumer():
    public = np.full((4, 2), 10.0)
    uncertainty = np.full_like(public, 3.0)
    stressed, inflated = persistent_reference_stress(
        public, uncertainty, np.array([4.0, -4.0])
    )
    np.testing.assert_allclose(stressed, [[14.0, 6.0]] * 4)
    np.testing.assert_allclose(inflated, 5.0)


def test_direct_release_control_uses_publication_clock_and_full_support():
    public = np.arange(104.0).reshape(52, 2)
    values = np.full_like(public, np.nan)
    mask = np.zeros_like(public, dtype=bool)
    values[0] = [7.0, 8.0]
    mask[0] = True
    prediction = published_release_carry(public, values, mask, _config())
    np.testing.assert_array_equal(prediction[:24], public[:24])
    np.testing.assert_array_equal(prediction[24], [7.0, 8.0])
    assert prediction.shape == public.shape

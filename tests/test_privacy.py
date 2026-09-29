import math

import numpy as np

from airproof.privacy import (
    PrivacyAccountant,
    laplace_threshold_margin,
    release_group_mean,
    release_group_residual_mean,
)
from airproof.records import Observation, raw_nullifier, release_nullifier


def record(user, epoch=0, value=5.0):
    return Observation(user, epoch, 0, 0, value, 1.0, 1.0, 100, f"n-{user}-{epoch}-{value}", 0, 0)


def test_user_history_aggregated_then_clipped_once():
    records = [record(1, value=100), record(1, value=100), record(2, value=0)]
    result = release_group_mean(
        records,
        group=0,
        epoch=0,
        clip=10,
        k_min=2,
        epsilon_mean=1e12,
        epsilon_count=0,
        accountant=PrivacyAccountant(2e12),
        rng=np.random.default_rng(1),
    )
    assert result.distinct_users == 2
    assert result.released
    assert math.isclose(result.value, 5.0, abs_tol=1e-8)


def test_sensitivity_and_suppression_symbol():
    result = release_group_mean(
        [record(1)], group=0, epoch=0, clip=50, k_min=20, epsilon_mean=1,
        epsilon_count=0, accountant=PrivacyAccountant(10), rng=np.random.default_rng(2)
    )
    assert result.sensitivity == 5.0
    assert result.value is None
    assert not result.released


def test_private_threshold_spends_budget_even_when_suppressed():
    accountant = PrivacyAccountant(2.0)
    result = release_group_mean(
        [record(1)], group=0, epoch=0, clip=10, k_min=100, epsilon_mean=1,
        epsilon_count=0.25, accountant=accountant, rng=np.random.default_rng(3),
        private_eligibility=True,
    )
    assert not result.released
    assert accountant.spent[1] == 0.25


def test_private_threshold_margin_matches_laplace_tail_and_hides_count():
    margin = laplace_threshold_margin(0.5, 0.01)
    assert math.isclose(0.5 * math.exp(-0.5 * margin), 0.01)
    result = release_group_mean(
        [record(user) for user in range(20)],
        group=0,
        epoch=0,
        clip=10,
        k_min=20,
        epsilon_mean=1,
        epsilon_count=0.5,
        accountant=PrivacyAccountant(10),
        rng=np.random.default_rng(5),
        private_eligibility=True,
        false_release_probability=0.01,
    )
    assert "distinct_users" not in result.protected_payload()
    assert result.eligibility_threshold > 20


def test_raw_and_release_nullifier_domains_do_not_collide():
    secret = bytes(range(32))
    raw = raw_nullifier(
        secret, city="hcmc", group=0, epoch=1, interval=1, policy_id="p"
    )
    released = release_nullifier(secret, city="hcmc", group=0, epoch=1, policy_id="p")
    assert raw != released
    assert released == release_nullifier(secret, city="hcmc", group=0, epoch=1, policy_id="p")


def test_residual_release_reduces_sensitivity_without_changing_center():
    result = release_group_residual_mean(
        [record(1, value=49.0), record(2, value=51.0)],
        group=0,
        epoch=0,
        public_baseline=50.0,
        residual_clip=5.0,
        k_min=2,
        epsilon_mean=1e12,
        epsilon_count=0.0,
        accountant=PrivacyAccountant(2e12),
        rng=np.random.default_rng(8),
    )
    assert result.released
    assert math.isclose(result.value, 50.0, abs_tol=1e-8)
    assert result.sensitivity == 5.0
    assert result.transcript_scope.startswith("residual-")

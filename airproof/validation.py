from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scipy import sparse

from .audit import (
    EpochAnchorRegistry,
    ReceiptAccountabilityLog,
    ReplicatedObjectStore,
    TransparencyLog,
    verify_consistency,
    verify_inclusion,
)
from .integrity import AtomicNullifierStore, MerkleTree, verify_proof
from .privacy import PrivacyAccountant, release_group_mean, stabilized_bounded_mean
from .records import Observation, raw_nullifier, release_nullifier
from .scheduler import select_evidence
from .twin import (
    FixedLagTwin,
    observation_variance,
    robust_score_correction_trajectory,
    robust_state_update,
    robust_trajectory_update,
    standardized_irls_weight,
)


def _record(user: int, group: int, epoch: int, value: float = 10.0, arrival: int = 0) -> Observation:
    return Observation(
        user,
        epoch,
        group,
        group,
        value,
        1.0,
        1.0,
        100,
        f"n-{user}-{group}-{epoch}",
        arrival,
        arrival,
    )


def run_p0_fixtures() -> dict[str, bool]:
    """Run the report's analytical fixtures as executable claim gates."""
    checks: dict[str, bool] = {}

    repeated = [_record(7, 0, 0, value=float(index)) for index in range(12)]
    repeated_release = release_group_mean(
        repeated,
        group=0,
        epoch=0,
        clip=50.0,
        k_min=1,
        epsilon_mean=1.0,
        epsilon_count=0.0,
        accountant=PrivacyAccountant(10.0),
        rng=np.random.default_rng(1),
    )
    checks["DP-01_one_epoch_aggregate"] = repeated_release.distinct_users == 1

    secret = bytes(range(32))
    release_a = release_nullifier(
        secret, city="hcmc", group=1, epoch=5, policy_id="policy-a"
    )
    release_b = release_nullifier(
        secret, city="hcmc", group=1, epoch=5, policy_id="policy-a"
    )
    store = AtomicNullifierStore()
    checks["DP-02_release_duplicate_rejected"] = store.accept_once(
        release_a, policy_id="policy-a", release_domain="release"
    ) and not store.accept_once(release_b, policy_id="policy-a", release_domain="release")
    store.close()
    checks["DP-03_adjacent_epoch_unlinkable"] = release_a != release_nullifier(
        secret, city="hcmc", group=1, epoch=6, policy_id="policy-a"
    )
    checks["DP-04_raw_intervals_distinct"] = raw_nullifier(
        secret, city="hcmc", group=1, epoch=5, interval=1, policy_id="policy-a"
    ) != raw_nullifier(
        secret, city="hcmc", group=1, epoch=5, interval=2, policy_id="policy-a"
    )

    lower = stabilized_bounded_mean([-50.0] * 20, clip=50.0, k_min=20)
    replaced = stabilized_bounded_mean([50.0, *([-50.0] * 19)], clip=50.0, k_min=20)
    checks["DP-05_replace_sensitivity_tight"] = math.isclose(abs(replaced - lower), 5.0)
    add_before = stabilized_bounded_mean([50.0] * 19, clip=50.0, k_min=20)
    add_after = stabilized_bounded_mean([50.0] * 20, clip=50.0, k_min=20)
    checks["DP-06_threshold_add_remove_bounded"] = abs(add_after - add_before) <= 5.0
    payload_keys = set(repeated_release.protected_payload())
    checks["DP-07_protected_interface_excludes_exact_count"] = "distinct_users" not in payload_keys
    accountant = PrivacyAccountant(2.0)
    for epoch in range(20):
        accountant.spend(1, 0.1, group=0, epoch=epoch, mechanism="mean")
    checks["DP-08_longitudinal_accounting"] = math.isclose(accountant.spent[1], 2.0)

    two_group = [_record(user, user % 2, 0) for user in range(10, 16)]
    result = select_evidence(
        two_group, budget_bytes=300, reserve_fraction=1.0, targets={0: 10, 1: 10}
    )
    checks["FAIR-01_two_groups_three_slots"] = sorted(result.counts.values()) == [1, 2]
    three_group = [_record(user, user % 3, 0) for user in range(20, 29)]
    result = select_evidence(
        three_group, budget_bytes=500, reserve_fraction=1.0, targets={0: 10, 1: 10, 2: 10}
    )
    checks["FAIR-02_three_groups_balanced"] = (
        max(result.counts.values()) - min(result.counts.values()) <= 1
    )
    result = select_evidence(
        [_record(1, 0, 0), _record(2, 0, 0)],
        budget_bytes=200,
        reserve_fraction=1.0,
        targets={0: 2, 1: 2},
    )
    checks["FAIR-03_shortfall_explicit"] = 1 in result.opportunity_shortfall_groups
    hard_result = select_evidence(
        [_record(10, 0, 0), _record(11, 0, 0), _record(12, 1, 0), _record(13, 1, 0)],
        budget_bytes=400,
        reserve_fraction=0.1,
        targets={0: 2, 1: 2},
        fairness_strength=1.0,
        constraint_mode="hard_if_feasible",
    )
    checks["FAIR-04_hard_floor_when_globally_feasible"] = (
        hard_result.globally_feasible and hard_result.constraint_satisfied
    )

    weight_sigma_1 = standardized_irls_weight(
        quality=1.0, sigma=1.0, residual_standard=0.0, huber=True, delta=1.5
    )
    weight_sigma_2 = standardized_irls_weight(
        quality=1.0, sigma=2.0, residual_standard=0.0, huber=True, delta=1.5
    )
    checks["ROB-01_precision_ratio_four"] = math.isclose(weight_sigma_1 / weight_sigma_2, 4.0)
    outlier_weight = standardized_irls_weight(
        quality=1.0, sigma=1.0, residual_standard=15.0, huber=True, delta=1.5
    )
    checks["ROB-02_huber_weight_tenth"] = math.isclose(outlier_weight, 0.1)
    prior = np.array([0.0])
    obs = [_record(1, 0, 0, value=0.5), _record(2, 0, 0, value=1.0)]
    laplacian = sparse.csr_matrix((1, 1))
    huber_state, _ = robust_state_update(
        prior,
        obs,
        laplacian,
        huber=True,
        delta=10.0,
        lambda_prior=1.0,
        lambda_spatial=0.0,
        max_irls=5,
        tolerance=1e-10,
    )
    gls_state, _ = robust_state_update(
        prior,
        obs,
        laplacian,
        huber=False,
        delta=1.5,
        lambda_prior=1.0,
        lambda_spatial=0.0,
        max_irls=5,
        tolerance=1e-10,
    )
    checks["ROB-03_quadratic_region_matches_gls"] = np.allclose(huber_state, gls_state)
    checks["ROB-04_path_variance_separated"] = math.isclose(
        observation_variance(1.0, 2.0, 3.0, dp_laplace_scale=2.0)
        - observation_variance(1.0, 2.0, 3.0),
        8.0,
    )
    correction_zero, _, clean_tail = robust_score_correction_trajectory(
        np.array([[10.0]]),
        [[_record(1, 0, 0, value=11.0)]],
        laplacian,
        delta=3.0,
        lambda_correction=1.0,
        lambda_temporal=1.0,
        lambda_spatial=0.0,
        correction_clip=8.0,
        tolerance=1e-12,
    )
    checks["ROB-05_residual_squared_limit"] = clean_tail == 0.0 and np.array_equal(
        correction_zero, np.zeros((1, 1))
    )
    correction_tail, _, attack_tail = robust_score_correction_trajectory(
        np.array([[10.0]]),
        [[_record(1, 0, 0, value=25.0)]],
        laplacian,
        delta=3.0,
        lambda_correction=0.5,
        lambda_temporal=0.5,
        lambda_spatial=0.0,
        correction_clip=8.0,
        tolerance=1e-12,
    )
    checks["ROB-06_excess_influence_opposed"] = (
        attack_tail == 1.0 and -8.0 <= correction_tail[0, 0] < 0.0
    )
    predictive_kwargs = {
        "side": 2,
        "steps": 1,
        "fixed_lag": 0,
        "huber": True,
        "delta": 1.345,
        "lambda_prior": 1.0,
        "lambda_spatial": 0.0,
        "max_irls": 8,
        "tolerance": 1e-10,
        "initial_state": np.full(4, 10.0),
        "predictive_residual": True,
        "correction_delta": 2.0,
    }
    moderate_twin = FixedLagTwin(**predictive_kwargs)
    extreme_twin = FixedLagTwin(**predictive_kwargs)
    moderate_twin.ingest_at(0, [_record(1, 0, 0, value=20.0)])
    extreme_twin.ingest_at(0, [_record(1, 0, 0, value=100.0)])
    checks["ROB-07_predictor_precedes_current_batch"] = np.array_equal(
        moderate_twin.predictive_states, extreme_twin.predictive_states
    ) and np.array_equal(
        moderate_twin.release_baselines, extreme_twin.release_baselines
    )
    predictive_clean = FixedLagTwin(
        **{**predictive_kwargs, "correction_delta": 3.0}
    )
    squared_clean = FixedLagTwin(
        2, 1, 0, False, 1.345, 1.0, 0.0, 8, 1e-10, np.full(4, 10.0)
    )
    clean_observations = [_record(1, 0, 0, value=11.0), _record(2, 0, 0, value=12.0)]
    predictive_clean.ingest_at(0, clean_observations)
    squared_clean.ingest_at(0, clean_observations)
    checks["ROB-08_predictive_quadratic_equivalence"] = np.allclose(
        predictive_clean.states, squared_clean.states
    )

    late = Observation(1, 1, 0, 0, 20.0, 0.5, 1.0, 100, "late", 3, 3)
    twin = FixedLagTwin(2, 5, 3, False, 1.5, 1.0, 0.1, 5, 1e-9, np.full(4, 10.0))
    twin.ingest_at(0, [])
    before = twin.states[1, 0]
    twin.ingest_at(3, [late])
    checks["LATE-01_updates_acquisition_state"] = twin.states[1, 0] > before + 1.0
    immutable_before = twin.states[0].copy()
    too_late = Observation(2, 0, 0, 0, 40.0, 1.0, 1.0, 100, "too-late", 4, 4)
    twin.ingest_at(4, [too_late])
    checks["LATE-02_pre_window_immutable"] = np.array_equal(
        immutable_before, twin.states[0]
    ) and too_late in twin.retrospective_records

    trajectory_observation = _record(3, 0, 1, value=4.0)
    trajectory, _ = robust_trajectory_update(
        np.array([0.0]),
        np.zeros((3, 1)),
        [[], [trajectory_observation], []],
        sparse.csr_matrix((1, 1)),
        huber=False,
        delta=1.5,
        lambda_temporal=1.0,
        lambda_spatial=0.0,
        max_irls=4,
        tolerance=1e-12,
    )
    dense_system = np.array([[2.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 1.0]])
    dense_solution = np.linalg.solve(dense_system, np.array([0.0, 4.0, 0.0]))
    checks["LATE-03_block_solver_matches_dense"] = np.allclose(
        trajectory[:, 0], dense_solution, atol=1e-8
    )

    tree = MerkleTree([b"alpha", b"beta", b"gamma"])
    proof = tree.proof(tree.payloads.index(b"beta"))
    checks["LEDGER-01_tamper_rejected"] = verify_proof(
        b"beta", proof, tree.root
    ) and not verify_proof(b"betx", proof, tree.root)
    store = AtomicNullifierStore()
    with ThreadPoolExecutor(max_workers=16) as pool:
        accepted = list(pool.map(lambda _: store.accept_once("same-nullifier"), range(100)))
    store.close()
    checks["LEDGER-02_concurrent_replay_exactly_once"] = sum(accepted) == 1

    signing_key = Ed25519PrivateKey.generate()
    receipts = ReceiptAccountabilityLog(signing_key)
    receipt = receipts.issue(b"ciphertext", policy_id="p", accepted_at=0, inclusion_deadline=1)
    checks["LEDGER-03_unresolved_receipt_detected"] = receipt.receipt_id in receipts.audit(
        now=2
    )["unresolved"]
    object_store = ReplicatedObjectStore(2)
    digest = object_store.put(b"ciphertext")
    anchored_root = tree.root
    object_store.delete_replica(digest, 0)
    object_store.delete_replica(digest, 1)
    checks["LEDGER-04_deletion_is_availability_failure"] = (
        object_store.get(digest) is None and tree.root == anchored_root
    )
    registry = EpochAnchorRegistry()
    registry.anchor(city="hcmc", epoch=1, version=0, root_hex=tree.root_hex)
    try:
        registry.anchor(city="hcmc", epoch=1, version=0, root_hex="00" * 32)
        competing_rejected = False
    except ValueError:
        competing_rejected = True
    checks["LEDGER-05_competing_root_rejected"] = competing_rejected

    log = TransparencyLog(signing_key)
    for value in (b"a", b"b", b"c", b"d", b"e"):
        log.append(value)
    inclusion = log.inclusion_proof(3)
    consistency = log.consistency_proof(3)
    checks["LOG_inclusion_and_consistency"] = verify_inclusion(
        b"d", index=3, tree_size=5, proof=inclusion, expected_root=log.root()
    ) and verify_consistency(
        old_size=3,
        new_size=5,
        old_root=log.root(3),
        new_root=log.root(),
        proof=consistency,
    )
    protected = repeated_release.protected_payload()
    forbidden_fields = {"user_id", "cell", "nullifier", "distinct_users", "direct_arrival"}
    checks["PRIVACY-LEAK-01_no_raw_fields"] = not forbidden_fields.intersection(protected)

    checks = {key: bool(value) for key, value in checks.items()}
    checks["all"] = all(checks.values())
    return checks

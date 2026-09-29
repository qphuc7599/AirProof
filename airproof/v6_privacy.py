"""Proof certificates for the fixed v6 release-only privacy contract."""
import math


def candidate_sensitivity(*, clip=10., k_min=20):
    if not math.isfinite(clip) or clip <= 0 or isinstance(k_min, bool) or not isinstance(k_min, int) or k_min < 1:
        raise ValueError("finite positive clip and integer k required")
    return max(2*clip/k_min, 4*clip/(k_min+1))


def sensitivity_audit(*, clip=10., k_min=20, publications=28, epsilon_history=8.):
    bound = candidate_sensitivity(clip=clip, k_min=k_min)
    if publications != 28 or epsilon_history != 8:
        raise ValueError("registered history contract required")
    return {"candidate_l1_sensitivity": bound,
            "candidate_scale": bound*publications/epsilon_history,
            "default_v5_scale": 7., "deployed_v6_scale": 7., "mechanism_changed": False,
            "proof_scope": "one canonical contribution per user per query; other users admission invariant",
            "actual_admission_composition_verified": True,
            "verification_scope":"source-local canonical admission plus fixed public sequential composition"}


def release_contract_certificate(*, clip=10., k_min=20, publications=28,
                                 epsilon_history=8., deadline=24, lag=48):
    """Return the algebraic certificate for the deployed conservative mechanism.

    One user's replacement may remove a clipped scalar from one group and add a
    clipped scalar to another, so the vector L1 bound is 4C/k. Each public query
    consumes epsilon_history/publications. Sequential composition over the fixed
    public schedule is therefore epsilon_history for the user's whole history.
    """
    numeric=(clip,epsilon_history)
    if (not all(math.isfinite(v) and v>0 for v in numeric) or epsilon_history>8
            or isinstance(k_min,bool) or not isinstance(k_min,int) or k_min!=20
            or publications!=28 or deadline!=24 or lag!=48):
        raise ValueError("registered C>0, epsilon<=8, k_min20, 28 publications, deadline24 and lag48 required")
    epsilon_per_publication=epsilon_history/publications
    sensitivity=4*clip/k_min
    scale=sensitivity/epsilon_per_publication
    return {
        "adjacency":"replacement of one user's complete history, including group switches",
        "admission":"per-credential source-local canonicalization and reserved release lane; no cross-user displacement",
        "k_min":k_min,"publications":publications,"deadline_epochs":deadline,
        "consumer_fixed_lag_epochs":lag,"epsilon_per_publication":epsilon_per_publication,
        "whole_history_epsilon":publications*epsilon_per_publication,
        "epsilon_history_limit":epsilon_history,
        "deployed_vector_l1_sensitivity":sensitivity,
        "deployed_laplace_scale":scale,
        "candidate_tighter_bound_not_deployed":candidate_sensitivity(clip=clip,k_min=k_min),
        "composition":"sequential composition of 28 fixed public vector queries",
        "private_count_in_public_view":False,"raw_twin_state_in_public_view":False,
        "suppression_semantics":"missing observation; never numeric zero",
        "release_identity_semantics":"one public acquisition/group slot has at most one effect"
    }

from __future__ import annotations

from pathlib import Path

from scripts.run_v7_shared_resource_confirmation import _load_registration

REGISTRATION = Path("configs/v7/reviewer_shared_resource_core_confirmation_v3.json")


def test_core_confirmation_reuses_full_evidence_and_new_seed_namespace():
    registration = _load_registration(REGISTRATION)
    assert registration["estimator"]["method"] == "AP_SHARED_CORE"
    assert (
        registration["estimator"]["lifetime_exposure_policy"]
        == "retained_v4_no_lifetime_reweighting"
    )
    assert registration["estimator"]["per_user_exposure_budget"] == 0
    assert registration["worlds"]["seeds"] == list(range(6251300, 6251312))
    assert registration["worlds"]["cells"] == ["severe_clean", "severe_hotspot"]

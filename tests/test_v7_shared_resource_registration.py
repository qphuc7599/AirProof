from __future__ import annotations

from scripts.run_v7_shared_resource_confirmation import (
    _config,
    _estimator,
    _load_registration,
)


def test_registered_matrix_and_resource_contract_are_exact():
    registration = _load_registration()
    assert len(registration["worlds"]["seeds"]) == 12
    assert registration["worlds"]["cells"] == ["severe_clean", "severe_hotspot"]
    config = _config(registration, "severe_clean")
    contract = registration["execution"]
    assert config["world"]["agents"] == 1000
    assert config["world"]["steps"] == 672
    assert config["transport"]["capacity_bytes_per_direction"] == contract[
        "contact_capacity_bytes_per_direction"
    ]
    assert config["transport"]["release_gateway_reservation_bytes"] == contract[
        "release_reservation_bytes_per_gateway_contact"
    ]
    assert config["scheduler"]["algorithm"] == contract["allocation_policy"]
    assert config["audit"]["public_checkpoint"]["enabled"] is False
    assert config["audit"]["return_mode"] == "incremental_selected_receipts_v1"


def test_registered_estimator_is_the_validated_budget7_candidate():
    registration = _load_registration()
    estimator = _estimator(registration)
    assert estimator.cap == 8
    assert estimator.lag == 6
    assert estimator.lambda_zero == 0.05
    assert estimator.lambda_temporal == 0.05
    assert estimator.lambda_spatial == 0.02
    assert registration["estimator"]["per_user_exposure_budget"] == 7
    assert registration["estimator"]["controls"] == [
        "PUBLIC",
        "SQ_EXPOSURE_MATCHED",
        "HUBER_EXPOSURE_MATCHED",
    ]

from airproof.config import load_config
from airproof.validation import run_p0_fixtures


def test_claim_gate_ids_exist_in_executable_p0_registry():
    claims = load_config("configs/claims_registry.yaml")
    checks = run_p0_fixtures()
    executable = set(checks) - {"all"}
    required = {
        fixture
        for claim in claims["claims"].values()
        for fixture in claim.get("gate", [])
    }
    assert required <= executable


def test_local_confirmatory_is_frozen_and_full_anchor_remains_deferred():
    registry = load_config("configs/experiment_registry.yaml")
    assert registry["status"] == "airproof24h-confirmatory-frozen"
    assert registry["local_confirmatory"]["status"] == "authorized-frozen"
    assert {study["id"] for study in registry["deferred_large"]} >= {"E1", "E3", "E4", "E5"}

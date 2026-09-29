import numpy as np
import pytest

from airproof.v6_mechanistic_provenance import (
    assert_causal_availability,
    assert_prediction_schema,
    assert_producer_manifest,
)


def manifest(**updates):
    value = {
        "producer_id": "fixed-causal-producer-v1",
        "output_fields": ["innovation_variance", "transition_operator", "residual_forcing"],
        "prediction_epochs": [10, 11, 12],
        "latest_input_epochs": [10, 10, 12],
        "source_hashes": {"producer.py": "abc"},
        "uses_hidden_truth": False,
        "uses_attack_flags": False,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize("field", ["truth", "target", "attack_flag", "attack_label", "future_public"])
def test_prediction_schema_rejects_hidden_outcome_fields(field):
    with pytest.raises(ValueError, match="forbidden prediction inputs"):
        assert_prediction_schema(["public_center", field])


def test_future_input_is_rejected_at_prediction_boundary():
    with pytest.raises(ValueError, match="future producer input"):
        assert_causal_availability(np.array([4, 5]), np.array([4, 6]))


def test_manifest_rejects_hidden_truth_and_attack_flags():
    with pytest.raises(ValueError, match="hidden truth"):
        assert_producer_manifest(manifest(uses_hidden_truth=True))
    with pytest.raises(ValueError, match="attack flags"):
        assert_producer_manifest(manifest(uses_attack_flags=True))


def test_manifest_accepts_only_causal_scoring_blind_producer():
    assert_producer_manifest(manifest()) is None


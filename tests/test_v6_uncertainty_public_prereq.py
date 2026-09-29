from pathlib import Path
import json

import numpy as np

from airproof.v6_uncertainty_public_prereq import (
    ARRAY_KEYS, artifact_identity, array_digest, prove_historical_cell_invariance,
    require_unambiguous_publish_state, target_inventory,
)
import pytest


def arrays(offset=0):
    return {"live": np.ones((2, 2)) + offset,
            "reconstructed": np.ones((2, 2)) + offset,
            "truth": np.ones((2, 2)) + offset,
            "public": np.ones((2, 2)) + offset,
            "cell_groups": np.array([0, 1]),
            "scales": np.ones((2, 2))}


def save(path, values):
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, **values)


def test_cell_bound_identity_changes_without_changing_array_digest():
    digest = array_digest(arrays())
    first = artifact_identity(seed=1, cell="a", source_digest="s",
                              configuration_hash="c1", array_digest=digest)
    second = artifact_identity(seed=1, cell="b", source_digest="s",
                               configuration_hash="c2", array_digest=digest)
    assert first != second


def test_historical_invariance_requires_exact_arrays(tmp_path):
    for cell in ("a", "b"):
        save(tmp_path / f"1_{cell}" / "PUBLIC.npz", arrays())
    proof = prove_historical_cell_invariance(tmp_path, [1], ["a", "b"])
    assert proof["exact_equal"] is True
    assert proof["array_keys"] == list(ARRAY_KEYS)


def test_target_inventory_exposes_partial_state(tmp_path):
    paths, present = target_inventory(tmp_path, [1, 2], ["a", "b"])
    assert len(paths) == 4 and present == []
    save(paths[0], arrays())
    _, present = target_inventory(tmp_path, [1, 2], ["a", "b"])
    assert present == [paths[0]]
    with pytest.raises(RuntimeError, match="ambiguous partial"):
        require_unambiguous_publish_state(paths, present, tmp_path / "staging")


def test_publish_state_refuses_staging_and_reports_empty(tmp_path):
    expected = [tmp_path / "final" / "one"]
    assert require_unambiguous_publish_state(expected, [], tmp_path / "staging") == "empty"
    (tmp_path / "staging").mkdir()
    with pytest.raises(RuntimeError, match="will not be overwritten"):
        require_unambiguous_publish_state(expected, [], tmp_path / "staging")


def test_v3_1_scores_each_registered_seed_once():
    config = json.loads(Path("configs/v6_uncertainty_event_v3_1.json").read_text())
    seeds = [seed for role in ("fit", "calibration", "evaluation")
             for seed in config[role]["seeds"]]
    assert seeds == list(range(6202000, 6202008))
    assert config["canonical_public_cell"] == "anchor_clean"
    assert "once" in config["physical_cell_policy"]

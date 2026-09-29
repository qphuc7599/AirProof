import copy
import json
import numpy as np
import pytest
from scripts.analyze_v5_uncertainty import (fit_radii, apply_radii, world_bootstrap_indices,
                                           bootstrap_summary, freeze, evaluate, METHODS)


def test_frozen_quantiles_ignore_test_residuals_and_attack_label():
    errors = np.arange(200)/100
    fitted = fit_radii([errors], {"group=0": [errors[:120]], "group=1": [errors[120:]]})
    frozen_copy = copy.deepcopy(fitted)
    predictions = np.full((4, 2), 10.)
    groups = np.array([0, 1])
    a = apply_radii(fitted, predictions, "clean", groups)
    b = apply_radii(fitted, predictions, "attack", groups)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    # Evaluation targets are deliberately not an input to the interval function.
    assert fitted == frozen_copy
    assert fitted["strata"]["group=1"]["fallback"]
    np.testing.assert_allclose(a[1][0, 1]-10., fitted["pooled_radii"])


def test_world_bootstrap_preserves_world_blocks_and_pairing():
    indices = world_bootstrap_indices(2, 100, 9)
    np.testing.assert_array_equal(indices, world_bootstrap_indices(2, 100, 9))
    assert indices.shape == (100, 2)
    rows = []
    for method in ("AP", "SQ"):
        for seed, value in ((10, .2), (11, .8)):
            for cell in ("a", "b"):
                rows.append({"seed": seed, "method": method, "clock": "live", "stratum": "pooled",
                             "level": .9, "n": 50, "coverage": value, "mean_width": 2., "mean_interval_score": 3.})
    summaries = bootstrap_summary(rows, [10, 11], indices)
    assert summaries[0]["metrics"] == summaries[1]["metrics"]
    assert summaries[0]["metrics"]["coverage"] == {"estimate": .5, "lower95": .2, "upper95": .8}
    assert summaries[0]["n"] == 200
    assert summaries[0]["supported_worlds"] == 2


def write_campaign(root, stage, seeds, selected):
    root.mkdir()
    manifest = {"stage": stage, "seeds": seeds, "cells": ["clean"], "source_hash": "source",
                "protocol_hash": "protocol", "base_hash": "base", "overrides": {}, "selected": selected}
    (root/"manifest.json").write_text(json.dumps(manifest))
    for seed in seeds:
        folder = root/"jobs"/str(seed)/"clean"; folder.mkdir(parents=True)
        for method in METHODS:
            truth = np.full((30, 2), 10.)
            np.savez(folder/f"{method}.npz", truth=truth, live=truth+1, reconstructed=truth+.5, cell_groups=np.array([0, 1]))
            (folder/f"{method}.json").write_text(json.dumps({"seed": seed, "method": method, "cell": "clean", "source_hash": "source"}))


def test_disjoint_freeze_evaluate_and_tamper_rejection(tmp_path):
    selected = {"objective": "residual"}
    selected_path = tmp_path/"selected.json"; selected_path.write_text(json.dumps(selected))
    training = tmp_path/"training.json"; training.write_text(json.dumps({"seeds": [1, 2]}))
    calibration = tmp_path/"calibration"; write_campaign(calibration, "calibration", list(range(10, 18)), selected)
    frozen = tmp_path/"frozen"
    freeze(calibration, selected_path, frozen, burn=0, training_manifest=training)
    validation = tmp_path/"validation"; write_campaign(validation, "validation", [20, 21], selected)
    before = (frozen/"frozen_quantiles.json").read_bytes()
    evaluate(validation, frozen/"frozen_quantiles.json", tmp_path/"evaluated", draws=20)
    assert (frozen/"frozen_quantiles.json").read_bytes() == before
    overlapping = tmp_path/"overlap"; write_campaign(overlapping, "validation", [10, 21], selected)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(overlapping, frozen/"frozen_quantiles.json", tmp_path/"bad", draws=20)
    (frozen/"frozen_quantiles.json").write_bytes(before+b" ")
    with pytest.raises(ValueError, match="changed"):
        evaluate(validation, frozen/"frozen_quantiles.json", tmp_path/"tampered", draws=20)

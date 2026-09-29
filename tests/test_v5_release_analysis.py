import json
import numpy as np
import pytest
from airproof.v5_release_twin import ReleaseCalibration, PublicReleaseObservation
from scripts.analyze_v5_release_twin import protected_inputs, predict_release_only, freeze_dense


def test_protected_loader_ignores_private_arrays_and_preserves_clock(tmp_path):
    path = tmp_path/"release.npz"
    baseline = np.full((672, 1), 20.)
    values = np.full_like(baseline, np.nan); values[::24] = 23.
    mask = np.isfinite(values); mask[0] = False
    np.savez(path, baseline=baseline, residual=values, residual_mask=mask,
             truth=np.array(["forbidden"]), residual_query=np.array(["forbidden"]),
             raw=np.array(["forbidden"]), counts=np.array(["forbidden"]))
    metadata = {"residual": {"laplace_variance": 98.}, "deadline": 24,
                "scheduled_epochs": list(range(0, 672, 24))}
    public, releases = protected_inputs(path, metadata)
    np.testing.assert_array_equal(public, baseline)
    assert releases[0].value is None
    assert releases[-1].acquisition_epoch == 648
    assert releases[-1].publication_epoch == 672
    assert releases[-1].laplace_scale == 7.
    metadata["residual"]["laplace_variance"] = 24.5
    with pytest.raises(ValueError, match="scale7"):
        protected_inputs(path, metadata)


def test_delayed_live_drain_and_suppression():
    cal = ReleaseCalibration(np.zeros(2), np.eye(2)*.9, np.eye(2), np.eye(2), 1)
    public = np.full((30, 1), 20.)
    releases = [PublicReleaseObservation("r", 0, 24, 0, 30., 7.),
                PublicReleaseObservation("final", 29, 53, 0, 35., 7.),
                PublicReleaseObservation("suppressed", 1, 25, 0, None, 7.)]
    predictions, diagnostics = predict_release_only(public, releases, cal)
    np.testing.assert_array_equal(predictions["CALIBRATED"][0][:24], public[:24])
    assert predictions["DIRECT"][0][24, 0] == 30.
    assert predictions["DIRECT"][1][29, 0] == 35.
    assert diagnostics["CALIBRATED"]["assimilated_releases"] == 2
    assert diagnostics["CALIBRATED"]["suppressed_releases"] == 1


def test_sparse_hourly_calibration_is_rejected(tmp_path):
    campaign = tmp_path/"cal"; campaign.mkdir()
    (campaign/"manifest.json").write_text(json.dumps({"stage": "calibration", "seeds": list(range(8)), "cells": ["clean"]}))
    folder = campaign/"jobs"/"0"/"clean"; folder.mkdir(parents=True)
    (folder/"release.json").write_text(json.dumps({"dense_query_calibration_only": True}))
    sparse = np.full((60, 1), np.nan); sparse[::24] = 20.
    np.savez(folder/"release.npz", truth=np.full((60, 1), 20.), baseline=np.full((60, 1), 20.), residual_query_dense=sparse)
    with pytest.raises(ValueError, match="sparse24h"):
        freeze_dense(campaign, tmp_path/"output")


def test_public_calibrated_never_receives_release_information():
    from dataclasses import replace
    cal = ReleaseCalibration(np.array([2., 3.]), np.eye(2)*.9, np.eye(2), np.eye(2), 1)
    public = np.full((30, 1), 20.)
    original = [PublicReleaseObservation("r", 0, 24, 0, 30., 7.)]
    changed = [replace(original[0], value=1000., laplace_scale=100.)]
    a, diagnostic_a = predict_release_only(public, original, cal)
    b, diagnostic_b = predict_release_only(public, changed, cal)
    empty, _ = predict_release_only(public, [], cal)
    for clock in range(2):
        np.testing.assert_array_equal(a["PUBLIC_CALIBRATED"][clock], b["PUBLIC_CALIBRATED"][clock])
        np.testing.assert_array_equal(a["PUBLIC_CALIBRATED"][clock], empty["PUBLIC_CALIBRATED"][clock])
        np.testing.assert_array_equal(a["PUBLIC_CALIBRATED"][clock], public+2.)
    assert not np.array_equal(a["CALIBRATED"][1], b["CALIBRATED"][1])
    for diagnostic in (diagnostic_a, diagnostic_b):
        assert diagnostic["PUBLIC_CALIBRATED"]["assimilated_releases"] == 0
        assert diagnostic["PUBLIC_CALIBRATED"]["suppressed_releases"] == 0

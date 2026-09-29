from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from airproof.records import Observation
from airproof.v6_calibration import (
    PublicLOSOResidual,
    fit_public_context_calibrator,
    frozen_backbone_scale_context,
    produce_frozen_backbone_loso,
    produce_public_loso_residuals,
)


def rows(n=120):
    return [PublicLOSOResidual(2., 0., .05, float(i), float(i), float(i), float(i),
                              "held", ("other",), str(i), "train-A", "public-fixed")
            for i in range(n)]


def fit(data=None, **kwargs):
    return fit_public_context_calibrator(rows() if data is None else data, training_end=200.,
        training_split_id="train-A", public_model_id="public-fixed", **kwargs)


def predict(model, age=0., geometry=.05, **kwargs):
    return model.predict(age, geometry, prediction_epoch=kwargs.pop("prediction_epoch", 201.),
                         available_at=kwargs.pop("available_at", 201.),
                         context_available_at=kwargs.pop("context_available_at", 201.), **kwargs)


def test_frozen_context_radii_and_broadcast():
    model = fit()
    out = predict(model, np.zeros((3, 1)), np.full((1, 4), .05))
    assert out.innovation_scales.shape == (3, 4)
    assert out.radii.shape == (3, 4, 2)
    np.testing.assert_array_equal(out.innovation_scales, 2.)
    np.testing.assert_array_equal(out.radii, 2.)
    assert not out.pooled_fallback.any()
    assert np.all(out.calibration_counts == 120)
    with pytest.raises(FrozenInstanceError):
        model.training_end = 0
    with pytest.raises(ValueError):
        out.innovation_scales[0, 0] = 4


def test_stratum_100_boundary_unseen_and_pooled_count():
    data = rows(199)
    data = [replace(r, reference_age=0. if i < 100 else 3., residual=2. if i < 100 else 8.)
            for i, r in enumerate(data)]
    model = fit(data)
    assert len(model.strata) == 1
    out = predict(model, [0., 3., 100.])
    np.testing.assert_array_equal(out.pooled_fallback, [False, True, True])
    np.testing.assert_array_equal(out.calibration_counts, [100, 199, 199])
    assert out.innovation_scales[1] == pytest.approx(np.sqrt((100*4+99*64)/199))
    assert model.pooled.radii == (8., 8.)


@pytest.mark.parametrize("change", [
    {"role": "test"}, {"source_class": "citizen"}, {"split_id": "confirmation"},
    {"public_model_id": "tuned-on-test"}, {"predictor_station_ids": ("held",)},
    {"latest_predictor_input_at": 1.}, {"label_available_at": 201.},
    {"prediction_available_at": 1.}, {"label_available_at": -1.},
])
def test_rejects_leaking_training_metadata(change):
    data = rows()
    data[0] = replace(data[0], **change)
    with pytest.raises(ValueError):
        fit(data)


def test_duplicates_do_not_inflate_calibration_support():
    data = rows()
    with pytest.raises(ValueError):
        fit(data+[data[0]])
    with pytest.raises(ValueError):
        fit(data+[replace(data[0], sample_id="different-id")])


def test_temporal_boundary_and_future_context_guards():
    model = fit()
    predict(model, prediction_epoch=200., available_at=200., context_available_at=200.)
    for kwargs in ({"available_at": 199.}, {"context_available_at": 202.},
                   {"prediction_epoch": 119.}, {"available_at": 202.}):
        with pytest.raises(ValueError):
            predict(model, **kwargs)


def test_floor_empty_support_and_locked_minimum():
    model = fit([replace(r, residual=0.) for r in rows()])
    assert predict(model).innovation_scales == 1e-3
    with pytest.raises(ValueError):
        fit(rows(19))
    with pytest.raises(ValueError):
        fit(minimum_stratum_size=99)


def test_causal_loso_producer_excludes_entire_held_station_and_future():
    def ref(cell, epoch, value, arrival=None):
        return Observation(cell, epoch, cell, 0, value, 1., 1., 512, f"{cell}:{epoch}",
                           epoch if arrival is None else arrival, epoch, source_class="regulatory")
    data = [ref(0, 0, 10.), ref(1, 0, 20.), ref(0, 1, 11.), ref(1, 1, 1e6, 3)]
    kwargs = {"training_end": 1, "training_split_id": "train"}
    rows, diagnostics = produce_public_loso_residuals(data, [[0., 0.], [1., 0.]], **kwargs)
    target = next(row for row in rows if row.station_id == "0" and row.epoch == 1)
    assert target.residual == -9.
    assert target.reference_age == 1.
    assert target.predictor_station_ids == ("1",)
    assert diagnostics["excluded_after_training_end"] == 1
    changed = [replace(item, value=1e9) if item.cell == 0 and item.epoch == 0 else item for item in data]
    changed_rows, _ = produce_public_loso_residuals(changed, [[0., 0.], [1., 0.]], **kwargs)
    changed_target = next(row for row in changed_rows if row.station_id == "0" and row.epoch == 1)
    assert target == changed_target


def test_frozen_backbone_matches_existing_loso_objective_and_no_future_scale_leakage():
    from airproof.reference import calibrated_public_reference

    data = [Observation(cell, epoch, cell, 0, 10+cell+.1*epoch, 1., 1., 512,
                        f"{cell}:{epoch}", epoch, epoch, source_class="regulatory")
            for epoch in range(10) for cell in (0, 2, 6, 8)]
    backbone = calibrated_public_reference(data, side=3, steps=10, calibration_epochs=8)
    selected = backbone.diagnostics["selected"]
    rows, _ = produce_frozen_backbone_loso(data, side=3, steps=8, training_end=7,
        training_split_id="public-prefix", public_model_id="frozen-v4", selected=selected)
    mse = np.mean([row.residual**2 for row in rows])
    assert mse == pytest.approx(selected["public_loso_mse"], abs=1e-12)
    kwargs = {"side": 3, "calibration_steps": 8, "horizon": 10, "selected": selected,
              "training_split_id": "public-prefix", "public_model_id": "frozen-v4"}
    scales, provenance = frozen_backbone_scale_context(data, **kwargs)
    altered = [replace(o, value=1e9) if o.epoch >= 8 else o for o in data]
    changed, _ = frozen_backbone_scale_context(altered, **kwargs)
    np.testing.assert_array_equal(scales, changed)
    assert scales.shape == (10, 9)
    assert provenance["first_valid_epoch"] == 8
    assert provenance["calibration_count"] == 28
    with pytest.raises(ValueError):
        frozen_backbone_scale_context(data[:-1], **kwargs)


def test_public_calibration_runner_freezes_source_and_refuses_overwrite(tmp_path):
    import json
    from dataclasses import asdict

    from scripts.run_v6_public_calibration import run

    source = tmp_path / "rows.json"
    source.write_text(json.dumps({"role": "component_diagnostic", "records": [asdict(r) for r in rows()],
        "metadata": {"training_end": 200., "training_split_id": "train-A", "public_model_id": "public-fixed"}}))
    output = run(source, tmp_path / "output")
    report = json.loads(output.read_text())
    assert not report["selection_performed"]
    assert len(report["bounded_candidate_family"]) == 3
    assert all(cfg["cap"] == 8 for cfg in report["bounded_candidate_family"].values())
    assert len(report["input_sha256"]) == 64
    with pytest.raises(FileExistsError):
        run(source, tmp_path / "output")

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beijing_shared_block_ratios_preserve_paired_methods_and_missingness():
    module = load_script("analyze_beijing_v4.py")
    baseline = np.random.default_rng(18).uniform(.1, 4, (48, 4))
    baseline[2, 1] = np.nan
    result = module.transfer_intervals({"idw_frozen": baseline, "airproof_v4_city": .81 * baseline},
                                       draws=100, blocks=(6, 12))
    for estimates in (result["paired_station_bootstrap"], *result["shared_time_block_bootstrap"].values()):
        for key in ("estimate", "lower95", "upper95"):
            assert estimates["airproof_v4_city"][key] == pytest.approx(.9)
    changed_support = baseline.copy()
    changed_support[2, 1] = 1
    with pytest.raises(ValueError, match="same target support"):
        module.transfer_intervals({"idw_frozen": baseline, "airproof_v4_city": changed_support}, draws=100, blocks=(6,))


def test_privacy_paired_effect_is_not_ratio_of_unpaired_observations():
    module = load_script("analyze_privacy_grid_v4.py")
    raw = np.array([1., 10., 100.])
    result = module.paired_effect(raw, .2 * raw, draws=100)
    assert result["mean_paired_rmse_reduction"] == pytest.approx(.8)
    assert result["lower95"] == pytest.approx(.8)
    assert result["upper95"] == pytest.approx(.8)
    with pytest.raises(ValueError):
        module.paired_effect([1, 2], [1, np.nan])


def test_registered_core_contrasts_holm_and_ratio_definitions():
    module = load_script("v4_core_analysis_common.py")
    index = {}
    seeds = [1, 2, 3]
    for seed in seeds:
        for cell in ("anchor_clean", "outage_clean", "severe_clean", "severe_adversarial_drift", "severe_hotspot_suppression"):
            attack = cell in ("severe_adversarial_drift", "severe_hotspot_suppression")
            for method in (module.AP, module.SQ, module.NF, module.NR):
                rmse = seed + 10 + (3 if attack and method == module.AP else 5 if attack else 0)
                index[seed, cell, method] = {"metrics": {"rmse": rmse,
                    "coverage_gap": .6 if method == module.AP else 1.,
                    "worst_group_rmse": 8. if method == module.AP else 10.}}
    values = module.core_contrasts(index, seeds)
    np.testing.assert_allclose(values["H1"], -.05 * np.array([11., 12., 13.]))
    np.testing.assert_allclose(values["H4"], -1.)
    np.testing.assert_allclose(values["H5"], -1.)
    np.testing.assert_allclose(values["H6"], -.2)
    np.testing.assert_allclose(values["H7"], -1.)
    assert module.holm_adjust({"A": .001, "B": .02, "C": .04}) == {"A": .003, "B": .04, "C": .04}
    ratio = module.paired_ratio([1., 4.], [2., 4.], draws=100)
    assert ratio["estimate"] == pytest.approx(5 / 6)
    unsupported = module.paired_ratio([1., 4.], [-1., 4.], draws=100)
    assert unsupported["nonpositive_bootstrap_denominators"] > 0
    assert unsupported["lower95"] is None and unsupported["upper95"] is None


def test_prospective_noncentral_t_sample_size_reaches_target_without_using_primary():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        module = load_script("lock_v4_primary_power.py")
        alpha, target = .05 / 7, .9
        count = module.required_worlds(-1., 1., alpha, target)
        assert count >= 3
        assert module.one_sided_power(-1., 1., count, alpha) >= target
        assert module.one_sided_power(-1., 1., count - 1, alpha) < target
        assert module.required_worlds(0., 1., alpha, target) is None
        assert module.required_worlds(.1, 1., alpha, target) is None
    finally:
        sys.path.remove(scripts)

import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/"reports/v5/transport_analysis_correction/tools/analyze_corrected_transport.py"
spec=importlib.util.spec_from_file_location("corrected_transport_analysis",PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("ap,expected",[(.78,0.),(.80,-.02),(.77,.01)])
def test_delivery_deficit_margin_against_epidemic_boundary(ap,expected):
    rows=[]
    for seed in range(5):
        for policy,delivery in (("airproof_deadline",ap),("epidemic_cap",.80),("binary_spray_wait",.1)):
            rows.append(dict(seed=seed,cell="unit",policy=policy,metrics=dict(deadline_delivery_ratio=delivery,
                restricted_mean_delay_seconds=10.,group_delivery_gap=.1)))
    result=module.analyze_transport_family(rows,seeds=list(range(5)),cells=["unit"])
    assert result["tests"]["D1"]["mean_contrast"] == pytest.approx(expected,abs=1e-14)
    assert result["tests"]["D2"]["mean_contrast"] == pytest.approx(1.5)
    assert result["tests"]["D3"]["mean_contrast"] == pytest.approx(.02)
    assert result["comparator"]["D1"] == "epidemic_cap"
    assert not result["family_pass"]


def varied_rows(constant=False):
    rows=[]
    for seed in range(5):
        for policy in ("airproof_deadline","epidemic_cap","binary_spray_wait"):
            ap=policy=="airproof_deadline"
            jitter=0 if constant else seed*.001
            rows.append(dict(seed=seed,cell="unit",policy=policy,metrics=dict(
                deadline_delivery_ratio=.8+jitter if ap else .8,
                restricted_mean_delay_seconds=50+seed*(not constant) if ap else 100.,
                group_delivery_gap=.1+jitter if ap else .3)))
    return rows


def test_negative_zero_variance_family_is_invalid_not_favorable():
    result=module.analyze_transport_family(varied_rows(True),seeds=list(range(5)),cells=["unit"])
    assert not result["family_valid"] and not result["family_pass"]
    for item in result["tests"].values():
        assert not item['valid'] and item['one_sided_p'] is None
        assert item['upper_one_sided95'] is None and item['upper_bonferroni_family95'] is None
        assert item['holm_p']==1. and not item['conditional_threshold_met']


def test_nonzero_variance_family_retains_ordinary_inference():
    result=module.analyze_transport_family(varied_rows(),seeds=list(range(5)),cells=["unit"])
    assert result['family_valid']
    assert all(item['valid'] and item['one_sided_p'] is not None for item in result['tests'].values())
    assert all(item['conditional_threshold_met'] for item in result['tests'].values())
    assert not result['family_pass']

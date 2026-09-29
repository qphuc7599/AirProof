import importlib.util
from pathlib import Path
import pytest
from airproof.v5_audit import run_same_stream_audit

spec=importlib.util.spec_from_file_location('audit_supplement',Path(__file__).parents[1]/'reports/v5/tools/audit_supplement.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_supplement_matches_original_streams_and_verifies_real_terminal():
    old=run_same_stream_audit(batch_sizes=(8,),trials=2)
    rows,roots=module.run_supplement(batch_sizes=(8,),trials=2)
    assert len(rows)==12 and len(roots)==2
    for row in rows:
        assert row['stream_sha256']==next(r['stream_sha256'] for r in old if r['trial']==row['trial'])
        assert row['clean_control_valid'] and row['clean_terminal_valid']
        if row['fault']=='forged_terminal_witness':
            assert row['forged_signature_valid'] and row['detected'] and row['detection_epoch']==11

def test_delayed_anchor_is_not_detected_early_and_other_policies_censor():
    rows,_=module.run_supplement(batch_sizes=(8,),trials=1)
    for row in rows:
        if row['fault']=='delayed_anchor':
            assert not row['detected_before_anchor']
            if row['policy']=='anchored_transparency_log':
                assert row['detection_epoch']==31 and row['time_from_fault']==21
            else:
                assert row['right_censored'] and row['detection_epoch'] is None and row['restricted_detection_time']==50

def test_anchor_beyond_horizon_does_not_become_detection():
    rows,_=module.run_supplement(batch_sizes=(8,),trials=1,anchor_at=70,readable_at=71)
    assert all(r['right_censored'] for r in rows if r['fault']=='delayed_anchor')
    with pytest.raises(ValueError):
        module.run_supplement(batch_sizes=(8,),trials=1,anchor_at=30,readable_at=20)

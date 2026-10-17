import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).parents[1]
OUT=ROOT/'reports/v6/uncertainty_development_validation_v2'


def test_uncertainty_artifact_provenance_and_hashes():
    registration=(ROOT/'configs/v6_uncertainty_development_validation_v2.json').read_bytes()
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert (OUT/'registration.json').read_bytes()==registration
    assert manifest['registration_sha256']==hashlib.sha256(registration).hexdigest()
    assert manifest['citizen_estimator_selected'] is None
    assert manifest['confirmation_authorized'] is False
    assert manifest['event_threshold_quantile']==.95
    assert manifest['event_threshold_source']=='development-6201000-6201003-anchor-clean'
    assert (ROOT/'reports/v6/uncertainty_development_validation/INVALID.md').exists()
    assert len(manifest['input_hashes'])==24 and len(manifest['input_identities'])==24
    for name,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest


def test_clock_specific_calibration_and_registered_strata():
    calibration=json.loads((OUT/'calibrations.json').read_text())
    assert set(calibration)=={'live','reconstructed'}
    for clock,age in (('live','age=0'),('reconstructed','age=6')):
        row=calibration[clock]
        assert row['clock']==clock and row['count']==638976
        assert len(row['pooled_radii'])==2 and row['pooled_radii'][0]<=row['pooled_radii'][1]
        assert row['fallback_strata']==[] and len(row['stratum_counts'])==12
        assert all(age in label and count>=100 for label,count in row['stratum_counts'])


def test_uncertainty_validation_retains_event_undercoverage_and_descriptive_scope():
    results=json.loads((OUT/'results.json').read_text())
    assert set(results)=={'anchor_clean','outage_clean','severe_clean','severe_drift','severe_hotspot'}
    reference=None
    for cell,clocks in results.items():
        assert set(clocks)=={'live','reconstructed'}
        for report in clocks.values():
            assert report['levels']==[.9,.95]
            assert report['bootstrap_replicates']==50 and report['block_length']==24
            assert report['inference_scope'].startswith('descriptive')
            assert 'no exact conformal' in report['coverage_claim']
            overall=report['groups']['all'];event=report['groups']['event']
            assert overall['coverage']['estimate'][0]>.9
            assert event['coverage']['estimate'][0]<.6
            assert event['coverage']['estimate'][1]<.7
            assert overall['width']['estimate'][0]>0
            assert overall['interval_score']['estimate'][0]>=overall['width']['estimate'][0]
        pair=(clocks['live']['groups']['all']['coverage']['estimate'],
              clocks['reconstructed']['groups']['all']['coverage']['estimate'])
        assert pair[0]==pair[1]
        reference=pair if reference is None else reference
        assert pair==reference

import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'v6'/'integration_smoke_accountability_batch_v3'


def test_v3_batch_smoke_is_registered_integral_and_retains_adverse_comparison():
    registration=(ROOT/'configs'/'v6_integration_smoke_accountability_batch_v3.json').read_bytes()
    frozen=(OUT/'registration.json').read_bytes();manifest=json.loads((OUT/'manifest.json').read_text())
    outcome=json.loads((OUT/'outcome.json').read_text())
    assert frozen==registration
    assert hashlib.sha256(registration).hexdigest()==manifest['registration_sha256']
    assert hashlib.sha256((OUT/'outcome.json').read_bytes()).hexdigest()==manifest['outcome_sha256']
    for name in ('configs/v6_integration_smoke_accountability_batch_v3.json',
                 'scripts/run_v6_integration_smoke_accountability_batch_v3.py','airproof/v6_transport.py'):
        snapshot=OUT/'source_snapshot'/name
        source=snapshot if snapshot.is_file() else ROOT/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_hashes'][name]
    assert all(len(digest)==64 for digest in manifest['source_hashes'].values())
    assert outcome['comparators']=={'legacy_engineering_selected':181,
        'per_record_accountability_selected':39,'stale_gate_per_record_selected':50}
    assert outcome['selected']==45 and outcome['origin_verified']==32
    metrics=outcome['transport']
    assert metrics['raw_delivered']==289 and metrics['raw_timely_delivered']==181
    assert metrics['audit_stale_rejected']==108 and metrics['audit_capacity_rejected']==136
    assert sum(metrics['audit_flushed_batch_sizes'])==45
    assert metrics['audit_terminal_objects_issued']==len(metrics['audit_flushed_batch_sizes'])==32
    assert metrics['receipt_payload_bytes']==64000 and metrics['audit_return_control_bytes']==2000
    assert outcome['finite_estimates']

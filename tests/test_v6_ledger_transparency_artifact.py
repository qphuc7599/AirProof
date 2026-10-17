import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'v6'/'ledger_transparency_same_stream_v2'

def test_v2_same_stream_artifact_is_complete_integral_and_retains_misses():
    reg=(ROOT/'configs'/'v6_ledger_transparency_same_stream_v2.json').read_bytes()
    manifest=json.loads((OUT/'manifest.json').read_text());analysis=json.loads((OUT/'analysis.json').read_text())
    assert (OUT/'registration.json').read_bytes()==reg
    assert hashlib.sha256(reg).hexdigest()==manifest['registration_sha256']
    assert hashlib.sha256((OUT/'results.jsonl').read_bytes()).hexdigest()==manifest['results_sha256']
    assert hashlib.sha256((OUT/'analysis.json').read_bytes()).hexdigest()==manifest['analysis_sha256']
    for path,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert analysis['complete'] and analysis['rows']==analysis['expected_rows']==8000
    assert analysis['all_clean_controls_valid'] and analysis['paired_stream_invariant']
    assert analysis['interface_regression']['pass']
    partitions=[g for g in analysis['groups'] if g['fault']=='auditor_partition']
    assert len(partitions)==4 and all(g['detected']==0 and g['right_censored']==100 for g in partitions)
    assert all(g['detected']==g['trials'] for g in analysis['groups'] if g['fault']!='auditor_partition')
    assert not analysis['v5_reuse_disposition']['compatible']
    assert (ROOT/'reports'/'v6'/'ledger_transparency_same_stream_v1'/'INVALID.md').exists()
